"""Export the selected warm-start C3 student to ONNX and verify it.

Run in a Colab runtime after ``00b_install_trainers.py``:
    python 11_export_selected_c3_onnx.py

The export uses the student-plan patch size, writes the ONNX file to Drive,
and checks that ONNX Runtime logits closely match PyTorch logits.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


PROJECT_ROOT = Path(os.environ.get("CAS2023_ROOT", "/content/drive/MyDrive/cas2023"))
MODEL_DIR = PROJECT_ROOT / "nnUNet_results" / "Dataset501_CAS2023" / "nnUNetTrainerKD_C3_Combined__nnUNetPlansStudent__3d_fullres"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"


def selected_warm_start_fold() -> Path:
    """Support both the preserved and restored warm-start C3 folder layouts."""
    preserved = MODEL_DIR / "fold_0_warmstart"
    active = MODEL_DIR / "fold_0"
    if preserved.is_dir():
        return preserved
    if active.is_dir():
        return active
    raise RuntimeError(f"No C3 checkpoint folder found in {MODEL_DIR}")


def weights_for_model(checkpoint: dict, model: torch.nn.Module) -> dict:
    source = checkpoint.get("network_weights", checkpoint)
    target_keys = set(model.state_dict())
    matched = {}
    for key, value in source.items():
        plain_key = key.removeprefix("module.").removeprefix("_orig_mod.")
        candidates = (key, plain_key, f"_orig_mod.{plain_key}")
        destination = next((candidate for candidate in candidates if candidate in target_keys), None)
        if destination is not None:
            matched[destination] = value
    return matched


def build_model() -> tuple[torch.nn.Module, tuple[int, ...]]:
    plans_path = MODEL_DIR / "plans.json"
    dataset_path = MODEL_DIR / "dataset.json"
    with open(plans_path) as stream:
        plans_manager = PlansManager(json.load(stream))
    with open(dataset_path) as stream:
        dataset_json = json.load(stream)
    configuration = plans_manager.get_configuration("3d_fullres")
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_input_channels = determine_num_input_channels(plans_manager, configuration, dataset_json)
    model = get_network_from_plans(
        configuration.network_arch_class_name,
        configuration.network_arch_init_kwargs,
        configuration.network_arch_init_kwargs_req_import,
        num_input_channels,
        label_manager.num_segmentation_heads,
        allow_init=True,
        deep_supervision=False,
    ).cpu().eval()
    checkpoint = torch.load(selected_warm_start_fold() / "checkpoint_final.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(weights_for_model(checkpoint, model), strict=True)
    return model, tuple(int(value) for value in configuration.patch_size)


def verify_onnx(model: torch.nn.Module, example: torch.Tensor, onnx_path: Path) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Install ONNX verification dependencies with: pip install onnx onnxruntime") from error
    with torch.no_grad():
        pytorch_logits = model(example).cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"image": example.cpu().numpy()})[0]
    difference = np.abs(pytorch_logits - onnx_logits)
    maximum_error = float(difference.max())
    mean_error = float(difference.mean())
    if not np.allclose(pytorch_logits, onnx_logits, rtol=1e-3, atol=3e-3):
        raise RuntimeError(f"ONNX verification failed: max error={maximum_error:.6g}, mean error={mean_error:.6g}")
    return {"max_abs_error": maximum_error, "mean_abs_error": mean_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEPLOYMENT_DIR / "c3_warmstart_3d_fullres.onnx")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, patch_size = build_model()
    example = torch.randn(1, 1, *patch_size, dtype=torch.float32)
    print(f"Exporting C3 warm-start model with input shape {tuple(example.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            str(args.output),
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
        )
    verification = verify_onnx(model, example, args.output)
    metadata = {
        "model": "C3 combined KD warm-start",
        "checkpoint": str(selected_warm_start_fold() / "checkpoint_final.pth"),
        "input_name": "image",
        "output_name": "logits",
        "input_shape": list(example.shape),
        "output_shape": list(model(example).shape),
        "onnx_file": str(args.output),
        "onnx_size_mb": round(args.output.stat().st_size / (1024**2), 2),
        "verification": verification,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved ONNX model: {args.output}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Verified ONNX output: max abs error {verification['max_abs_error']:.3g}")


if __name__ == "__main__":
    main()
