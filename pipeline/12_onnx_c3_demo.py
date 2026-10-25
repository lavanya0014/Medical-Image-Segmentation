"""Small web demo for the exported C3 ONNX model.

This is a presentation/demo interface, not a replacement for nnU-Net's full
preprocessing and sliding-window inference pipeline. It center-crops or pads a
single 3-D NIfTI scan to the ONNX input patch size and applies z-score
normalization. Use it with representative, pre-cropped scans for a faithful
demonstration.

Run in Colab:
    !pip -q install gradio nibabel onnxruntime scipy
    !python /content/drive/MyDrive/cas2023/pipeline/12_onnx_c3_demo.py --share
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gradio as gr
import nibabel as nib
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path("/content/drive/MyDrive/cas2023")
DEFAULT_MODEL = PROJECT_ROOT / "deployment" / "c3_warmstart_3d_fullres.onnx"
DEFAULT_OUTPUT = PROJECT_ROOT / "deployment" / "demo_outputs"


def load_nifti(path: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(path)
    data = image.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise gr.Error("Upload one 3-D, single-modality NIfTI scan (.nii or .nii.gz).")
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, image


def normalise(volume: np.ndarray) -> np.ndarray:
    foreground = volume[volume != 0]
    values = foreground if foreground.size else volume.reshape(-1)
    mean = float(values.mean())
    std = float(values.std())
    return (volume - mean) / max(std, 1e-6)


def crop_or_pad(volume: np.ndarray, target_shape: tuple[int, int, int]) -> tuple[np.ndarray, tuple[slice, ...], tuple[slice, ...]]:
    output = np.zeros(target_shape, dtype=np.float32)
    source_slices: list[slice] = []
    destination_slices: list[slice] = []
    for source_length, target_length in zip(volume.shape, target_shape):
        if source_length >= target_length:
            source_start = (source_length - target_length) // 2
            source_slices.append(slice(source_start, source_start + target_length))
            destination_slices.append(slice(0, target_length))
        else:
            destination_start = (target_length - source_length) // 2
            source_slices.append(slice(0, source_length))
            destination_slices.append(slice(destination_start, destination_start + source_length))
    output[tuple(destination_slices)] = volume[tuple(source_slices)]
    return output, tuple(source_slices), tuple(destination_slices)


def restore_shape(segmentation: np.ndarray, original_shape: tuple[int, int, int], source_slices: tuple[slice, ...], destination_slices: tuple[slice, ...]) -> np.ndarray:
    restored = np.zeros(original_shape, dtype=np.uint8)
    restored[source_slices] = segmentation[destination_slices].astype(np.uint8)
    return restored


def preview(image: np.ndarray, segmentation: np.ndarray, slice_index: int) -> np.ndarray:
    """Render an axial slice.

    NIfTI volumes conventionally store the head-foot (slice) direction in the
    final axis. Using axis zero showed a thin side view for this dataset.
    """
    slice_index = int(np.clip(slice_index, 0, image.shape[2] - 1))
    source = image[:, :, slice_index].T
    low, high = np.percentile(source, (1, 99))
    gray = np.clip((source - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.repeat((gray[..., None] * 255).astype(np.uint8), 3, axis=-1)
    mask = segmentation[:, :, slice_index].T > 0
    rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.array([235, 55, 55])).astype(np.uint8)
    return rgb


def make_demo(model_path: Path, output_dir: Path) -> gr.Blocks:
    metadata_path = model_path.with_suffix(".json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        target_shape = tuple(int(value) for value in metadata["input_shape"][-3:])
    else:
        target_shape = (64, 192, 192)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_inference(scan_path: str, label_path: str | None) -> tuple[Any, str, str, dict[str, np.ndarray], Any]:
        if not scan_path:
            raise gr.Error("Upload a NIfTI scan first.")
        volume, original = load_nifti(scan_path)
        model_input, source_slices, destination_slices = crop_or_pad(normalise(volume), target_shape)
        logits = session.run(None, {input_name: model_input[None, None].astype(np.float32)})[0]
        fixed_segmentation = np.argmax(logits[0], axis=0).astype(np.uint8)
        full_segmentation = restore_shape(fixed_segmentation, volume.shape, source_slices, destination_slices)
        output_path = output_dir / f"{Path(scan_path).stem.replace('.nii', '')}_c3_prediction.nii.gz"
        nib.save(nib.Nifti1Image(full_segmentation, original.affine, original.header), output_path)
        message = (
            f"**Prediction complete.** Model input: `{target_shape}`; output shape: `{tuple(full_segmentation.shape)}`. "
            "This demo uses center crop/pad + z-score normalization; it is intended for pre-cropped presentation scans."
        )
        if label_path:
            label, _ = load_nifti(label_path)
            if label.shape != volume.shape:
                raise gr.Error("The optional label must have exactly the same shape as the input scan.")
            prediction_mask = full_segmentation > 0
            label_mask = label > 0
            dice = (2 * np.logical_and(prediction_mask, label_mask).sum()) / max(prediction_mask.sum() + label_mask.sum(), 1)
            message += f" Foreground Dice for this uploaded label: **{dice:.4f}**."
        middle_slice = volume.shape[2] // 2
        state = {"image": volume, "segmentation": full_segmentation}
        return preview(volume, full_segmentation, middle_slice), str(output_path), message, state, gr.update(maximum=volume.shape[2] - 1, value=middle_slice)

    def update_preview(slice_index: int, state: dict[str, np.ndarray] | None) -> np.ndarray | None:
        if not state:
            return None
        return preview(state["image"], state["segmentation"], slice_index)

    with gr.Blocks(title="CAS2023 C3 KD ONNX Demo") as demo:
        gr.Markdown(
            "# CAS2023 C3 Knowledge-Distillation Demo\n"
            "Upload a single 3-D NIfTI scan to run the selected C3 warm-start ONNX model. "
            "For the presentation, use a representative, pre-cropped scan; this page does not reproduce full nnU-Net preprocessing or sliding-window inference."
        )
        with gr.Row():
            scan = gr.File(label="CT angiography scan (.nii/.nii.gz)", file_types=[".nii", ".gz"], type="filepath")
            label = gr.File(label="Optional ground-truth label", file_types=[".nii", ".gz"], type="filepath")
        run_button = gr.Button("Run C3 ONNX inference", variant="primary")
        image_preview = gr.Image(label="Axial slice: red = predicted vessel", type="numpy")
        slice_slider = gr.Slider(label="Axial slice", minimum=0, maximum=1, step=1, value=0)
        report = gr.Markdown()
        download = gr.File(label="Download predicted segmentation")
        state = gr.State()
        run_button.click(run_inference, inputs=[scan, label], outputs=[image_preview, download, report, state, slice_slider])
        slice_slider.change(update_preview, inputs=[slice_slider, state], outputs=image_preview)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio URL (needed in Colab).")
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")
    make_demo(args.model, args.output_dir).launch(share=args.share)


if __name__ == "__main__":
    main()
