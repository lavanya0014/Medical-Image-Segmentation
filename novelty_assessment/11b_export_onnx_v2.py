"""
Export ANY model in the MODELS registry (student=B5, c1, c2, c3) to ONNX and verify it
against PyTorch -- same recipe as 11_export_selected_c3_onnx.py (opset 17, input 'image',
output 'logits'), just generalised.

    !python /content/drive/MyDrive/cas2023/pipeline/run_novelty_v2.py --stage export_c1
or  (after setting the env, see novelty_env.py)  python 11b_export_onnx_v2.py c1
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import novelty_env_v2  # noqa: F401
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_settings.py")).read())

import numpy as np
import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

which = sys.argv[1] if len(sys.argv) > 1 else "student"
assert which in MODELS, f"usage: python 11b_export_onnx_v2.py [{'|'.join(MODELS)}]"
model_dir_ = Path(get_model_dir(which))
out_dir = Path(WORK_DIR) / "deployment"
out_dir.mkdir(parents=True, exist_ok=True)
onnx_path = out_dir / f"{which}{RUN_TAG}_3d_fullres.onnx"

plans_manager = PlansManager(json.load(open(model_dir_ / "plans.json")))
dataset_json = json.load(open(model_dir_ / "dataset.json"))
cfg = plans_manager.get_configuration(CONFIG)
label_manager = plans_manager.get_label_manager(dataset_json)
n_in = determine_num_input_channels(plans_manager, cfg, dataset_json)
model = get_network_from_plans(cfg.network_arch_class_name, cfg.network_arch_init_kwargs,
                               cfg.network_arch_init_kwargs_req_import, n_in,
                               label_manager.num_segmentation_heads,
                               allow_init=True, deep_supervision=False).cpu().eval()

ck = torch.load(model_dir_ / f"fold_{FOLD}" / "checkpoint_final.pth", map_location="cpu", weights_only=False)
src = ck.get("network_weights", ck)
keys = set(model.state_dict())
weights = {}
for k, v in src.items():
    plain = k.removeprefix("module.").removeprefix("_orig_mod.")
    for cand in (k, plain, f"_orig_mod.{plain}"):
        if cand in keys:
            weights[cand] = v
            break
model.load_state_dict(weights, strict=True)   # strict: fails loudly if width/plan does not match
n_params = sum(p.numel() for p in model.parameters())
print(f"{which}: {n_params/1e6:.2f} M parameters")

patch = tuple(int(v) for v in cfg.patch_size)
x = torch.randn(1, n_in, *patch)
with torch.no_grad():
    torch.onnx.export(model, x, str(onnx_path), input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    ref = model(x).numpy()

import onnxruntime as ort
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
got = sess.run(["logits"], {"image": x.numpy()})[0]
err = float(np.abs(ref - got).max())
scale = float(max(1.0, np.abs(ref).max()))          # logits of KD-trained models can be large
rel = err / scale
agree = float((ref.argmax(1) == got.argmax(1)).mean())
print(f"max |PyTorch - ONNX| = {err:.3g}  (logit range +-{scale:.3g}, relative {rel:.2g}); "
      f"predicted-label agreement {agree:.6f}")
if rel > 1e-3 and agree < 0.9995:
    raise RuntimeError(f"ONNX export does not match PyTorch (abs {err:.3g}, rel {rel:.2g}, agreement {agree:.5f})")

meta = dict(model=which, tag=RUN_TAG, params_million=n_params / 1e6, patch_size=list(patch),
            onnx_file=onnx_path.name, onnx_size_mb=round(onnx_path.stat().st_size / 1024**2, 2), max_abs_error=err, relative_error=rel, label_agreement=agree)
json.dump(meta, open(onnx_path.with_suffix(".json"), "w"), indent=2)
print("Saved", onnx_path, f"({meta['onnx_size_mb']} MB)")
