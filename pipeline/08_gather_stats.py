"""
Params / checkpoint size for any model in the MODELS registry. Usage:
    python 08_gather_stats.py teacher
    python 08_gather_stats.py c3
"""
import os, sys, json
import torch
exec(open(os.path.join(os.path.dirname(__file__), "00_settings.py")).read())

which = sys.argv[1] if len(sys.argv) > 1 else "teacher"
assert which in MODELS, f"usage: python 08_gather_stats.py [{'|'.join(MODELS)}]"

model_dir_ = get_model_dir(which)
ckpt_path = f"{model_dir_}/fold_{FOLD}/checkpoint_final.pth"

ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
weights = {k: v for k, v in ck["network_weights"].items() if v.is_floating_point()}
n_params = sum(v.numel() for v in weights.values())
size_mb = os.path.getsize(ckpt_path) / 1e6

print(f"{which.capitalize()} parameters       : {n_params/1e6:.2f} M")
print(f"Checkpoint file size     : {size_mb:.0f} MB")
print(f"Weights only, float32    : {n_params*4/1e6:.0f} MB")

stats = {
    f"{which}_params_million": n_params / 1e6,
    "checkpoint_mb": size_mb,
    "weights_fp32_mb": n_params * 4 / 1e6,
    "trainer": MODELS[which]["trainer"], "plans": MODELS[which]["plans"], "config": CONFIG,
}
out = f"{WORK_DIR}/report_assets"
os.makedirs(out, exist_ok=True)
json.dump(stats, open(f"{out}/{which}_stats.json", "w"), indent=2)
print("Saved", f"{out}/{which}_stats.json")
