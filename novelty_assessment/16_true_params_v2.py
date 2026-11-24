"""
Counts the REAL number of parameters of every model by building the network from its plans.json
(exactly what the ONNX export does) -- no double counting. Writes report_assets/true_params.json.

    !python /content/drive/MyDrive/cas2023/pipeline/16_true_params_v2.py
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import novelty_env_v2  # noqa: F401
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_settings.py")).read())
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

def true_params(model_dir_):
    d = Path(model_dir_)
    pm = PlansManager(json.load(open(d / "plans.json")))
    dj = json.load(open(d / "dataset.json"))
    cfg = pm.get_configuration(CONFIG)
    n_in = determine_num_input_channels(pm, cfg, dj)
    net = get_network_from_plans(cfg.network_arch_class_name, cfg.network_arch_init_kwargs,
                                 cfg.network_arch_init_kwargs_req_import, n_in,
                                 pm.get_label_manager(dj).num_segmentation_heads,
                                 allow_init=True, deep_supervision=False)
    return sum(p.numel() for p in net.parameters()) / 1e6, cfg.network_arch_init_kwargs["features_per_stage"]

targets = {
    "teacher": TEACHER_MODEL_DIR,
    "half_width_old": model_dir(TRAINER, "nnUNetPlansStudent"),          # the earlier scale-0.5 student
    f"student{RUN_TAG}": model_dir(TRAINER, STUDENT_PLANS_NAME),          # current experiment
}
out = {}
for name, path in targets.items():
    if not (Path(path) / "plans.json").exists():
        print(f"{name:<18} (no plans.json at {path}, skipped)"); continue
    n, feats = true_params(path)
    out[name] = n
    print(f"{name:<18} {n:8.2f} M   features_per_stage={feats}")
os.makedirs(f"{WORK_DIR}/report_assets", exist_ok=True)
old = {}
p = f"{WORK_DIR}/report_assets/true_params.json"
if os.path.exists(p): old = json.load(open(p))
old.update(out); json.dump(old, open(p, "w"), indent=2); print("Saved", p)
