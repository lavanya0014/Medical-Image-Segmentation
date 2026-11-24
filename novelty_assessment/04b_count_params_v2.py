"""
Prints the parameter count for several student width scales WITHOUT training anything,
so you can pick the scale that lands closest to 5.0M. Needs 04/05-style plans on Drive
(i.e. PP_DS/nnUNetPlans.json from the teacher preprocessing).

    !python /content/drive/MyDrive/cas2023/pipeline/04b_count_params_v2.py
"""
import os, json
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_settings.py")).read())
from copy import deepcopy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

plans = json.load(open(os.path.join(PP_DS, "nnUNetPlans.json")))
dataset_json = json.load(open(os.path.join(PP_DS, "dataset.json")))
n_in = len(dataset_json.get("channel_names", dataset_json.get("modality", {"0": "x"})))
n_out = len(dataset_json["labels"])

def count(scale):
    p = deepcopy(plans)
    for cfg in p["configurations"].values():
        k = cfg.get("architecture", {}).get("arch_kwargs", {})
        if "features_per_stage" in k:
            k["features_per_stage"] = [max(8, int(round(f * scale))) for f in k["features_per_stage"]]
    cm = PlansManager(p).get_configuration(CONFIG)
    net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                 cm.network_arch_init_kwargs_req_import, n_in, n_out,
                                 allow_init=True, deep_supervision=True)
    return sum(x.numel() for x in net.parameters()), cm.network_arch_init_kwargs["features_per_stage"]

print(f"{'scale':>6} {'params (M)':>11}  features_per_stage")
for s in (1.0, 0.5, 0.45, 0.42, 0.4, 0.38, 0.35, 0.3, 0.25):
    n, f = count(s)
    print(f"{s:>6} {n/1e6:>11.2f}  {f}")
