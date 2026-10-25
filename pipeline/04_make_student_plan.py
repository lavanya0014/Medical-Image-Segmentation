"""
Build the student's plan: same architecture/stages as the teacher, but with
`features_per_stage` scaled down (STUDENT_WIDTH_SCALE, default 0.5 -> ~4x fewer
conv params). This is the key field nnU-Net actually reads to build the network --
there is no separate "base_num_features" field in current nnU-Net plans.
"""
import os, json
exec(open(os.path.join(os.path.dirname(__file__), "00_settings.py")).read())

with open(os.path.join(PP_DS, "nnUNetPlans.json")) as f:
    plans = json.load(f)

plans["plans_name"] = STUDENT_PLANS_NAME
for cfg in plans["configurations"].values():
    arch = cfg.get("architecture", {})
    kwargs = arch.get("arch_kwargs", {})
    if "features_per_stage" in kwargs:
        before = kwargs["features_per_stage"]
        kwargs["features_per_stage"] = [max(8, int(round(f * STUDENT_WIDTH_SCALE))) for f in before]
        print(f"features_per_stage: {before} -> {kwargs['features_per_stage']}")

out_path = os.path.join(PP_DS, f"{STUDENT_PLANS_NAME}.json")
with open(out_path, "w") as f:
    json.dump(plans, f, indent=2)
print("Saved", out_path)
