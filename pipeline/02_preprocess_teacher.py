"""
nnU-Net's own analysis + preprocessing, for the default (teacher) plan.
Idempotent via a marker file (nnUNetv2_plan_and_preprocess itself doesn't skip).
"""
import os, subprocess
exec(open(os.path.join(os.path.dirname(__file__), "00_settings.py")).read())

marker = os.path.join(PP_DS, ".preprocess_teacher_done")

if os.path.exists(marker):
    print("Teacher preprocessing already done - skipping.")
else:
    subprocess.run(
        ["nnUNetv2_plan_and_preprocess", "-d", str(DATASET_ID), "-c", CONFIG,
         "--verify_dataset_integrity"],
        check=True,
    )
    open(marker, "w").close()
    print("Teacher preprocessing done.")
