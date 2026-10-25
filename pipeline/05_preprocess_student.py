"""
Preprocess (just the resampling/normalization step) for the student plan.
Reuses the already-cropped/normalized fingerprint -- fast relative to full
plan_and_preprocess, but still touches every case, so it's marker-guarded.
"""
import os, subprocess
exec(open(os.path.join(os.path.dirname(__file__), "00_settings.py")).read())

marker = os.path.join(PP_DS, f".preprocess_{STUDENT_PLANS_NAME}_done")

if os.path.exists(marker):
    print("Student preprocessing already done - skipping.")
else:
    subprocess.run(
        ["nnUNetv2_preprocess", "-d", str(DATASET_ID), "-plans_name", STUDENT_PLANS_NAME, "-c", CONFIG],
        check=True,
    )
    open(marker, "w").close()
    print("Student preprocessing done.")
