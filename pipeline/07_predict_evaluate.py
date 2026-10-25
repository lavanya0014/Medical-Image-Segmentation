"""
Run inference on the held-out test set and evaluate, for any model in the
MODELS registry (see 00_settings.py): teacher, student, c1, c2, c3. Usage:
    python 07_predict_evaluate.py teacher
    python 07_predict_evaluate.py c2
"""
import os, sys, json, shutil, subprocess
exec(open(os.path.join(os.path.dirname(__file__), "00_settings.py")).read())

which = sys.argv[1] if len(sys.argv) > 1 else "teacher"
assert which in MODELS, f"usage: python 07_predict_evaluate.py [{'|'.join(MODELS)}]"

cfg = MODELS[which]
model_dir_ = get_model_dir(which)
pred_dir = f"{WORK_DIR}/{which}_predictions"
os.makedirs(pred_dir, exist_ok=True)

predict_cmd = ["nnUNetv2_predict", "-i", f"{RAW_DS}/imagesTs", "-o", pred_dir,
               "-d", str(DATASET_ID), "-c", CONFIG, "-f", str(FOLD), "-tr", cfg["trainer"],
               "-p", cfg["plans"], "--save_probabilities"]
print("Running:", " ".join(predict_cmd))
subprocess.run(predict_cmd, check=True)

evaluate_cmd = ["nnUNetv2_evaluate_folder", f"{RAW_DS}/labelsTs", pred_dir,
                "-djfile", f"{model_dir_}/dataset.json", "-pfile", f"{model_dir_}/plans.json"]
print("Running:", " ".join(evaluate_cmd))
subprocess.run(evaluate_cmd, check=True)

with open(f"{pred_dir}/summary.json") as f:
    summary = json.load(f)
print(f"\n{which} mean over test cases (vessel class):")
for k, v in summary["foreground_mean"].items():
    if k in ("Dice", "IoU"):
        print(f"  {k:5s}: {v:.4f}")

out = f"{WORK_DIR}/report_assets"
os.makedirs(out, exist_ok=True)
shutil.copy(f"{pred_dir}/summary.json", f"{out}/{which}_test_summary.json")
print("Saved", f"{out}/{which}_test_summary.json")
