"""
Train the student (same trainer/fold/epochs as the teacher, student plan).
Always passes --c -> safe to re-run after every disconnect.

Same local-disk-copy trick as the teacher's training script, for the same
reason: avoids Drive FUSE drops under heavy concurrent read load from the
data loader workers.
"""
import os, shutil, subprocess
exec(open(os.path.join(os.path.dirname(__file__), "00_settings.py")).read())

LOCAL_PREPROCESSED_ROOT = "/content/local_nnUNet_preprocessed"
local_ds = os.path.join(LOCAL_PREPROCESSED_ROOT, DS_FOLDER)

print(f"Syncing preprocessed data to local disk ({PP_DS} -> {local_ds}) ...")
verified_copytree(PP_DS, local_ds)
print("Sync done.")

env = os.environ.copy()
env["nnUNet_preprocessed"] = LOCAL_PREPROCESSED_ROOT

cmd = ["nnUNetv2_train", str(DATASET_ID), CONFIG, str(FOLD), "-tr", TRAINER,
       "-p", STUDENT_PLANS_NAME, "--c"]
print("Running:", " ".join(cmd), "(nnUNet_preprocessed =", LOCAL_PREPROCESSED_ROOT, ")")
try:
    subprocess.run(cmd, check=True, env=env)
finally:
    split_file = os.path.join(local_ds, "splits_final.json")
    if os.path.exists(split_file):
        shutil.copy(split_file, os.path.join(PP_DS, "splits_final.json"))

print("Student training step finished (or already complete).")
