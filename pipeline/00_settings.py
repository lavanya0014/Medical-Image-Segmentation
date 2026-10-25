"""
Settings for the CAS2023 teacher/student nnU-Net pipeline.

Import this first in every script/cell:
    from importlib import import_module
    import sys; sys.path.insert(0, "/content/drive/MyDrive/cas2023/pipeline")
    import importlib, settings  (or: exec(open('00_settings.py').read()))

KEY FIX vs your original notebook: nnUNet_raw / nnUNet_preprocessed / nnUNet_results
all live under WORK_DIR on Drive (not /content). That is slower per-epoch I/O than local
disk, but it means a Colab disconnect NEVER wipes your converted data or preprocessed
data again -- you only ever lose in-flight (unsaved) training progress, and training
resumes from the last checkpoint with --c.
"""
import os

# ---- mount drive, but only if it isn't already mounted ----
# NOTE: drive.mount() only works inside the actual Colab kernel process. Since
# every pipeline stage runs as its own `!python script.py` subprocess, calling
# mount() there fails even if Drive is already mounted in the notebook. So:
# mount here ONLY if nothing is there yet (which means you're running this
# settings file directly in a notebook cell, not as a subprocess).
if not os.path.isdir('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        pass  # not running in Colab (e.g. local test)

# ================= EDIT THESE IF THEY DIFFER FROM YOUR ORIGINAL SETTINGS CELL =================
CAS_DIR    = "/content/drive/MyDrive/cas2023/CAS2023_trainingdataset"
DATASET_ID, DATASET_NAME = 501, "CAS2023"
WORK_DIR   = "/content/drive/MyDrive/cas2023"
CONFIG     = "3d_fullres"
FOLD       = 0
TRAINER    = "nnUNetTrainer_5epochs"

SEED            = 42     # only used if WORK_DIR/data_split.json does not exist yet
TEST_FRACTION   = 0.15   # 15/100 cases -> matches your existing teacher test set (15 held out)
ROI_MARGIN_VOXELS = 20   # only used if the raw dataset has to be rebuilt from scratch
# =================================================================================================

STUDENT_PLANS_NAME = "nnUNetPlansStudent"
STUDENT_WIDTH_SCALE = 0.5   # halves channel width at every stage vs the teacher

DS_FOLDER = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"

# ---- everything persistent lives on Drive now ----
os.environ["nnUNet_raw"]          = f"{WORK_DIR}/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = f"{WORK_DIR}/nnUNet_preprocessed"
os.environ["nnUNet_results"]      = f"{WORK_DIR}/nnUNet_results"

for p in (os.environ["nnUNet_raw"], os.environ["nnUNet_preprocessed"], os.environ["nnUNet_results"]):
    os.makedirs(p, exist_ok=True)

RAW_DS   = f"{os.environ['nnUNet_raw']}/{DS_FOLDER}"
PP_DS    = f"{os.environ['nnUNet_preprocessed']}/{DS_FOLDER}"

def model_dir(trainer, plans_name="nnUNetPlans"):
    return f"{os.environ['nnUNet_results']}/{DS_FOLDER}/{trainer}__{plans_name}__{CONFIG}"

TEACHER_MODEL_DIR = model_dir(TRAINER, "nnUNetPlans")
STUDENT_MODEL_DIR = model_dir(TRAINER, STUDENT_PLANS_NAME)

TEACHER_CHECKPOINT = f"{TEACHER_MODEL_DIR}/fold_{FOLD}/checkpoint_final.pth"

# registry of every model this project compares (see "How will we know whether
# distillation helps?" in the prep deck): teacher(A), scratch student(B), and the
# three KD variants C1/C2/C3. "student" below == model B.
MODELS = {
    "teacher":  dict(trainer=TRAINER, plans="nnUNetPlans"),
    "student":  dict(trainer=TRAINER, plans=STUDENT_PLANS_NAME),
    "c1":       dict(trainer="nnUNetTrainerKD_C1_OutputOnly", plans=STUDENT_PLANS_NAME),
    "c2":       dict(trainer="nnUNetTrainerKD_C2_Feature",    plans=STUDENT_PLANS_NAME),
    "c3":       dict(trainer="nnUNetTrainerKD_C3_Combined",   plans=STUDENT_PLANS_NAME),
}

def get_model_dir(tag):
    cfg = MODELS[tag]
    return model_dir(cfg["trainer"], cfg["plans"])

def verified_copytree(src, dst, retries=4):
    """
    Like shutil.copytree(src, dst, dirs_exist_ok=True), but verifies every file
    actually landed (by size) and retries if not. Plain copytree can silently
    report success on a partial copy when the source is a Drive FUSE mount that
    just got shared with this account -- Drive's indexing can lag, so a file
    shows up in a directory listing before it's actually readable, and the
    FUSE layer can hand back an empty/short read instead of erroring.
    """
    import shutil, time
    last_missing = []
    for attempt in range(1, retries + 1):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        missing = []
        for root, _dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            dst_root = dst if rel == "." else os.path.join(dst, rel)
            for fn in files:
                sp = os.path.join(root, fn)
                dp = os.path.join(dst_root, fn)
                try:
                    ssz = os.path.getsize(sp)
                except OSError:
                    continue
                if not os.path.exists(dp) or os.path.getsize(dp) != ssz:
                    missing.append(sp)
        if not missing:
            print(f"Verified sync: all files match (attempt {attempt}).")
            return
        last_missing = missing
        print(f"Attempt {attempt}/{retries}: {len(missing)} file(s) missing/mismatched "
              f"(e.g. {missing[:3]}), retrying in {10 * attempt}s ...")
        time.sleep(10 * attempt)
    raise RuntimeError(
        f"Sync verification failed after {retries} attempts; still missing/mismatched: "
        f"{last_missing[:10]} (+{max(0, len(last_missing) - 10)} more). This usually means "
        f"Drive hasn't finished indexing a newly-shared folder yet -- wait a few minutes and "
        f"re-run, or copy the data into the new account's own Drive instead of relying on a "
        f"live share."
    )

print("Settings loaded.")
print("  RAW_DS  =", RAW_DS)
print("  PP_DS   =", PP_DS)
print("  TEACHER_MODEL_DIR =", TEACHER_MODEL_DIR)
print("  STUDENT_MODEL_DIR =", STUDENT_MODEL_DIR)
