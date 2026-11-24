"""
Run ONCE (safe to re-run). Lets you choose the number of training epochs from the command line without
touching the finished 5-epoch results:
  * 00_settings.py : the student trainer name can be overridden (TRAINER_NAME); the TEACHER keeps
                     nnUNetTrainer_5epochs so its existing checkpoint is still found.
  * 09_train_kd_variant.py : passes KD_NUM_EPOCHS from the environment (it used to ignore it and always use 5).

    !python /content/drive/MyDrive/cas2023/pipeline/apply_patch_epochs.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))

def patch(fname, pairs):
    path = os.path.join(HERE, fname)
    text = open(path).read()
    for old, new in pairs:
        if new in text:
            print(f"  {fname}: already patched -> {old[:55]!r}")
            continue
        assert text.count(old) == 1, f"{fname}: expected exactly one {old!r}, found {text.count(old)}. Send me that line."
        text = text.replace(old, new)
        print(f"  {fname}: patched -> {old[:55]!r}")
    open(path, "w").write(text)

patch("00_settings.py", [
    ('TRAINER    = "nnUNetTrainer_5epochs"',
     'TEACHER_TRAINER = "nnUNetTrainer_5epochs"   # the teacher was trained for 5 epochs; keep finding it\n'
     'TRAINER    = os.environ.get("TRAINER_NAME", "nnUNetTrainer_5epochs")   # student trainer; e.g. nnUNetTrainer_10epochs'),
    ('TEACHER_MODEL_DIR = model_dir(TRAINER, "nnUNetPlans")',
     'TEACHER_MODEL_DIR = model_dir(TEACHER_TRAINER, "nnUNetPlans")'),
    ('"teacher":  dict(trainer=TRAINER, plans="nnUNetPlans"),',
     '"teacher":  dict(trainer=TEACHER_TRAINER, plans="nnUNetPlans"),'),
])
patch("09_train_kd_variant.py", [
    ('environment["KD_NUM_EPOCHS"] = str(globals().get("KD_NUM_EPOCHS", 5))',
     'environment["KD_NUM_EPOCHS"] = str(os.environ.get("KD_NUM_EPOCHS", globals().get("KD_NUM_EPOCHS", 5)))'),
])
print("Patch done.")
