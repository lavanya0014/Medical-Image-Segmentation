"""Run C1, C2, or C3 KD using a frozen full-width teacher.

This script always trains against a verified local preprocessing cache. It uses
the already-narrowed student plans without applying a second width multiplier.
Set KD_WARM_START=0 to disable baseline-student warm-starting.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

exec((Path(__file__).resolve().parent / "00_settings.py").read_text())

TRAINERS = {
    "c1": "nnUNetTrainerKD_C1_OutputOnly",
    "c2": "nnUNetTrainerKD_C2_Feature",
    "c3": "nnUNetTrainerKD_C3_Combined",
}
LOCAL_PREPROCESSED_ROOT = Path(os.environ.get("NNUNET_LOCAL_PREPROCESSED", "/content/local_nnUNet_preprocessed"))


def verified_copytree(source: str | Path, destination: str | Path) -> None:
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise RuntimeError(f"Preprocessed dataset is missing on Drive: {source}")
    for current, directories, filenames in os.walk(source):
        relative = Path(current).relative_to(source)
        target_dir = destination / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            (target_dir / directory).mkdir(exist_ok=True)
        for filename in filenames:
            original, cached = Path(current) / filename, target_dir / filename
            if not cached.exists() or original.stat().st_size != cached.stat().st_size:
                shutil.copy2(original, cached)
    if not (destination / "dataset.json").is_file():
        raise RuntimeError(f"Local preprocessing cache is incomplete: {destination}")


def baseline_checkpoint() -> Path:
    configured = globals().get("STUDENT_CHECKPOINT")
    if configured:
        return Path(configured)
    return Path(STUDENT_MODEL_DIR) / f"fold_{FOLD}" / "checkpoint_final.pth"


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in TRAINERS:
        raise SystemExit("Usage: python 09_train_kd_variant.py [c1|c2|c3]")
    variant = sys.argv[1]
    teacher_checkpoint = Path(TEACHER_CHECKPOINT)
    teacher_plans = Path(TEACHER_MODEL_DIR) / "plans.json"
    if not teacher_checkpoint.is_file() or not teacher_plans.is_file():
        raise RuntimeError("Teacher checkpoint or teacher plans.json is missing; finish teacher training first.")

    local_dataset = LOCAL_PREPROCESSED_ROOT / DS_FOLDER
    print(f"Refreshing local preprocessing cache: {PP_DS} -> {local_dataset}")
    verified_copytree(PP_DS, local_dataset)

    environment = os.environ.copy()
    environment["nnUNet_preprocessed"] = str(LOCAL_PREPROCESSED_ROOT)
    environment["KD_TEACHER_CHECKPOINT"] = str(teacher_checkpoint)
    environment["KD_TEACHER_PLANS"] = str(teacher_plans)
    environment["KD_NUM_EPOCHS"] = str(globals().get("KD_NUM_EPOCHS", 5))
    environment["KD_WARM_START"] = environment.get("KD_WARM_START", "1")
    candidate = baseline_checkpoint()
    if candidate.is_file():
        environment["KD_STUDENT_CHECKPOINT"] = str(candidate)
        print(f"Baseline warm-start candidate: {candidate}")
    else:
        print("No baseline student checkpoint found; KD starts from random student weights.")

    plans = MODELS[variant]["plans"]
    command = [
        "nnUNetv2_train", str(DATASET_ID), CONFIG, str(FOLD),
        "-tr", TRAINERS[variant], "-p", plans, "--c",
    ]
    try:
        subprocess.run(command, check=True, env=environment)
    finally:
        split_file = local_dataset / "splits_final.json"
        if split_file.exists():
            shutil.copy2(split_file, Path(PP_DS) / split_file.name)


if __name__ == "__main__":
    main()
