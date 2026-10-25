"""Install the three IE643 KD trainers into the current Colab runtime.

Run this after installing nnU-Net, and before invoking any C1/C2/C3 stage.
The source files must be in ``pipeline/trainers`` beside this script.
"""
from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "trainers"
DESTINATION_DIR = (
    Path(nnunetv2.__file__).resolve().parent
    / "training"
    / "nnUNetTrainer"
    / "variants"
    / "custom_kd"
)
TRAINER_CLASSES = (
    "nnUNetTrainerKD_C1_OutputOnly",
    "nnUNetTrainerKD_C2_Feature",
    "nnUNetTrainerKD_C3_Combined",
)


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise RuntimeError(f"Missing trainer source directory: {SOURCE_DIR}")

    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    (DESTINATION_DIR / "__init__.py").touch()
    for source in SOURCE_DIR.glob("*.py"):
        shutil.copy2(source, DESTINATION_DIR / source.name)

    importlib.invalidate_caches()
    trainer_root = Path(nnunetv2.__file__).resolve().parent / "training" / "nnUNetTrainer"
    for class_name in TRAINER_CLASSES:
        found = recursive_find_python_class(
            str(trainer_root), class_name, "nnunetv2.training.nnUNetTrainer"
        )
        if found is None:
            raise RuntimeError(f"nnU-Net cannot find installed trainer {class_name}.")
        print(f"Verified: {class_name}")

    print(f"Installed KD trainers in {DESTINATION_DIR}")


if __name__ == "__main__":
    main()
