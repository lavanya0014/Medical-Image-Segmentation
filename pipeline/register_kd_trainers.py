"""Copy the project KD trainer classes into nnU-Net's trainer discovery tree.

nnUNetv2_train and nnUNetv2_predict discover trainer classes from the installed
nnunetv2/training/nnUNetTrainer package. This script makes our Drive-resident
trainer definitions discoverable without asking you to edit site-packages by hand.
"""
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "kd_trainers"


def main() -> None:
    try:
        import nnunetv2
    except ImportError as e:
        raise SystemExit(
            "nnunetv2 is not installed in this Python environment. Run `pip install nnunetv2 SimpleITK` first."
        ) from e

    package_root = Path(nnunetv2.__file__).resolve().parent
    trainer_root = package_root / "training" / "nnUNetTrainer"
    target = trainer_root / "variants" / "network_architecture"
    target.mkdir(parents=True, exist_ok=True)

    # Ensure nested directories are importable by nnU-Net's recursive class finder.
    for init_file in (trainer_root / "variants" / "__init__.py", target / "__init__.py"):
        if not init_file.exists():
            init_file.write_text('"""Custom trainer variants."""\n', encoding="utf-8")

    files = [
        "_kd_utils.py",
        "nnUNetTrainer_KDOutput_5epochs.py",
        "nnUNetTrainer_KDFeature_5epochs.py",
        "nnUNetTrainer_KDCombined_5epochs.py",
        "nnUNetTrainer_KDTopology_5epochs.py",
    ]
    for filename in files:
        source = SOURCE / filename
        if not source.is_file():
            raise FileNotFoundError(f"Expected trainer source file was not found: {source}")
        shutil.copy2(source, target / filename)

    print(f"Registered C1-C4 nnU-Net trainers in: {target}")
    print("Available trainers:")
    print("  nnUNetTrainer_KDOutput_5epochs       (C1 output-level KD)")
    print("  nnUNetTrainer_KDFeature_5epochs      (C2 feature-level KD)")
    print("  nnUNetTrainer_KDCombined_5epochs     (C3 combined KD)")
    print("  nnUNetTrainer_KDTopology_5epochs     (optional C4 + soft clDice)")


if __name__ == "__main__":
    main()
