"""Train the teacher from a verified local cache of Drive preprocessing."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

exec((Path(__file__).resolve().parent / "00_settings.py").read_text())

LOCAL_PREPROCESSED_ROOT = Path(os.environ.get("NNUNET_LOCAL_PREPROCESSED", "/content/local_nnUNet_preprocessed"))


def verified_copytree(source: str | Path, destination: str | Path) -> None:
    """Refresh a local cache without deleting valid files or recopying unchanged ones."""
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


def main() -> None:
    local_dataset = LOCAL_PREPROCESSED_ROOT / DS_FOLDER
    print(f"Refreshing local preprocessing cache: {PP_DS} -> {local_dataset}")
    verified_copytree(PP_DS, local_dataset)

    environment = os.environ.copy()
    environment["nnUNet_preprocessed"] = str(LOCAL_PREPROCESSED_ROOT)
    command = ["nnUNetv2_train", str(DATASET_ID), CONFIG, str(FOLD), "-tr", TRAINER, "--c"]
    try:
        subprocess.run(command, check=True, env=environment)
    finally:
        split_file = local_dataset / "splits_final.json"
        if split_file.exists():
            shutil.copy2(split_file, Path(PP_DS) / split_file.name)


if __name__ == "__main__":
    main()
