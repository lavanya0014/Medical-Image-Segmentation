"""Prepare browser-demo patches from held-out raw nnU-Net test cases.

The image tensor is preprocessed by the same nnU-Net v2 preprocessor used for
inference, using the plans and configuration recorded in the supplied trained
model folder. It deliberately does *not* read ``nnUNet_preprocessed``: that
cache normally contains training cases only.

Example (adjust paths for the local installation)::

    python 15_prepare_web_demo_assets.py \
      --project-root /content/drive/MyDrive/cas2023 \
      --model-folder /content/drive/MyDrive/cas2023/nnUNet_results/.../fold_model

The model folder must contain ``plans.json``, ``dataset.json``, and the chosen
checkpoint under ``fold_<n>/``. Run ``--help`` for paths that differ from the
project's conventional layout.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
from PIL import Image


DEFAULT_DATASET_NAME = "Dataset501_CAS2023"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=os.environ.get("CAS2023_ROOT"),
        help="Project root. Defaults to $CAS2023_ROOT; required unless every path below is supplied.",
    )
    parser.add_argument("--raw-dataset-dir", type=Path, help="Raw nnU-Net dataset directory containing imagesTs/labelsTs.")
    parser.add_argument("--images-dir", type=Path, help="Raw held-out image directory (normally <raw-dataset-dir>/imagesTs).")
    parser.add_argument("--labels-dir", type=Path, help="Raw held-out label directory (normally <raw-dataset-dir>/labelsTs).")
    parser.add_argument("--split-file", type=Path, help="JSON file whose 'test' list names the held-out cases.")
    parser.add_argument("--output-dir", type=Path, help="Directory for browser assets.")
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=os.environ.get("NNUNET_MODEL_FOLDER"),
        help="Required trained-model folder holding plans.json, dataset.json, and fold_<n>/ checkpoints. Defaults to $NNUNET_MODEL_FOLDER.",
    )
    parser.add_argument("--fold", default="0", help="Model fold to load (default: 0; use 'all' only when that folder exists).")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth", help="Checkpoint filename within fold_<n> (default: checkpoint_final.pth).")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of split-file test cases to export (default: 3).")
    parser.add_argument("--case", dest="cases", action="append", help="Explicit held-out case ID. Repeat to bypass split-file selection; every ID must resolve uniquely.")
    args = parser.parse_args()

    if args.num_examples < 1:
        parser.error("--num-examples must be at least 1")
    if args.model_folder is None:
        parser.error("--model-folder (or $NNUNET_MODEL_FOLDER) is required")

    if args.raw_dataset_dir is None and args.project_root is not None:
        args.raw_dataset_dir = args.project_root / "nnUNet_raw" / DEFAULT_DATASET_NAME
    if args.images_dir is None and args.raw_dataset_dir is not None:
        args.images_dir = args.raw_dataset_dir / "imagesTs"
    if args.labels_dir is None and args.raw_dataset_dir is not None:
        args.labels_dir = args.raw_dataset_dir / "labelsTs"
    if args.split_file is None and args.project_root is not None:
        args.split_file = args.project_root / "data_split.json"
    if args.output_dir is None and args.project_root is not None:
        args.output_dir = args.project_root / "deployment" / "web_demo_assets"

    missing = [name for name in ("images_dir", "labels_dir", "output_dir") if getattr(args, name) is None]
    if missing:
        parser.error("Set --project-root or provide " + ", ".join("--" + item.replace("_", "-") for item in missing))
    if not args.cases and args.split_file is None:
        parser.error("Set --split-file/--project-root or supply one or more --case values")
    return args


def fuse_recovery_message(path: Path) -> str:
    return (
        f"Unable to read {path}. This looks like a disconnected Google Drive FUSE mount.\n"
        "In Colab, remount Drive before retrying:\n"
        "  from google.colab import drive\n"
        "  drive.flush_and_unmount()\n"
        "  drive.mount('/content/drive', force_remount=True)\n"
        "Then rerun this script. Do not treat a partially written asset directory as valid."
    )


def is_fuse_failure(error: BaseException) -> bool:
    message = str(error).lower()
    return (
        isinstance(error, OSError) and error.errno in {errno.ENOTCONN, errno.ESTALE, errno.EIO}
    ) or any(token in message for token in ("transport endpoint is not connected", "blosc2", "returned null"))


def require_readable_directory(path: Path, description: str) -> None:
    try:
        if not path.is_dir():
            raise FileNotFoundError(f"{description} is not a directory: {path}")
        # Force a small directory operation now, so FUSE failures get useful guidance.
        next(path.iterdir(), None)
    except OSError as error:
        if is_fuse_failure(error):
            raise RuntimeError(fuse_recovery_message(path)) from error
        raise


def file_ending(dataset_json: dict) -> str:
    ending = dataset_json.get("file_ending")
    if not isinstance(ending, str) or not ending:
        raise RuntimeError("Model dataset.json has no valid 'file_ending'; cannot identify raw image files.")
    return ending


def expected_modalities(dataset_json: dict) -> list[int]:
    channels = dataset_json.get("channel_names")
    if not isinstance(channels, dict) or not channels:
        raise RuntimeError("Model dataset.json has no valid 'channel_names'; cannot determine model modalities.")
    try:
        indices = sorted(int(index) for index in channels)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Model dataset.json channel_names keys must be numeric modality indices.") from error
    if indices != list(range(len(indices))):
        raise RuntimeError(f"Expected consecutive modality indices 0..{len(indices) - 1}, found {indices}.")
    return indices


def image_case_map(images_dir: Path, ending: str, modalities: Iterable[int]) -> dict[str, list[Path]]:
    """Return only complete, modality-correct raw test cases."""
    available: dict[str, list[Path]] = {}
    first_suffix = f"_0000{ending}"
    for first_modality in sorted(images_dir.glob(f"*{first_suffix}")):
        case_id = first_modality.name[: -len(first_suffix)]
        paths = [images_dir / f"{case_id}_{index:04d}{ending}" for index in modalities]
        if all(path.is_file() for path in paths):
            available[case_id] = paths
    if not available:
        raise RuntimeError(
            f"No complete raw test cases found in {images_dir}. Expected names like <case>_0000{ending}."
        )
    return available


def case_aliases(case_id: str) -> set[str]:
    """Conservative aliases for common CAS2023 ID spelling differences."""
    aliases = {str(case_id)}
    match = re.fullmatch(r"(?:CAS2023_)?(\d+)", str(case_id))
    if match:
        number = str(int(match.group(1)))
        aliases.update({number, number.zfill(3), f"CAS2023_{number.zfill(3)}"})
    return aliases


def resolve_requested_cases(requested: Iterable[object], available: dict[str, list[Path]], count: int) -> list[str]:
    """Resolve only explicit/split-listed cases; never substitute arbitrary files."""
    resolved: list[str] = []
    aliases_to_case: dict[str, set[str]] = {}
    for available_case in available:
        for alias in case_aliases(available_case):
            aliases_to_case.setdefault(alias, set()).add(available_case)

    for requested_case in requested:
        raw = str(requested_case)
        matches = aliases_to_case.get(raw, set())
        if not matches:
            matches = set().union(*(aliases_to_case.get(alias, set()) for alias in case_aliases(raw)))
        if len(matches) != 1:
            found = ", ".join(sorted(matches)) or "none"
            raise RuntimeError(
                f"Held-out case {raw!r} does not resolve uniquely in imagesTs (matches: {found}). "
                "Correct data_split.json, use --case with the exact raw case ID, or rename the raw files consistently."
            )
        case_id = next(iter(matches))
        if case_id not in resolved:
            resolved.append(case_id)
        if len(resolved) == count:
            return resolved
    raise RuntimeError(f"Only resolved {len(resolved)} requested held-out case(s), but --num-examples={count}.")


def load_requested_cases(args: argparse.Namespace) -> list[object]:
    if args.cases:
        return args.cases
    try:
        with args.split_file.open() as handle:
            split = json.load(handle)
    except OSError as error:
        if is_fuse_failure(error):
            raise RuntimeError(fuse_recovery_message(args.split_file)) from error
        raise
    if not isinstance(split, dict) or not isinstance(split.get("test"), list) or not split["test"]:
        raise RuntimeError(f"{args.split_file} must contain a non-empty JSON list at key 'test'.")
    return split["test"]


class PreprocessingContext(NamedTuple):
    plans_manager: object
    configuration_manager: object
    dataset_json: dict


def load_preprocessing_context(model_folder: Path, fold: str, checkpoint: str) -> PreprocessingContext:
    """Load exactly the preprocessing configuration recorded in a checkpoint.

    This intentionally does not build the inference network. Building a network
    requires importing a custom trainer (such as a KD trainer), but raw-data
    preprocessing only needs plans, dataset metadata, and the configuration
    name stored in that checkpoint.
    """
    try:
        import torch
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    except ImportError as error:
        raise RuntimeError(
            "Could not import PyTorch or nnUNetv2's PlansManager in the Python environment running this script."
        ) from error

    require_readable_directory(model_folder, "Model folder")
    checkpoint_path = model_folder / f"fold_{fold}" / checkpoint
    for required in (model_folder / "plans.json", model_folder / "dataset.json", checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required model file is missing: {required}")
    try:
        with (model_folder / "plans.json").open() as handle:
            plans_manager = PlansManager(json.load(handle))
        with (model_folder / "dataset.json").open() as handle:
            dataset_json = json.load(handle)
        # Do not use nnunetv2.utilities.checkpoints here: older nnU-Net v2
        # releases do not expose that module. All supported .pth checkpoints
        # are ordinary PyTorch dictionaries and only init_args is needed.
        try:
            checkpoint_data = torch.load(str(checkpoint_path), map_location=torch.device("cpu"), weights_only=False)
        except TypeError:  # PyTorch releases before the weights_only argument
            checkpoint_data = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
    except Exception as error:
        if is_fuse_failure(error):
            raise RuntimeError(fuse_recovery_message(model_folder)) from error
        raise
    try:
        configuration_name = checkpoint_data["init_args"]["configuration"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not record init_args.configuration; cannot select its nnU-Net plan."
        ) from error
    return PreprocessingContext(plans_manager, plans_manager.get_configuration(configuration_name), dataset_json)


def preprocess_case(context: PreprocessingContext, image_files: list[Path], label_file: Path):
    """Use nnU-Net's own reader, transpose, crop, normalization, and resampling."""
    preprocessor = context.configuration_manager.preprocessor_class(verbose=False)
    try:
        data, segmentation, _ = preprocessor.run_case(
            [str(path) for path in image_files],
            str(label_file),
            context.plans_manager,
            context.configuration_manager,
            context.dataset_json,
        )
    except Exception as error:
        if is_fuse_failure(error):
            raise RuntimeError(fuse_recovery_message(image_files[0])) from error
        raise
    if segmentation is None:
        raise RuntimeError(f"nnU-Net did not load segmentation {label_file}")
    if data.shape[1:] != segmentation.shape[1:]:
        raise RuntimeError(f"Preprocessed image/label shapes differ: {data.shape} vs {segmentation.shape}")
    return data.astype(np.float32, copy=False), (segmentation[0] > 0).astype(np.uint8, copy=False)


def centered_crop(image: np.ndarray, label: np.ndarray, patch_size: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != len(patch_size) + 1 or label.shape != image.shape[1:]:
        raise ValueError(f"Expected image (C, *spatial) and label (*spatial), got {image.shape} and {label.shape}")
    coordinates = np.argwhere(label > 0)
    center = coordinates.mean(axis=0) if len(coordinates) else np.asarray(label.shape, dtype=float) / 2
    starts = [max(0, min(int(round(position - size / 2)), dimension - size)) if dimension >= size else 0
              for position, dimension, size in zip(center, label.shape, patch_size)]

    image_patch = np.zeros((image.shape[0], *patch_size), dtype=np.float32)
    label_patch = np.zeros(patch_size, dtype=np.uint8)
    source = tuple(slice(start, min(start + size, dimension)) for start, size, dimension in zip(starts, patch_size, label.shape))
    destination = tuple(slice(0, item.stop - item.start) for item in source)
    image_patch[(slice(None), *destination)] = image[(slice(None), *source)]
    label_patch[destination] = label[source]
    return image_patch, label_patch


def write_bytes_atomically(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(payload)
    try:
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def save_preview(image_patch: np.ndarray, path: Path) -> None:
    middle = image_patch.shape[1] // 2
    image_slice = image_patch[0, middle]
    low, high = np.percentile(image_slice, (1, 99))
    grayscale = np.clip((image_slice - low) / max(high - low, 1e-6), 0, 1)
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    Image.fromarray((grayscale * 255).astype(np.uint8)).save(temporary_path, format="PNG")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    for path, description in ((args.images_dir, "imagesTs"), (args.labels_dir, "labelsTs")):
        require_readable_directory(path, description)

    context = load_preprocessing_context(args.model_folder, args.fold, args.checkpoint)
    ending = file_ending(context.dataset_json)
    modalities = expected_modalities(context.dataset_json)
    available = image_case_map(args.images_dir, ending, modalities)
    test_cases = resolve_requested_cases(load_requested_cases(args), available, args.num_examples)
    patch_size = tuple(int(value) for value in context.configuration_manager.patch_size)
    print(f"Using model configuration patch size: {patch_size}")
    print(f"Using held-out raw test cases: {test_cases}")

    manifest = {
        "patch_size": list(patch_size),
        "channels": len(modalities),
        "image_shape": [1, len(modalities), *patch_size],
        "label_shape": list(patch_size),
        "cases": {},
    }
    for case_id in test_cases:
        label_file = args.labels_dir / f"{case_id}{ending}"
        if not label_file.is_file():
            raise FileNotFoundError(f"Held-out test label is missing: {label_file}")
        image, label = preprocess_case(context, available[case_id], label_file)
        image_patch, label_patch = centered_crop(image, label, patch_size)

        write_bytes_atomically(args.output_dir / f"{case_id}_image.bin", image_patch.astype("<f4", copy=False).tobytes())
        write_bytes_atomically(args.output_dir / f"{case_id}_label.bin", label_patch.astype("<u1", copy=False).tobytes())
        save_preview(image_patch, args.output_dir / f"{case_id}_preview.png")
        manifest["cases"][case_id] = {
            "image_file": f"{case_id}_image.bin",
            "label_file": f"{case_id}_label.bin",
            "preview_file": f"{case_id}_preview.png",
            "vessel_voxels": int(label_patch.sum()),
        }
        print(f"{case_id}: vessel voxels in patch = {int(label_patch.sum())}")

    write_bytes_atomically(args.output_dir / "manifest.json", json.dumps(manifest, indent=2).encode() + b"\n")
    print(f"Saved {len(test_cases)} case(s) to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as error:
        if is_fuse_failure(error):
            print(fuse_recovery_message(Path("the Drive-mounted project directory")), file=sys.stderr)
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
