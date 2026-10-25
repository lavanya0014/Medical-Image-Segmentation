"""Evaluate every IE643 model from saved test predictions.

Place this file in ``cas2023/pipeline`` and run:
    python 10_evaluate_all_models.py

Add ``--with-latency`` to benchmark end-to-end nnU-Net inference on the test
scans. The latency benchmark is deliberately optional because it reruns model
inference and can take substantial time.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize


PROJECT_ROOT = Path(os.environ.get("CAS2023_ROOT", "/content/drive/MyDrive/cas2023"))
LABELS_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset501_CAS2023" / "labelsTs"
IMAGES_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset501_CAS2023" / "imagesTs"
RESULTS_ROOT = PROJECT_ROOT / "nnUNet_results" / "Dataset501_CAS2023"
REPORTS_DIR = PROJECT_ROOT / "report_assets"


def c3_fold_names() -> tuple[str, str | None]:
    """Find the saved warm-start and fair C3 checkpoints without assumptions."""
    model_dir = RESULTS_ROOT / "nnUNetTrainerKD_C3_Combined__nnUNetPlansStudent__3d_fullres"
    if (model_dir / "fold_0_warmstart").is_dir():
        return "fold_0_warmstart", "fold_0"
    if (model_dir / "fold_0_fair_coldstart").is_dir():
        return "fold_0", "fold_0_fair_coldstart"
    return "fold_0", None


def model_specs() -> list[dict]:
    c3_dir = RESULTS_ROOT / "nnUNetTrainerKD_C3_Combined__nnUNetPlansStudent__3d_fullres"
    warm_fold, fair_fold = c3_fold_names()
    specs = [
        {
            "name": "teacher",
            "label": "Teacher",
            "prediction_dir": PROJECT_ROOT / "teacher_predictions",
            "model_dir": RESULTS_ROOT / "nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres",
            "fold_dir": "fold_0",
        },
        {
            "name": "scratch_student",
            "label": "Scratch student",
            "prediction_dir": PROJECT_ROOT / "student_predictions",
            "model_dir": RESULTS_ROOT / "nnUNetTrainer_5epochs__nnUNetPlansStudent__3d_fullres",
            "fold_dir": "fold_0",
        },
        {
            "name": "c1_warmstart",
            "label": "C1 output KD (warm-start)",
            "prediction_dir": PROJECT_ROOT / "c1_predictions",
            "model_dir": RESULTS_ROOT / "nnUNetTrainerKD_C1_OutputOnly__nnUNetPlansStudent__3d_fullres",
            "fold_dir": "fold_0",
        },
        {
            "name": "c2_warmstart",
            "label": "C2 feature KD (warm-start)",
            "prediction_dir": PROJECT_ROOT / "c2_predictions",
            "model_dir": RESULTS_ROOT / "nnUNetTrainerKD_C2_Feature__nnUNetPlansStudent__3d_fullres",
            "fold_dir": "fold_0",
        },
        {
            "name": "c3_warmstart",
            "label": "C3 combined KD (warm-start)",
            "prediction_dir": PROJECT_ROOT / "c3_warmstart_predictions",
            "model_dir": c3_dir,
            "fold_dir": warm_fold,
        },
    ]
    if fair_fold is not None:
        specs.append(
            {
                "name": "c3_fair_coldstart",
                "label": "C3 combined KD (fair cold-start)",
                "prediction_dir": PROJECT_ROOT / "c3_predictions",
                "model_dir": c3_dir,
                "fold_dir": fair_fold,
            }
        )
    return specs


def safe_ratio(numerator: float, denominator: float, empty_value: float) -> float:
    return empty_value if denominator == 0 else numerator / denominator


def binary_metrics(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> dict:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    true_positive = np.logical_and(prediction, target).sum(dtype=np.float64)
    prediction_count = prediction.sum(dtype=np.float64)
    target_count = target.sum(dtype=np.float64)
    dice = safe_ratio(2 * true_positive, prediction_count + target_count, 1.0)
    iou = safe_ratio(true_positive, np.logical_or(prediction, target).sum(dtype=np.float64), 1.0)
    precision = safe_ratio(true_positive, prediction_count, 1.0 if target_count == 0 else 0.0)
    recall = safe_ratio(true_positive, target_count, 1.0 if prediction_count == 0 else 0.0)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "hd95_mm": hd95(prediction, target, spacing),
        "cldice": cldice(prediction, target),
    }


def hd95(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float | None:
    """Symmetric 95th-percentile surface distance in millimetres."""
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return None
    prediction_surface = np.logical_xor(prediction, ndimage.binary_erosion(prediction))
    target_surface = np.logical_xor(target, ndimage.binary_erosion(target))
    distances_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distances_to_prediction = ndimage.distance_transform_edt(~prediction_surface, sampling=spacing)
    distances = np.concatenate((distances_to_target[prediction_surface], distances_to_prediction[target_surface]))
    return float(np.percentile(distances, 95))


def cldice(prediction: np.ndarray, target: np.ndarray) -> float:
    """Hard-mask clDice using 3D skeletons; larger values are better."""
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    prediction_skeleton = skeletonize(prediction)
    target_skeleton = skeletonize(target)
    topology_precision = safe_ratio(
        np.logical_and(prediction_skeleton, target).sum(dtype=np.float64),
        prediction_skeleton.sum(dtype=np.float64),
        0.0,
    )
    topology_recall = safe_ratio(
        np.logical_and(target_skeleton, prediction).sum(dtype=np.float64),
        target_skeleton.sum(dtype=np.float64),
        0.0,
    )
    return float(safe_ratio(2 * topology_precision * topology_recall, topology_precision + topology_recall, 0.0))


def case_files() -> list[Path]:
    files = sorted(LABELS_DIR.glob("*.nii.gz"))
    if not files:
        raise RuntimeError(f"No ground-truth test labels found in {LABELS_DIR}")
    return files


def evaluate_model(spec: dict, labels: list[Path]) -> tuple[list[dict], dict]:
    prediction_dir = spec["prediction_dir"]
    if not prediction_dir.is_dir():
        raise RuntimeError(f"Missing predictions for {spec['label']}: {prediction_dir}")
    cases = []
    for label_path in labels:
        prediction_path = prediction_dir / label_path.name
        if not prediction_path.is_file():
            raise RuntimeError(f"Missing prediction: {prediction_path}")
        target_image = sitk.ReadImage(str(label_path))
        prediction_image = sitk.ReadImage(str(prediction_path))
        target = sitk.GetArrayFromImage(target_image) > 0
        prediction = sitk.GetArrayFromImage(prediction_image) > 0
        if prediction.shape != target.shape:
            raise RuntimeError(f"Shape mismatch for {label_path.name}: {prediction.shape} vs {target.shape}")
        metrics = binary_metrics(prediction, target, tuple(reversed(target_image.GetSpacing())))
        cases.append({"model": spec["name"], "label": spec["label"], "case": label_path.stem, **metrics})
    summary = {"model": spec["name"], "label": spec["label"], "num_cases": len(cases)}
    for metric in ("dice", "iou", "precision", "recall", "hd95_mm", "cldice"):
        values = np.array([case[metric] for case in cases if case[metric] is not None], dtype=float)
        summary[f"mean_{metric}"] = float(values.mean()) if values.size else None
        summary[f"std_{metric}"] = float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else None
    return cases, summary


def checkpoint_stats(spec: dict) -> dict:
    checkpoint = spec["model_dir"] / spec["fold_dir"] / "checkpoint_final.pth"
    if not checkpoint.is_file():
        return {"checkpoint_mb": None}
    return {"checkpoint_mb": round(checkpoint.stat().st_size / (1024**2), 2)}


def make_predictor_folder(spec: dict) -> tuple[Path, Path | None]:
    """Expose a nonstandard saved fold as fold_0 without changing Drive files."""
    model_dir = spec["model_dir"]
    fold_dir = spec["fold_dir"]
    if fold_dir == "fold_0":
        return model_dir, None
    temporary_dir = Path(tempfile.mkdtemp(prefix=f"ie643_{spec['name']}_"))
    for filename in ("dataset.json", "plans.json"):
        shutil.copy2(model_dir / filename, temporary_dir / filename)
    os.symlink(model_dir / fold_dir, temporary_dir / "fold_0", target_is_directory=True)
    return temporary_dir, temporary_dir


def benchmark_latency(spec: dict, image_files: list[Path], warmup_runs: int) -> dict:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    source_folder, cleanup_folder = make_predictor_folder(spec)
    work_dir = Path(tempfile.mkdtemp(prefix=f"ie643_latency_{spec['name']}_"))
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    try:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=device.type == "cuda",
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(str(source_folder), use_folds=(0,), checkpoint_name="checkpoint_final.pth")
        warmup_image = image_files[0]
        predictor.predict_from_files([[str(warmup_image)]], [str(work_dir / "warmup")], save_probabilities=False, overwrite=True, num_processes_preprocessing=1, num_processes_segmentation_export=1)
        durations = []
        for image_file in image_files:
            output_name = image_file.name.replace("_0000.nii.gz", "")
            start = time.perf_counter()
            predictor.predict_from_files([[str(image_file)]], [str(work_dir / output_name)], save_probabilities=False, overwrite=True, num_processes_preprocessing=1, num_processes_segmentation_export=1)
            durations.append(time.perf_counter() - start)
        return {
            "mean_latency_s": float(np.mean(durations)),
            "std_latency_s": float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0,
            "latency_cases": len(durations),
            "warmup_runs": warmup_runs,
            "device": str(device),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if cleanup_folder is not None:
            shutil.rmtree(cleanup_folder, ignore_errors=True)


def write_reports(per_case: list[dict], summaries: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "all_models_per_case_metrics.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_case[0]))
        writer.writeheader()
        writer.writerows(per_case)
    with open(REPORTS_DIR / "all_models_summary.json", "w") as stream:
        json.dump(summaries, stream, indent=2)
    with open(REPORTS_DIR / "all_models_summary.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-latency", action="store_true", help="Benchmark end-to-end inference after quality metrics.")
    parser.add_argument("--latency-cases", type=int, default=15, help="Number of sorted test scans to benchmark per model.")
    args = parser.parse_args()
    labels = case_files()
    specs = model_specs()
    per_case, summaries = [], []
    for spec in specs:
        print(f"Evaluating quality metrics: {spec['label']}")
        model_cases, summary = evaluate_model(spec, labels)
        summary.update(checkpoint_stats(spec))
        per_case.extend(model_cases)
        summaries.append(summary)
    if args.with_latency:
        image_files = sorted(IMAGES_DIR.glob("*_0000.nii.gz"))[: args.latency_cases]
        if not image_files:
            raise RuntimeError(f"No test images found in {IMAGES_DIR}")
        for summary, spec in zip(summaries, specs):
            print(f"Benchmarking latency: {spec['label']} ({len(image_files)} test scans)")
            summary.update(benchmark_latency(spec, image_files, warmup_runs=1))
    write_reports(per_case, summaries)
    print(f"Saved per-case metrics: {REPORTS_DIR / 'all_models_per_case_metrics.csv'}")
    print(f"Saved summary table: {REPORTS_DIR / 'all_models_summary.csv'}")
    print(f"Saved machine-readable summary: {REPORTS_DIR / 'all_models_summary.json'}")


if __name__ == "__main__":
    main()
