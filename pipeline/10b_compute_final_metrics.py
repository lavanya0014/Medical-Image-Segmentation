"""
Computes the metrics your deck's "How will we know whether distillation
helps?" / "Two Dimensions of Success" slides ask for, for all 5 models
(A=teacher, B=scratch student, C1, C2, C3) -- Dice, Precision, Recall, HD95,
and clDice -- plus parameter count / checkpoint size already saved by
08_gather_stats.py. Assembles everything into one final comparison table.

IMPORTANT: this is a pure POST-HOC evaluation step. It does not retrain or
re-run inference on anything -- it only compares the prediction volumes
07_predict_evaluate.py already saved to disk against the held-out ground
truth labels. Run it any time after all 5 models have been trained and
predicted on the test set; it takes a few minutes on CPU, no GPU needed.

Requires: pip install scikit-image scipy nibabel -q  (scikit-image is only
needed for 3D skeletonization, used by clDice)

Run in Colab:
    python 17_compute_final_metrics.py
Produces, in deployment/report_assets/ (or wherever REPORT_DIR points):
    final_comparison.csv
    final_comparison.json
    final_comparison.png   (grouped bar chart)
and prints a Markdown table to paste directly into your report.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

try:
    from skimage.morphology import skeletonize
except ImportError as error:
    raise ImportError("Run: pip install scikit-image -q") from error

PROJECT_ROOT = Path(os.environ.get("CAS2023_ROOT", "/content/drive/MyDrive/cas2023"))
RAW_DS = PROJECT_ROOT / "nnUNet_raw" / "Dataset501_CAS2023"
LABELS_TS = RAW_DS / "labelsTs"
STATS_DIR = PROJECT_ROOT / "report_assets"
REPORT_DIR = PROJECT_ROOT / "report_assets"

MODELS = ["teacher", "student", "c1", "c2", "c3"]
DISPLAY_NAMES = {
    "teacher": "A: Teacher",
    "student": "B: Scratch student",
    "c1": "C1: Output-level KD",
    "c2": "C2: Feature-level KD",
    "c3": "C3: Combined KD",
}


def find_predictions_dir(tag: str) -> Path:
    """Mirrors 14_build_huggingface_space.py's lookup for C3's warm-start
    folder naming, generalized to all 5 models in case any were similarly
    renamed. Checks the most specific candidate first."""
    candidates = [
        PROJECT_ROOT / f"{tag}_warmstart_predictions",
        PROJECT_ROOT / f"{tag}_predictions",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.nii.gz")):
            return candidate
    raise FileNotFoundError(
        f"No predictions directory found for '{tag}'. Checked: {[str(c) for c in candidates]}\n"
        f"Has 07_predict_evaluate.py been run for this model yet?"
    )


def load_stats(tag: str) -> dict:
    """Pulls params/checkpoint size from 08_gather_stats.py's output, if present."""
    path = STATS_DIR / f"{tag}_stats.json"
    if path.is_file():
        return json.loads(path.read_text())
    print(f"  (no {path.name} found -- params/checkpoint size will be blank for {tag})")
    return {}


def read_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32)) > 0
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, spacing


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(precision), float(recall)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(2 * np.logical_and(pred, gt).sum() / max(pred.sum() + gt.sum(), 1))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """95th-percentile symmetric surface distance (mm), via distance transforms
    -- avoids needing the extra `medpy` dependency."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    pred_border = pred ^ binary_erosion(pred)
    gt_border = gt ^ binary_erosion(gt)
    if not pred_border.any() or not gt_border.any():
        return float("nan")
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    pred_to_gt = dt_gt[pred_border]
    gt_to_pred = dt_pred[gt_border]
    return float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95)))


def cl_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Topology-aware clDice on hard binary masks (your deck's 'Topology-aware
    KD' row references this metric). Uses skeletonize, which works directly
    on 3D boolean arrays in recent scikit-image versions."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)
    t_prec = np.logical_and(skel_pred, gt).sum() / max(skel_pred.sum(), 1)
    t_sens = np.logical_and(skel_gt, pred).sum() / max(skel_gt.sum(), 1)
    if t_prec + t_sens == 0:
        return 0.0
    return float(2 * t_prec * t_sens / (t_prec + t_sens))


def evaluate_model(tag: str) -> dict:
    pred_dir = find_predictions_dir(tag)
    gt_files = sorted(LABELS_TS.glob("*.nii.gz"))
    cases = [f.name for f in gt_files if (pred_dir / f.name).is_file()]
    if not cases:
        raise RuntimeError(f"No matching prediction/label pairs found for '{tag}' in {pred_dir}")

    per_case = {"dice": [], "precision": [], "recall": [], "hd95": [], "cldice": []}
    for case in cases:
        gt, spacing = read_mask(LABELS_TS / case)
        pred, _ = read_mask(pred_dir / case)
        if pred.shape != gt.shape:
            print(f"  WARNING: shape mismatch for {case} ({pred.shape} vs {gt.shape}), skipping")
            continue
        per_case["dice"].append(dice_score(pred, gt))
        p, r = precision_recall(pred, gt)
        per_case["precision"].append(p)
        per_case["recall"].append(r)
        per_case["hd95"].append(hd95(pred, gt, spacing))
        per_case["cldice"].append(cl_dice(pred, gt))

    def mean_ignore_nan(values):
        arr = np.array(values, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        return float(arr.mean()) if len(arr) else float("nan")

    result = {
        "tag": tag,
        "display_name": DISPLAY_NAMES[tag],
        "n_cases": len(cases),
        "dice_mean": mean_ignore_nan(per_case["dice"]),
        "precision_mean": mean_ignore_nan(per_case["precision"]),
        "recall_mean": mean_ignore_nan(per_case["recall"]),
        "hd95_mean_mm": mean_ignore_nan(per_case["hd95"]),
        "cldice_mean": mean_ignore_nan(per_case["cldice"]),
    }
    stats = load_stats(tag)
    result["params_millions"] = stats.get("params_millions")
    result["checkpoint_size_mb"] = stats.get("checkpoint_size_mb")
    return result


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in MODELS:
        print(f"Evaluating {tag} ...")
        try:
            rows.append(evaluate_model(tag))
        except (FileNotFoundError, RuntimeError) as error:
            print(f"  SKIPPING {tag}: {error}")

    if not rows:
        raise RuntimeError("No models could be evaluated -- check that 07_predict_evaluate.py has been run.")

    # CSV + JSON
    csv_path = REPORT_DIR / "final_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (REPORT_DIR / "final_comparison.json").write_text(json.dumps(rows, indent=2))

    # Markdown table for the report
    print("\n--- Markdown table (paste into your report) ---\n")
    header = "| Model | Dice | Precision | Recall | HD95 (mm) | clDice | Params (M) | Checkpoint (MB) |"
    sep = "|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in rows:
        def fmt(v, nd=4):
            return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"
        print(
            f"| {r['display_name']} | {fmt(r['dice_mean'])} | {fmt(r['precision_mean'])} | "
            f"{fmt(r['recall_mean'])} | {fmt(r['hd95_mean_mm'], 2)} | {fmt(r['cldice_mean'])} | "
            f"{fmt(r['params_millions'], 2)} | {fmt(r['checkpoint_size_mb'], 1)} |"
        )

    # Grouped bar plot
    try:
        import matplotlib.pyplot as plt

        names = [r["display_name"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        metrics_01 = ["dice_mean", "precision_mean", "recall_mean", "cldice_mean"]
        labels_01 = ["Dice", "Precision", "Recall", "clDice"]
        x = np.arange(len(names))
        width = 0.2
        for i, (key, label) in enumerate(zip(metrics_01, labels_01)):
            values = [r[key] if r[key] == r[key] else 0 for r in rows]  # nan -> 0 for plotting
            axes[0].bar(x + (i - 1.5) * width, values, width, label=label)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=20, ha="right")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Segmentation quality")
        axes[0].legend()

        hd95_values = [r["hd95_mean_mm"] if r["hd95_mean_mm"] == r["hd95_mean_mm"] else 0 for r in rows]
        axes[1].bar(names, hd95_values, color="#c0504d")
        axes[1].set_title("HD95 (mm, lower is better)")
        axes[1].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        plot_path = REPORT_DIR / "final_comparison.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\nSaved plot: {plot_path}")
    except ImportError:
        print("\n(matplotlib not available -- skipped plot, CSV/JSON/table still saved)")

    print(f"Saved: {csv_path}")
    print(f"Saved: {REPORT_DIR / 'final_comparison.json'}")


if __name__ == "__main__":
    main()
