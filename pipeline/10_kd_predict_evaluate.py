"""Predict a saved KD model on the held-out cases and compute per-case metrics.

The test split is read from WORK_DIR/data_split.json. Cases must remain in the
same split used by the original teacher/student experiments.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "c1": "nnUNetTrainer_KDOutput_5epochs",
    "c2": "nnUNetTrainer_KDFeature_5epochs",
    "c3": "nnUNetTrainer_KDCombined_5epochs",
    "c4": "nnUNetTrainer_KDTopology_5epochs",
}


def _case_id(value, strip_channel: bool = False):
    if isinstance(value, dict):
        for key in ("case_id", "case", "id", "name", "patient_id", "image", "path", "file"):
            if key in value and isinstance(value[key], (str, int)):
                return _case_id(str(value[key]))
        raise ValueError(f"Cannot identify a case id in split record: {value!r}")
    raw = str(value)
    name = Path(raw).name
    had_image_extension = name.endswith(".nii.gz") or name.endswith(".nii")
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    # Strip a channel suffix only for image-file records or the explicit image index.
    # Do not strip IDs such as patient_0001 when they are plain case identifiers/labels.
    should_strip_channel = strip_channel or (had_image_extension and name.endswith("_0000"))
    if should_strip_channel:
        name = re.sub(r"_\d{4}$", "", name)
    return name


def _iter_test_lists(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            looks_like_test = (
                normalized in {"test", "testing", "testids", "testcases", "testset", "testcaseids",
                               "testcase", "testsplit", "heldout", "holdout", "testpatients"}
                or ("test" in normalized and any(token in normalized for token in ("id", "case", "set", "file", "data", "patient")))
            )
            if looks_like_test and isinstance(value, list) and value:
                try:
                    ids = [_case_id(v) for v in value]
                    if ids:
                        yield ids
                except (ValueError, TypeError):
                    pass
            yield from _iter_test_lists(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_test_lists(value)


def _read_test_ids(work_dir: str):
    split_path = Path(work_dir) / "data_split.json"
    if not split_path.is_file():
        return None
    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    candidates = list(_iter_test_lists(split))
    if not candidates:
        raise ValueError(
            f"Could not find a test-case list in {split_path}. Expected a list under a key such as 'test', "
            "'test_ids', or 'test_cases'. Keep the original split file unchanged for a fair comparison."
        )
    # Prefer the most complete explicit list if the split stores more than one test-related field.
    return sorted(candidates, key=len, reverse=True)[0]


def _all_nii_files(folder: Path):
    if not folder.is_dir():
        return []
    return sorted([*folder.glob("*.nii.gz"), *folder.glob("*.nii")])


def _index_folder(folder: Path, is_images: bool):
    indexed = {}
    for path in _all_nii_files(folder):
        cid = _case_id(path.name, strip_channel=is_images)
        if is_images:
            indexed.setdefault(cid, []).append(path)
        else:
            indexed[cid] = path
    return indexed


def _prepare_test_folders(settings, variant: str, file_ending: str):
    raw_ds = Path(settings["RAW_DS"])
    work_dir = Path(settings["WORK_DIR"])
    test_ids = _read_test_ids(str(work_dir))

    images_tr = _index_folder(raw_ds / "imagesTr", is_images=True)
    images_ts = _index_folder(raw_ds / "imagesTs", is_images=True)
    labels_tr = _index_folder(raw_ds / "labelsTr", is_images=False)
    labels_ts = _index_folder(raw_ds / "labelsTs", is_images=False)

    # If there is no explicit split, only use a separately held-out imagesTs set when it has labels.
    if test_ids is None:
        labelled_ts = sorted(set(images_ts) & set(labels_ts))
        if labelled_ts:
            test_ids = labelled_ts
        else:
            raise FileNotFoundError(
                f"No {work_dir / 'data_split.json'} was found, and imagesTs/labelsTs do not form a labelled test set. "
                "The evaluation script will not invent a split because that would invalidate the teacher/student comparison."
            )

    image_source = {}
    label_source = {}
    missing_images, missing_labels = [], []
    for cid in test_ids:
        image_files = images_tr.get(cid) or images_ts.get(cid)
        label_file = labels_tr.get(cid) or labels_ts.get(cid)
        if not image_files:
            missing_images.append(cid)
        else:
            image_source[cid] = image_files
        if label_file is None:
            missing_labels.append(cid)
        else:
            label_source[cid] = label_file

    if missing_images or missing_labels:
        raise FileNotFoundError(
            "Could not locate all held-out files using WORK_DIR/data_split.json. "
            f"Missing image cases (first 10): {missing_images[:10]}; "
            f"missing label cases (first 10): {missing_labels[:10]}. "
            "The test IDs must match case names in nnUNet_raw/Dataset*/imagesTr|imagesTs and labelsTr|labelsTs."
        )

    base = work_dir / "kd_evaluation" / variant
    input_folder = base / "test_images"
    gt_folder = base / "test_labels"
    input_folder.mkdir(parents=True, exist_ok=True)
    gt_folder.mkdir(parents=True, exist_ok=True)

    # Remove old copied files so a changed split cannot leave stale cases in the prediction input.
    for folder in (input_folder, gt_folder):
        for old_file in _all_nii_files(folder):
            old_file.unlink()

    for cid in test_ids:
        for source in image_source[cid]:
            shutil.copy2(source, input_folder / source.name)
        label_path = label_source[cid]
        target_name = cid + file_ending
        shutil.copy2(label_path, gt_folder / target_name)

    return base, input_folder, gt_folder, test_ids


def _binary_metrics(gt: np.ndarray, pred: np.ndarray, spacing_zyx):
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

    gt = np.asarray(gt) > 0
    pred = np.asarray(pred) > 0
    if gt.shape != pred.shape:
        raise ValueError(f"Ground-truth shape {gt.shape} != predicted shape {pred.shape}")

    intersection = int(np.logical_and(gt, pred).sum())
    gt_count = int(gt.sum())
    pred_count = int(pred.sum())
    union = int(np.logical_or(gt, pred).sum())
    dice = 1.0 if gt_count + pred_count == 0 else (2.0 * intersection / (gt_count + pred_count))
    iou = 1.0 if union == 0 else intersection / union
    precision = 1.0 if pred_count == 0 and gt_count == 0 else (0.0 if pred_count == 0 else intersection / pred_count)
    recall = 1.0 if pred_count == 0 and gt_count == 0 else (0.0 if gt_count == 0 else intersection / gt_count)

    hd95 = float("nan")
    if not gt.any() and not pred.any():
        hd95 = 0.0
    elif gt.any() and pred.any():
        structure = generate_binary_structure(gt.ndim, 1)
        gt_surface = gt & ~binary_erosion(gt, structure=structure, border_value=0)
        pred_surface = pred & ~binary_erosion(pred, structure=structure, border_value=0)
        dist_to_gt = distance_transform_edt(~gt_surface, sampling=spacing_zyx)
        dist_to_pred = distance_transform_edt(~pred_surface, sampling=spacing_zyx)
        distances = np.concatenate([dist_to_gt[pred_surface], dist_to_pred[gt_surface]])
        if distances.size:
            hd95 = float(np.percentile(distances, 95))

    cldice = float("nan")
    try:
        from skimage.morphology import skeletonize
        gt_skel = skeletonize(gt)
        pred_skel = skeletonize(pred)
        skel_gt_n = int(gt_skel.sum())
        skel_pred_n = int(pred_skel.sum())
        if skel_gt_n == 0 and skel_pred_n == 0:
            cldice = 1.0 if not gt.any() and not pred.any() else 0.0
        else:
            topo_precision = (
                (1.0 if skel_gt_n == 0 else 0.0) if skel_pred_n == 0
                else float(np.logical_and(pred_skel, gt).sum() / skel_pred_n)
            )
            topo_sensitivity = (
                (1.0 if skel_pred_n == 0 else 0.0) if skel_gt_n == 0
                else float(np.logical_and(gt_skel, pred).sum() / skel_gt_n)
            )
            denom = topo_precision + topo_sensitivity
            cldice = 0.0 if denom == 0 else 2.0 * topo_precision * topo_sensitivity / denom
    except Exception:
        pass

    return {
        "dice": float(dice), "iou": float(iou), "precision": float(precision),
        "recall": float(recall), "hd95_mm": hd95, "cldice": float(cldice),
    }


def _find_model_checkpoint(settings, trainer: str) -> str:
    model_dir = Path(settings["WORK_DIR"]) / "nnUNet_results" / settings["DS_FOLDER"] / (
        f"{trainer}__{settings['STUDENT_PLANS_NAME']}__{settings['CONFIG']}"
    ) / f"fold_{settings['FOLD']}"
    for filename in (
        "checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth",
        "checkpoint_final.safetensors", "checkpoint_best.safetensors", "checkpoint_latest.safetensors",
    ):
        if (model_dir / filename).is_file():
            return filename
    raise FileNotFoundError(
        f"No trained KD checkpoint was found under {model_dir}. Run the corresponding train stage first."
    )


def _evaluate_cases(gt_folder: Path, pred_folder: Path, case_ids, file_ending: str):
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise SystemExit("SimpleITK is required to evaluate predictions; install it with `pip install SimpleITK`.") from e

    rows = []
    for cid in case_ids:
        gt_path = gt_folder / (cid + file_ending)
        pred_path = pred_folder / (cid + file_ending)
        if not gt_path.is_file() or not pred_path.is_file():
            raise FileNotFoundError(f"Missing ground truth or prediction for case {cid}: {gt_path}, {pred_path}")
        gt_image = sitk.ReadImage(str(gt_path))
        pred_image = sitk.ReadImage(str(pred_path))
        gt_array = sitk.GetArrayFromImage(gt_image)
        pred_array = sitk.GetArrayFromImage(pred_image)
        spacing = tuple(reversed(gt_image.GetSpacing()))
        metrics = _binary_metrics(gt_array, pred_array, spacing)
        rows.append({"case_id": cid, **metrics})

    return rows


def _write_metrics(base: Path, rows, variant: str, trainer: str, settings, file_ending: str):
    metric_names = ["dice", "iou", "precision", "recall", "hd95_mm", "cldice"]
    csv_path = base / "per_case_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", *metric_names])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "variant": variant,
        "trainer": trainer,
        "plans": settings["STUDENT_PLANS_NAME"],
        "configuration": settings["CONFIG"],
        "fold": int(settings["FOLD"]),
        "file_ending": file_ending,
        "num_test_cases": len(rows),
        "metrics_mean": {},
        "metrics_std": {},
        "note": "Foreground metrics aggregate all nonzero labels; HD95 is in mm. Per-case metrics are in per_case_metrics.csv.",
    }
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        summary["metrics_mean"][metric] = float(finite.mean()) if finite.size else None
        summary["metrics_std"][metric] = float(finite.std(ddof=1)) if finite.size > 1 else (0.0 if finite.size == 1 else None)
    with (base / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--force-predict", action="store_true", help="Re-run prediction even if output files already exist")
    args = parser.parse_args()

    settings = runpy.run_path(str(HERE / "00_settings.py"))
    subprocess.run([sys.executable, str(HERE / "register_kd_trainers.py")], check=True)

    dataset_json_path = Path(settings["RAW_DS"]) / "dataset.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {dataset_json_path}. Run data conversion first.")
    with dataset_json_path.open("r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    file_ending = dataset_json.get("file_ending", ".nii.gz")

    trainer = VARIANTS[args.variant]
    base, input_folder, gt_folder, case_ids = _prepare_test_folders(settings, args.variant, file_ending)
    pred_folder = base / "predictions"
    pred_folder.mkdir(parents=True, exist_ok=True)

    expected_predictions = [pred_folder / (cid + file_ending) for cid in case_ids]
    if args.force_predict or not all(p.is_file() for p in expected_predictions):
        command_name = shutil.which("nnUNetv2_predict")
        if command_name is None:
            raise SystemExit("Could not find nnUNetv2_predict. Install nnunetv2 in the active Colab environment.")
        checkpoint_name = _find_model_checkpoint(settings, trainer)
        command = [
            command_name,
            "-i", str(input_folder),
            "-o", str(pred_folder),
            "-d", str(settings["DATASET_ID"]),
            "-c", str(settings["CONFIG"]),
            "-p", str(settings["STUDENT_PLANS_NAME"]),
            "-tr", trainer,
            "-f", str(settings["FOLD"]),
            "-chk", checkpoint_name,
        ]
        print("Running:", " ".join(command))
        result = subprocess.run(command, env=os.environ.copy())
        if result.returncode:
            raise SystemExit(result.returncode)
    else:
        print("All held-out predictions already exist; skipping inference.")

    rows = _evaluate_cases(gt_folder, pred_folder, case_ids, file_ending)
    _write_metrics(base, rows, args.variant, trainer, settings, file_ending)
    print(f"Per-case metrics: {base / 'per_case_metrics.csv'}")
    print(f"Summary metrics:  {base / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
