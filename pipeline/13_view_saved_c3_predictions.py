"""Presentation viewer for valid, full-volume nnU-Net C3 predictions.

This does not rerun inference. It overlays the saved C3 warm-start predictions
made by nnU-Net's complete preprocessing and sliding-window pipeline, so the
visualisation matches the reported test results.

Run in Colab:
    !pip -q install gradio nibabel
    !python /content/drive/MyDrive/cas2023/pipeline/13_view_saved_c3_predictions.py --share
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import nibabel as nib
import numpy as np


PROJECT_ROOT = Path("/content/drive/MyDrive/cas2023")
IMAGES_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset501_CAS2023" / "imagesTs"
LABELS_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset501_CAS2023" / "labelsTs"


def c3_prediction_dir() -> Path:
    """Use warm-start predictions first; retain compatibility with older runs."""
    warm_start = PROJECT_ROOT / "c3_warmstart_predictions"
    fallback = PROJECT_ROOT / "c3_predictions"
    if warm_start.is_dir():
        return warm_start
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(
        "No C3 prediction folder found. Expected c3_warmstart_predictions or c3_predictions under the project root."
    )


def read_volume(path: Path) -> np.ndarray:
    array = nib.load(path).get_fdata(dtype=np.float32)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D NIfTI volume: {path}")
    return np.nan_to_num(array)


def image_path(case_name: str) -> Path:
    return IMAGES_DIR / case_name.replace(".nii.gz", "_0000.nii.gz")


def make_overlay(image: np.ndarray, prediction: np.ndarray, label: np.ndarray, slice_index: int) -> np.ndarray:
    """Return an axial overlay: red prediction, green ground truth, yellow overlap."""
    slice_index = int(np.clip(slice_index, 0, image.shape[2] - 1))
    source = image[:, :, slice_index].T
    low, high = np.percentile(source, (1, 99))
    gray = np.clip((source - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.repeat((gray[..., None] * 255).astype(np.uint8), 3, axis=-1)
    prediction_mask = prediction[:, :, slice_index].T > 0
    label_mask = label[:, :, slice_index].T > 0
    rgb[label_mask & ~prediction_mask] = [45, 210, 85]       # ground truth only
    rgb[prediction_mask & ~label_mask] = [235, 55, 55]       # prediction only
    rgb[prediction_mask & label_mask] = [255, 215, 0]        # correct overlap
    return rgb


def available_cases(prediction_dir: Path) -> list[str]:
    cases = []
    for label_path in sorted(LABELS_DIR.glob("*.nii.gz")):
        if (prediction_dir / label_path.name).is_file() and image_path(label_path.name).is_file():
            cases.append(label_path.name)
    if not cases:
        raise FileNotFoundError("No case has all three files: image, ground truth, and saved C3 prediction.")
    return cases


def create_app(prediction_dir: Path) -> gr.Blocks:
    cases = available_cases(prediction_dir)

    def load_case(case_name: str):
        image = read_volume(image_path(case_name))
        prediction = read_volume(prediction_dir / case_name)
        label = read_volume(LABELS_DIR / case_name)
        if image.shape != prediction.shape or image.shape != label.shape:
            raise gr.Error(f"Shape mismatch in {case_name}: image {image.shape}, prediction {prediction.shape}, label {label.shape}")
        foreground_by_slice = ((prediction > 0) | (label > 0)).sum(axis=(0, 1))
        best_slice = int(np.argmax(foreground_by_slice))
        state = {"image": image, "prediction": prediction, "label": label}
        prediction_mask = prediction > 0
        label_mask = label > 0
        dice = 2 * np.logical_and(prediction_mask, label_mask).sum() / max(prediction_mask.sum() + label_mask.sum(), 1)
        summary = (
            f"**{case_name}** — full nnU-Net C3 warm-start prediction. "
            f"Volume Dice: **{dice:.4f}**. Red = prediction only; green = ground truth only; yellow = overlap."
        )
        return make_overlay(image, prediction, label, best_slice), summary, state, gr.update(maximum=image.shape[2] - 1, value=best_slice)

    def select_slice(slice_index: int, state: dict | None):
        if state is None:
            return None
        return make_overlay(state["image"], state["prediction"], state["label"], slice_index)

    with gr.Blocks(title="CAS2023 C3 Full-Volume Prediction Viewer") as app:
        gr.Markdown(
            "# CAS2023 C3 KD: Valid Test-Prediction Viewer\n"
            "Displays **saved full-volume nnU-Net predictions**. It does not use the experimental one-patch ONNX demo."
        )
        case_selector = gr.Dropdown(choices=cases, value=cases[0], label="Test case")
        overlay = gr.Image(label="Red: C3 prediction | Green: ground truth | Yellow: correct overlap", type="numpy")
        slice_selector = gr.Slider(label="Axial slice", minimum=0, maximum=1, step=1, value=0)
        summary = gr.Markdown()
        state = gr.State()
        case_selector.change(load_case, inputs=case_selector, outputs=[overlay, summary, state, slice_selector])
        slice_selector.change(select_slice, inputs=[slice_selector, state], outputs=overlay)
        app.load(load_case, inputs=case_selector, outputs=[overlay, summary, state, slice_selector])
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio link for Colab.")
    args = parser.parse_args()
    create_app(c3_prediction_dir()).launch(share=args.share)


if __name__ == "__main__":
    main()
