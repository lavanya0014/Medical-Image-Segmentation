"""
Adds more examples to the web demo (index.html) using the 4.92 M C2 model.

  * on-device examples : 64x192x192 float32 patch + uint8 label (assets/<case>_image.bin / _label.bin)
  * gallery examples   : full-volume C2 prediction -> whole-head overlay PNG + whole-volume Dice

Everything is written to  <WORK_DIR>/demo_examples/  :
    index.html            <- your current index.html with the new examples added
    assets/*.bin, *.png   <- upload these into the Space's assets/ folder (not the folder itself, its files)

Run (Colab, same cell style as the other v2 scripts; GPU runtime recommended for the gallery part):

  !{E10} python {P}/17_build_demo_examples_v2.py --index /content/index.html \
        --ondevice CAS2023_041,CAS2023_042 --gallery CAS2023_041,CAS2023_042,CAS2023_072,CAS2023_073

--index    : your current index.html (upload it to /content/ first)
--ondevice : test cases to add to "Run on-device inference"   (must be test-split cases, see imagesTs)
--gallery  : test cases to add to "Example segmentations"
--model    : c2 (default; student / c1 also work)
--flipud / --fliplr : use if a gallery picture comes out upside-down / mirrored compared with Examples 1-3
Must be run with `if __name__ == "__main__"` (nnU-Net uses spawn), which this file does.
"""
import argparse, base64, io, json, os, re, shutil, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import novelty_env_v2  # noqa: F401
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_settings.py")).read())

import numpy as np

PATCH = (64, 192, 192)
OLD_GALLERY_TEXT = "Precomputed full-volume predictions from the earlier C3 (combined-KD, 7.7 M-parameter) student on held-out test scans, rendered as whole-head maximum-intensity-projection overlays for clarity."
NEW_GALLERY_TEXT = ("Precomputed full-volume predictions on held-out test scans, rendered as whole-head maximum-intensity-projection "
                    "overlays for clarity. 'Example 1-3' come from the earlier C3 (combined-KD, 7.7 M-parameter) student; "
                    "entries marked 'C2, 4.92 M' come from the final 4.92 M-parameter C2 student used in this demo.")


# ---------------------------------------------------------------- helpers
def crop_patch(data, seg):
    """data [1,z,y,x] float, seg [1,z,y,x] int -> patch (64,192,192) float32 + label uint8, centred on the vessels."""
    img = data[0].astype(np.float32); lab = (seg[0] > 0).astype(np.uint8)
    idx = np.argwhere(lab > 0)
    centre = idx.mean(0).round().astype(int) if len(idx) else np.array(img.shape) // 2
    out_i = np.full(PATCH, img.min(), np.float32); out_l = np.zeros(PATCH, np.uint8)
    src, dst = [], []
    for ax in range(3):
        n = img.shape[ax]; p = PATCH[ax]
        start = int(np.clip(centre[ax] - p // 2, 0, max(n - p, 0)))
        length = min(p, n)
        src.append(slice(start, start + length))
        d0 = (p - length) // 2
        dst.append(slice(d0, d0 + length))
    out_i[tuple(dst)] = img[tuple(src)]; out_l[tuple(dst)] = lab[tuple(src)]
    return out_i, out_l


def overlay_png(vol, gt, pr, flipud=False, fliplr=False):
    """vol/gt/pr are [z,y,x]. Background = axial slice with most vessels, overlay = whole-head MIP."""
    from PIL import Image
    z = int(np.argmax(gt.reshape(gt.shape[0], -1).sum(1))) if gt.any() else gt.shape[0] // 2
    bg = vol[z].astype(np.float32)
    lo, hi = np.percentile(bg, [1, 99.5]); bg = np.clip((bg - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([bg] * 3, -1) * 255
    g = gt.max(0) > 0; p = pr.max(0) > 0
    rgb[g & p] = (255, 215, 0); rgb[p & ~g] = (235, 55, 55); rgb[g & ~p] = (45, 210, 85)
    if flipud: rgb = rgb[::-1]
    if fliplr: rgb = rgb[:, ::-1]
    im = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def dice(a, b):
    a = a > 0; b = b > 0
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def patch_index_html(index_in, index_out, new_ondevice, new_gallery):
    s = open(index_in, encoding="utf-8").read()
    lines = s.split("\n")
    for k, ln in enumerate(lines):
        if ln.startswith("const MANIFEST = "):
            m = json.loads(ln[len("const MANIFEST = "):].rstrip().rstrip(";"))
            m["cases"].update(new_ondevice)
            lines[k] = "const MANIFEST = " + json.dumps(m) + ";"
        elif ln.startswith("const GALLERY_CASES = {"):
            g = json.loads(ln[len("const GALLERY_CASES = "):].rstrip().rstrip(";"))
            g.update(new_gallery)
            lines[k] = "const GALLERY_CASES = " + json.dumps(g) + ";"
    s = "\n".join(lines)
    assert "const MANIFEST = " in s and "const GALLERY_CASES = {" in s
    if OLD_GALLERY_TEXT in s:
        s = s.replace(OLD_GALLERY_TEXT, NEW_GALLERY_TEXT)
    open(index_out, "w", encoding="utf-8").write(s)


def main():
    import torch, SimpleITK as sitk
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--ondevice", default="")
    ap.add_argument("--gallery", default="")
    ap.add_argument("--model", default="c2")
    ap.add_argument("--flipud", action="store_true"); ap.add_argument("--fliplr", action="store_true")
    a = ap.parse_args()
    od = [c for c in a.ondevice.split(",") if c]; gl = [c for c in a.gallery.split(",") if c]
    cases = sorted(set(od + gl)); assert cases, "give --ondevice and/or --gallery"
    raw_img = lambda c: Path(RAW_DS, "imagesTs", f"{c}_0000.nii.gz")
    raw_lab = lambda c: Path(RAW_DS, "labelsTs", f"{c}.nii.gz")
    for c in cases:
        assert raw_img(c).exists() and raw_lab(c).exists(), f"{c} is not a test-split case (check imagesTs/labelsTs)"

    out = Path(WORK_DIR) / "demo_examples"; assets = out / "assets"
    shutil.rmtree(out, ignore_errors=True); assets.mkdir(parents=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                           perform_everything_on_device=(dev.type == "cuda"), device=dev,
                           verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    pred.initialize_from_trained_model_folder(get_model_dir(a.model), use_folds=(FOLD,), checkpoint_name="checkpoint_final.pth")

    new_ondevice = {}
    if od:
        pp = pred.configuration_manager.preprocessor_class(verbose=False)
        for c in od:
            data, seg, _ = pp.run_case([str(raw_img(c))], str(raw_lab(c)), pred.plans_manager,
                                       pred.configuration_manager, pred.dataset_json)
            im, lb = crop_patch(np.asarray(data), np.asarray(seg))
            im.tofile(assets / f"{c}_image.bin"); lb.tofile(assets / f"{c}_label.bin")
            new_ondevice[c] = {"image_file": f"assets/{c}_image.bin", "label_file": f"assets/{c}_label.bin",
                               "preview_file": f"assets/{c}_preview.png", "vessel_voxels": int(lb.sum())}
            print(f"on-device {c}: patch {im.shape}, vessel voxels {int(lb.sum())}, image range {im.min():.2f}..{im.max():.2f}")

    new_gallery = {}
    if gl:
        pdir = out / "_pred"; pdir.mkdir()
        pred.predict_from_files([[str(raw_img(c))] for c in gl], [str(pdir / c) for c in gl],
                                save_probabilities=False, overwrite=True,
                                num_processes_preprocessing=1, num_processes_segmentation_export=1)
        for c in gl:
            vol = sitk.GetArrayFromImage(sitk.ReadImage(str(raw_img(c))))
            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(raw_lab(c))))
            pr = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / f"{c}.nii.gz")))
            d = dice(gt, pr)
            png = overlay_png(vol, gt, pr, a.flipud, a.fliplr)
            (assets / f"{c}_gallery.png").write_bytes(png)
            new_gallery[f"{c} (C2, 4.92 M)"] = {"image": "data:image/png;base64," + base64.b64encode(png).decode(), "dice": d}
            print(f"gallery {c}: whole-volume Dice {d:.4f}")
        shutil.rmtree(pdir, ignore_errors=True)

    patch_index_html(a.index, out / "index.html", new_ondevice, new_gallery)
    print("\nDone. Upload to the Space:  index.html (replace)  and the files inside  assets/  (*.bin) -> Space's assets/ folder.")
    print("Output folder:", out)


if __name__ == "__main__":
    main()
