"""
Convert raw CAS2023 scans -> nnU-Net raw format, cropped to the vessel ROI.
Idempotent: skips entirely if RAW_DS/dataset.json already exists (which, now that
RAW_DS lives on Drive, should only ever need to run ONCE, ever).

If WORK_DIR/data_split.json already exists (from your earlier teacher run), it is
REUSED so the student gets the exact same train/test split as the teacher -- this
matters for a fair comparison and does not depend on SEED/TEST_FRACTION being correct.
"""
import os, json, random, shutil, time
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm

exec(open(os.path.join(os.path.dirname(__file__), "00_settings.py")).read())

FORCE = False   # set True to redo the conversion from scratch
raw = Path(RAW_DS)
if FORCE and raw.exists():
    shutil.rmtree(raw)

if (raw / "dataset.json").exists():
    print("Already converted - skipping (set FORCE = True to redo).")
else:
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
        for leftover in (raw / sub).glob("*_partial.nii.gz"):
            leftover.unlink()

    data_files = sorted(Path(CAS_DIR, "data").glob("*.nii.gz"))
    cases = sorted(p.name[:-7] for p in data_files)
    mask_dir = Path(CAS_DIR, "mask")
    missing = [c for c in cases if not (mask_dir / f"{c}.nii.gz").exists()]
    assert not missing, f"No mask found for cases: {missing[:5]}"
    print(f"Found {len(cases)} CAS2023 cases with matching masks.")

    split_file = Path(WORK_DIR, "data_split.json")
    if split_file.exists():
        split = json.load(open(split_file))
        train_ids, test_ids = split["train"], split["test"]
        print(f"Reusing existing split from {split_file} (train={len(train_ids)}, test={len(test_ids)}) "
              f"-- SAME split as your teacher run.")
    else:
        rng = random.Random(SEED); shuffled = cases[:]; rng.shuffle(shuffled)
        n_test = max(1, round(TEST_FRACTION * len(shuffled)))
        test_ids, train_ids = sorted(shuffled[:n_test]), sorted(shuffled[n_test:])
        print("No existing split found -- creating a NEW one (this will NOT match an earlier teacher run "
              "if one was done with different raw data / fold).")

    jobs = [("train", c) for c in train_ids] + [("test", c) for c in test_ids]

    def write_atomic(write_fn, dst):
        tmp = dst.with_name(dst.name.replace(".nii.gz", "_partial.nii.gz"))
        write_fn(tmp)
        os.replace(tmp, dst)

    def roi_crop_bounds(mask_arr, margin, vol_shape):
        nz = np.argwhere(mask_arr > 0)
        if len(nz) == 0:
            return (0, 0, 0), vol_shape
        mins = nz.min(axis=0) - margin
        maxs = nz.max(axis=0) + margin + 1
        mins = np.clip(mins, 0, None)
        maxs = np.minimum(maxs, vol_shape)
        return tuple(mins.tolist()), tuple(maxs.tolist())

    fractions, crop_sizes, n_train, n_done, n_skipped, t0 = [], [], 0, 0, 0, time.time()
    bar = tqdm(jobs, desc="Converting", unit="case")
    for split, c in bar:
        img_dir = raw / ("imagesTr" if split == "train" else "imagesTs")
        lab_dir = raw / ("labelsTr" if split == "train" else "labelsTs")
        dst_img = img_dir / f"CAS2023_{c}_0000.nii.gz"
        dst_lab = lab_dir / f"CAS2023_{c}.nii.gz"
        if dst_img.exists() and dst_lab.exists():
            n_train += (split == "train"); n_done += 1
            continue

        src_img = f"{CAS_DIR}/data/{c}.nii.gz"
        src_lab = f"{CAS_DIR}/mask/{c}.nii.gz"
        im = sitk.ReadImage(src_img)
        lb = sitk.ReadImage(src_lab)
        if im.GetSize() != lb.GetSize():
            print(f"[skip] case {c}: scan size {im.GetSize()} != mask size {lb.GetSize()}")
            n_skipped += 1
            continue

        im_arr = sitk.GetArrayFromImage(im)
        lb_arr = (sitk.GetArrayFromImage(lb) > 0).astype(np.uint8)

        mins, maxs = roi_crop_bounds(lb_arr, ROI_MARGIN_VOXELS, np.array(lb_arr.shape))
        sl = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
        im_crop, lb_crop = im_arr[sl], lb_arr[sl]
        crop_sizes.append(im_crop.shape)

        im_out = sitk.GetImageFromArray(im_crop)
        im_out.SetSpacing(im.GetSpacing())
        im_out.SetDirection(im.GetDirection())
        origin_shift_zyx = [mins[i] for i in range(3)]
        offset = im.TransformContinuousIndexToPhysicalPoint(
            [float(origin_shift_zyx[2]), float(origin_shift_zyx[1]), float(origin_shift_zyx[0])])
        im_out.SetOrigin(offset)
        lb_out = sitk.GetImageFromArray(lb_crop)
        lb_out.CopyInformation(im_out)

        write_atomic(lambda p: sitk.WriteImage(im_out, str(p), True, 1), dst_img)
        write_atomic(lambda p: sitk.WriteImage(lb_out, str(p), True, 1), dst_lab)

        fractions.append(lb_crop.mean()); n_train += (split == "train"); n_done += 1
        bar.set_postfix(sec_per_case=f"{(time.time() - t0) / max(len(fractions), 1):.1f}")

    with open(raw / "dataset.json", "w") as f:
        json.dump({"channel_names": {"0": "MRA"}, "labels": {"background": 0, "vessel": 1},
                   "numTraining": n_train, "file_ending": ".nii.gz"}, f, indent=2)

    os.makedirs(WORK_DIR, exist_ok=True)
    with open(f"{WORK_DIR}/data_split.json", "w") as f:
        json.dump({"seed": SEED, "train": train_ids, "test": test_ids}, f, indent=2)

    print(f"\nTraining cases: {n_train} | held-out test cases: {n_done - n_train} | skipped: {n_skipped}")
    if fractions:
        print("Vessel fraction after crop: mean %.2f%%, min %.2f%%, max %.2f%%" %
              (100 * np.mean(fractions), 100 * np.min(fractions), 100 * np.max(fractions)))
    print("Split saved to", f"{WORK_DIR}/data_split.json")
