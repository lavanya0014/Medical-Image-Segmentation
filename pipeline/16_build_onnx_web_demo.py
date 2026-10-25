"""
Builds a SECOND, separate Hugging Face Static Space that actually runs the
C3 student model in the visitor's browser via ONNX Runtime Web (WASM) --
unlike 14_build_huggingface_space.py, which only shows pre-rendered PNGs.

This is the real "phone demo": the .onnx file and a couple of precomputed
input patches are shipped as static files, onnxruntime-web (from a CDN) loads
the model client-side, inference runs on-device, and the page measures and
displays the actual client-side latency -- a real number for the deck's
"Latency (ms): inference time per scan on the phone" efficiency metric.

Prerequisites (run first, in order):
    python 11_export_selected_c3_onnx.py       # produces deployment/c3_warmstart_3d_fullres.onnx
    python 15_prepare_web_demo_assets.py        # produces deployment/web_demo_assets/*

Run:
    python 16_build_onnx_web_demo.py
Then upload the output folder's contents to a new Hugging Face Space
(Space SDK: "Static").
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/cas2023")
ONNX_PATH = PROJECT_ROOT / "deployment" / "c3_warmstart_3d_fullres.onnx"
ONNX_META_PATH = ONNX_PATH.with_suffix(".json")
ASSETS_DIR = PROJECT_ROOT / "deployment" / "web_demo_assets"
OUTPUT_DIR = PROJECT_ROOT / "deployment" / "hf_space_c3_onnx_web_demo"

ONNXRUNTIME_WEB_VERSION = "1.19.2"

README = """---
title: CAS2023 C3 On-Device Segmentation Demo
emoji: \U0001F9E0
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
---

# CAS2023 C3 Knowledge-Distillation -- On-Device Demo

This page downloads the compressed C3 (combined-KD) student model as ONNX
and runs segmentation **entirely in your browser** using ONNX Runtime Web
(WASM) -- nothing is sent to a server. It measures and reports the actual
inference time on the device you're viewing it from.

Input patches are precomputed crops (64x192x192 voxels) from already
nnU-Net-preprocessed CAS2023 test scans, since full DICOM/NIfTI
preprocessing in the browser is out of scope for this demo (per the
project's own "phone demo" design note: full scans are hundreds of MB, so
the demo uses pre-cropped patches).

This is a research/course demonstration, not a clinical decision-support
tool.
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CAS2023 C3 On-Device Segmentation Demo</title>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@__ORT_VERSION__/dist/ort.min.js"></script>
<style>
  body { margin: 0; background: #10131a; color: #f4f6fa; font-family: Arial, sans-serif; }
  main { max-width: 900px; margin: 0 auto; padding: 28px 20px 48px; }
  h1 { margin-bottom: 4px; font-size: 22px; }
  p { color: #c6cbd6; line-height: 1.5; }
  select, button { padding: 12px; border-radius: 8px; border: 1px solid #596275; background: #202633; color: white; font-size: 16px; }
  select { width: 100%; margin-bottom: 12px; }
  button { cursor: pointer; width: 100%; font-weight: 700; background: #5b6cff; border-color: #5b6cff; }
  button:disabled { opacity: 0.5; cursor: wait; }
  canvas { display: block; max-width: 100%; margin: 20px auto; border-radius: 10px; border: 1px solid #394153; background: #000; }
  .metrics { background: #202633; padding: 16px; border-radius: 10px; font-size: 16px; line-height: 1.8; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-weight: 600; margin-top: 14px; font-size: 14px; }
  .dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }
  .status { color: #8ea0ff; font-size: 14px; min-height: 20px; }
</style>
</head>
<body><main>
  <h1>CAS2023 C3 Knowledge-Distillation -- On-Device Demo</h1>
  <p>Runs the compressed student model directly in this browser via ONNX Runtime Web. No data leaves your device.</p>

  <label for="case"><strong>Example test scan (precomputed patch)</strong></label>
  <select id="case"></select>
  <button id="run">Load model &amp; run inference on-device</button>
  <div class="status" id="status"></div>

  <canvas id="canvas" width="192" height="192"></canvas>

  <div class="legend">
    <span><i class="dot" style="background:#ffd700"></i>Correct overlap</span>
    <span><i class="dot" style="background:#eb3737"></i>Prediction only</span>
    <span><i class="dot" style="background:#2dd255"></i>Ground truth only</span>
  </div>

  <div class="metrics" id="metrics">Select a case and click Run.</div>
</main>
<script>
const MANIFEST = __MANIFEST__;
const PATCH = MANIFEST.patch_size;      // [Z, Y, X]
const MID = Math.floor(PATCH[0] / 2);
const PLANE = PATCH[1] * PATCH[2];

const caseSelect = document.getElementById('case');
Object.keys(MANIFEST.cases).forEach(name => caseSelect.add(new Option(name, name)));

let session = null;
const statusEl = document.getElementById('status');
const metricsEl = document.getElementById('metrics');
const runBtn = document.getElementById('run');
const ctx = document.getElementById('canvas').getContext('2d');

async function fetchFloat32(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  return new Float32Array(buf);
}
async function fetchUint8(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  return new Uint8Array(buf);
}

function renderOverlay(imageSlice, predSlice, labelSlice) {
  let lo = Infinity, hi = -Infinity;
  for (const v of imageSlice) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const range = Math.max(hi - lo, 1e-6);
  const out = ctx.createImageData(PATCH[2], PATCH[1]);
  for (let i = 0; i < PLANE; i++) {
    const gray = Math.max(0, Math.min(255, Math.round(((imageSlice[i] - lo) / range) * 255)));
    const pred = predSlice[i] > 0, label = labelSlice[i] > 0;
    let r = gray, g = gray, b = gray;
    if (pred && label) { r = 255; g = 215; b = 0; }
    else if (pred && !label) { r = 235; g = 55; b = 55; }
    else if (!pred && label) { r = 45; g = 210; b = 85; }
    const o = i * 4;
    out.data[o] = r; out.data[o+1] = g; out.data[o+2] = b; out.data[o+3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

runBtn.addEventListener('click', async () => {
  runBtn.disabled = true;
  const caseName = caseSelect.value;
  const info = MANIFEST.cases[caseName];
  try {
    if (!session) {
      statusEl.textContent = 'Downloading model (once per session)...';
      const t0 = performance.now();
      session = await ort.InferenceSession.create(MANIFEST.onnx_file, { executionProviders: ['wasm'] });
      statusEl.textContent = `Model loaded in ${(performance.now() - t0).toFixed(0)} ms.`;
    }

    statusEl.textContent = 'Fetching precomputed input patch...';
    const [imageData, labelData] = await Promise.all([
      fetchFloat32(info.image_file),
      fetchUint8(info.label_file),
    ]);

    statusEl.textContent = 'Running inference on-device...';
    const inputTensor = new ort.Tensor('float32', imageData, [1, 1, ...PATCH]);
    const t1 = performance.now();
    const results = await session.run({ [MANIFEST.input_name]: inputTensor });
    const latencyMs = performance.now() - t1;

    const output = results[MANIFEST.output_name];
    // output shape: [1, numClasses, Z, Y, X] -- argmax over the class axis
    const numClasses = output.dims[1];
    const voxelsPerVolume = PATCH[0] * PATCH[1] * PATCH[2];
    const predLabels = new Uint8Array(voxelsPerVolume);
    const od = output.data;
    for (let v = 0; v < voxelsPerVolume; v++) {
      let best = 0, bestVal = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        const val = od[c * voxelsPerVolume + v];
        if (val > bestVal) { bestVal = val; best = c; }
      }
      predLabels[v] = best > 0 ? 1 : 0;
    }

    let intersection = 0, predSum = 0, labelSum = 0;
    for (let v = 0; v < voxelsPerVolume; v++) {
      const p = predLabels[v], l = labelData[v] > 0 ? 1 : 0;
      if (p && l) intersection++;
      predSum += p; labelSum += l;
    }
    const dice = (2 * intersection) / Math.max(predSum + labelSum, 1);

    const sliceOffset = MID * PLANE;
    renderOverlay(
      imageData.subarray(sliceOffset, sliceOffset + PLANE),
      predLabels.subarray(sliceOffset, sliceOffset + PLANE),
      labelData.subarray(sliceOffset, sliceOffset + PLANE),
    );

    statusEl.textContent = '';
    metricsEl.innerHTML = `<strong>${caseName}</strong><br>` +
      `On-device inference latency: <strong>${latencyMs.toFixed(1)} ms</strong><br>` +
      `On-device Dice (this patch): <strong>${dice.toFixed(4)}</strong><br>` +
      `Model: C3 combined KD student, ONNX Runtime Web (WASM), runs entirely in this browser.`;
  } catch (err) {
    statusEl.textContent = '';
    metricsEl.textContent = 'Error: ' + err.message;
    console.error(err);
  } finally {
    runBtn.disabled = false;
  }
});
</script>
</body></html>"""


def main():
    if not ONNX_PATH.is_file():
        raise FileNotFoundError(f"{ONNX_PATH} not found -- run 11_export_selected_c3_onnx.py first.")
    manifest_path = ASSETS_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found -- run 15_prepare_web_demo_assets.py first.")

    manifest = json.loads(manifest_path.read_text())
    onnx_meta = json.loads(ONNX_META_PATH.read_text()) if ONNX_META_PATH.is_file() else {}

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    assets_out = OUTPUT_DIR / "assets"
    assets_out.mkdir(parents=True)

    shutil.copy(ONNX_PATH, OUTPUT_DIR / ONNX_PATH.name)
    for case, info in manifest["cases"].items():
        shutil.copy(ASSETS_DIR / info["image_file"], assets_out / info["image_file"])
        shutil.copy(ASSETS_DIR / info["label_file"], assets_out / info["label_file"])
        info["image_file"] = f"assets/{info['image_file']}"
        info["label_file"] = f"assets/{info['label_file']}"

    manifest["onnx_file"] = ONNX_PATH.name
    manifest["onnx_size_mb"] = onnx_meta.get("onnx_size_mb")

    html = HTML_TEMPLATE.replace("__MANIFEST__", json.dumps(manifest)).replace("__ORT_VERSION__", ONNXRUNTIME_WEB_VERSION)
    (OUTPUT_DIR / "index.html").write_text(html)
    (OUTPUT_DIR / "README.md").write_text(README)

    zip_path = shutil.make_archive(str(OUTPUT_DIR), "zip", OUTPUT_DIR)
    total_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / (1024**2)

    print(f"Built on-device demo Space: {OUTPUT_DIR}  ({total_mb:.1f} MB total)")
    print(f"Zip for upload: {zip_path}")
    print("Create a new Hugging Face Space (SDK: Static) and upload this folder's contents")
    print("(index.html, README.md, the .onnx file, and assets/). The model and inference")
    print("run entirely client-side once loaded -- including on a phone browser.")


if __name__ == "__main__":
    main()
