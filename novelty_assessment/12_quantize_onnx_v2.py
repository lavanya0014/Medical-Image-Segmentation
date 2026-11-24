"""
Dynamic int8 quantization of an exported ONNX model, then a quick size/speed/agreement
check on desktop CPU. Try it on ONE model (your best KD variant).

    !pip -q install onnx onnxruntime
    !python /content/drive/MyDrive/cas2023/pipeline/12_quantize_onnx_v2.py c1
    (optional)  ... 12_quantize_onnx_v2.py c1 --weight int8     # try signed weights if uint8 fails
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import novelty_env_v2  # noqa: F401
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_settings.py")).read())

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

ap = argparse.ArgumentParser()
ap.add_argument("which", nargs="?", default="c1")
ap.add_argument("--weight", choices=["uint8", "int8"], default="uint8")
ap.add_argument("--runs", type=int, default=3)
a = ap.parse_args()

d = Path(WORK_DIR) / "deployment"
src = d / f"{a.which}{RUN_TAG}_3d_fullres.onnx"
dst = d / f"{a.which}{RUN_TAG}_3d_fullres_int8.onnx"
meta = json.load(open(src.with_suffix(".json")))
shape = [1, 1, *meta["patch_size"]]

quantize_dynamic(str(src), str(dst),
                 weight_type=QuantType.QUInt8 if a.weight == "uint8" else QuantType.QInt8,
                 op_types_to_quantize=["Conv"])
s_fp, s_q = src.stat().st_size / 1024**2, dst.stat().st_size / 1024**2
print(f"size: fp32 {s_fp:.2f} MB -> int8 {s_q:.2f} MB  ({s_fp/s_q:.1f}x smaller)")

x = np.random.randn(*shape).astype(np.float32)
res = {"model": a.which, "fp32_mb": s_fp, "int8_mb": s_q, "weight_type": a.weight}

def bench(path):
    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"image": x})[0]            # warm-up
    ts = []
    for _ in range(a.runs):
        t = time.perf_counter(); sess.run(["logits"], {"image": x}); ts.append(time.perf_counter() - t)
    return out, float(np.mean(ts))

try:
    o_fp, t_fp = bench(src)
    o_q, t_q = bench(dst)
    agree = float((o_fp.argmax(1) == o_q.argmax(1)).mean())
    print(f"desktop CPU latency per patch: fp32 {t_fp:.2f}s  int8 {t_q:.2f}s")
    print(f"voxel-wise agreement of predicted labels (random input, rough sanity check): {agree:.4f}")
    res.update(fp32_s=t_fp, int8_s=t_q, label_agreement=agree, runs_ok=True)
except Exception as e:
    print("\nint8 model could not run on ONNX Runtime CPU:", repr(e)[:300])
    print("-> try:  --weight int8  (or report 'quantization attempted, operator not supported' in the report)")
    res.update(runs_ok=False, error=repr(e)[:300])

json.dump(res, open(d / f"{a.which}{RUN_TAG}_quant_check.json", "w"), indent=2)
print("Next: upload", dst.name, "to the Space and time it on the phone; run 13_eval_onnx_dice_v2.py for Dice.")
