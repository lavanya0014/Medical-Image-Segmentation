"""
Builds the final novelty-assessment table (+ LaTeX rows) and a paired per-case comparison
against the scratch student B5 from the files the pipeline already wrote.

    !python /content/drive/MyDrive/cas2023/pipeline/15_novelty_table_v2.py

Optional phone timings: create  WORK_DIR/deployment/phone_latency_w40.json  like
  {"student_w40_3d_fullres.onnx": [9800, 9400, 9500],
   "c1_w40_3d_fullres.onnx":      [9700, 9300, 9400],
   "c1_w40_3d_fullres_int8.onnx": [7000, 6800, 6900]}
(milliseconds from the Space's "On-device inference latency" line; the FIRST value of each
list is treated as warm-up and dropped).
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import novelty_env_v2  # noqa: F401
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_settings.py")).read())

RA, DEP = Path(WORK_DIR) / "report_assets", Path(WORK_DIR) / "deployment"
NAMES = {"student": "B5 (scratch)", "c1": "C1 output KD", "c2": "C2 feature KD", "c3": "C3 combined"}

def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None

def per_case(summary):
    out = {}
    for e in summary["metric_per_case"]:
        out[os.path.basename(e["reference_file"])] = e["metrics"]["1"]["Dice"]
    return out

phone = load(DEP / f"phone_latency{RUN_TAG}.json") or {}
TRUE = load(RA / "true_params.json") or {}   # written by 16_true_params_v2.py
def phone_ms(fname):
    v = phone.get(fname)
    return f"{np.mean(v[1:] if len(v) > 1 else v):.0f}" if v else "--"

rows, cases = [], {}
# teacher + old half-width student from the earlier experiment (no tag), for context
for tag_key, label, summ, stats, onnx in [
    ("teacher", "Teacher (nnU-Net)", RA / "teacher_test_summary.json", RA / "teacher_stats.json", None),
]:
    s, st = load(summ), load(stats)
    if s and st:
        rows.append((label, TRUE.get("teacher", st.get(f"{tag_key}_params_million")), "--", s["foreground_mean"]["Dice"], s["foreground_mean"]["IoU"], "--"))

for w in ("student", "c1", "c2", "c3"):
    s = load(RA / f"{w}{RUN_TAG}_test_summary.json"); st = load(RA / f"{w}{RUN_TAG}_stats.json")
    if not (s and st):
        continue
    fn = f"{w}{RUN_TAG}_3d_fullres.onnx"
    meta = load(DEP / fn.replace(".onnx", ".json"))
    n_true = meta["params_million"] if meta else TRUE.get(f"student{RUN_TAG}", 4.92)
    rows.append((NAMES[w], n_true, f"{meta['onnx_size_mb']:.1f}" if meta else "--",
                 s["foreground_mean"]["Dice"], s["foreground_mean"]["IoU"], phone_ms(fn)))
    cases[w] = per_case(s)
    q = load(DEP / f"{w}{RUN_TAG}_quant_check.json")
    qs = load(RA / f"{w}{RUN_TAG}_onnx_int8_summary.json")
    if q and q.get("runs_ok") and qs:
        rows.append((NAMES[w] + " + int8", n_true, f"{q['int8_mb']:.1f}",
                     qs["foreground_mean"]["Dice"], qs["foreground_mean"]["IoU"], phone_ms(fn.replace(".onnx", "_int8.onnx"))))

print(f"{'Model':<24}{'Params(M)':>10}{'ONNX MB':>9}{'Dice':>8}{'IoU':>8}{'Phone ms':>10}")
for r in rows:
    print(f"{r[0]:<24}{r[1]:>10.2f}{r[2]:>9}{r[3]:>8.4f}{r[4]:>8.4f}{r[5]:>10}")

print("\nLaTeX rows:")
for r in rows:
    print(f"{r[0]} & {r[1]:.2f} & {r[2]} & {r[3]:.4f} & {r[4]:.4f} & {r[5]} \\\\")

if "student" in cases:
    print("\nPaired per-case Dice difference vs B5 (positive = KD better):")
    base = cases["student"]
    for w, d in cases.items():
        if w == "student":
            continue
        ks = sorted(set(base) & set(d))
        diff = np.array([d[k] - base[k] for k in ks])
        print(f"  {NAMES[w]:<16} mean {diff.mean():+.4f}  std {diff.std(ddof=1):.4f}  "
              f"better on {(diff > 0).sum()}/{len(diff)} cases")
    print("  (If |mean diff| is smaller than std/sqrt(n), say 'no clear difference' in the report.)")
