"""Aggregate available C1-C3 (and optional C4) per-case evaluation summaries."""
import argparse
import csv
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
VARIANTS = ("c1", "c2", "c3", "c4")
METRICS = ("dice", "iou", "precision", "recall", "cldice", "hd95_mm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-c4", action="store_true", help="Include C4 if its summary exists")
    args = parser.parse_args()
    settings = runpy.run_path(str(HERE / "00_settings.py"))
    output_dir = Path(settings["WORK_DIR"]) / "kd_evaluation"
    variants = ("c1", "c2", "c3", "c4") if args.include_c4 else ("c1", "c2", "c3")

    rows = []
    missing = []
    for variant in variants:
        summary_path = output_dir / variant / "summary_metrics.json"
        if not summary_path.is_file():
            missing.append(variant)
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        row = {
            "variant": variant,
            "trainer": summary.get("trainer", ""),
            "num_test_cases": summary.get("num_test_cases", 0),
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = summary.get("metrics_mean", {}).get(metric)
            row[f"std_{metric}"] = summary.get("metrics_std", {}).get(metric)
        rows.append(row)

    if not rows:
        raise SystemExit(
            f"No KD evaluation summaries were found under {output_dir}. Run predict_eval_c1/c2/c3 first."
        )

    fields = ["variant", "trainer", "num_test_cases"] + [
        key for metric in METRICS for key in (f"mean_{metric}", f"std_{metric}")
    ]
    csv_path = output_dir / "kd_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "kd_comparison.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "models": rows,
            "not_evaluated": missing,
            "metrics_note": "Foreground metrics aggregate all nonzero labels; HD95 is in mm.",
            "baseline_note": "Teacher and scratch-student summaries remain in the original pipeline outputs and are not guessed or copied here.",
        }, f, indent=2)

    print("\nKD comparison (means across held-out cases):")
    for row in rows:
        print(
            f"{row['variant'].upper():>3} | cases={row['num_test_cases']} | "
            f"Dice={row['mean_dice']} | IoU={row['mean_iou']} | "
            f"clDice={row['mean_cldice']} | HD95(mm)={row['mean_hd95_mm']}"
        )
    if missing:
        print("Not yet evaluated:", ", ".join(v.upper() for v in missing))
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
