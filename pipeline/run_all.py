"""
One command to run the whole pipeline, safe to re-run after any disconnect --
every step either skips itself (already done) or resumes (training, via --c).

Usage (from a single Colab cell):
    !python /content/drive/MyDrive/cas2023/pipeline/run_all.py
    !python /content/drive/MyDrive/cas2023/pipeline/run_all.py --stage train_student
    !python /content/drive/MyDrive/cas2023/pipeline/run_all.py --from train_teacher
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    ("convert",            "01_convert_data.py",        []),
    ("preprocess_teacher",  "02_preprocess_teacher.py",  []),
    ("train_teacher",       "03_train_teacher.py",       []),
    ("predict_eval_teacher","07_predict_evaluate.py",    ["teacher"]),
    ("stats_teacher",       "08_gather_stats.py",        ["teacher"]),
    ("make_student_plan",   "04_make_student_plan.py",   []),
    ("preprocess_student",  "05_preprocess_student.py",  []),
    ("train_student",       "06_train_student.py",       []),
    ("predict_eval_student","07_predict_evaluate.py",    ["student"]),
    ("stats_student",       "08_gather_stats.py",        ["student"]),
    ("install_trainers",    "00b_install_trainers.py",   []),
    ("train_c1",            "09_train_kd_variant.py",    ["c1"]),
    ("predict_eval_c1",     "07_predict_evaluate.py",    ["c1"]),
    ("stats_c1",            "08_gather_stats.py",        ["c1"]),
    ("train_c2",            "09_train_kd_variant.py",    ["c2"]),
    ("predict_eval_c2",     "07_predict_evaluate.py",    ["c2"]),
    ("stats_c2",            "08_gather_stats.py",        ["c2"]),
    ("train_c3",            "09_train_kd_variant.py",    ["c3"]),
    ("predict_eval_c3",     "07_predict_evaluate.py",    ["c3"]),
    ("stats_c3",            "08_gather_stats.py",        ["c3"]),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", help="run only this one stage")
    ap.add_argument("--from", dest="from_stage", help="run this stage and every stage after it")
    args = ap.parse_args()

    names = [s[0] for s in STAGES]
    if args.stage:
        todo = [s for s in STAGES if s[0] == args.stage]
    elif args.from_stage:
        idx = names.index(args.from_stage)
        todo = STAGES[idx:]
    else:
        todo = STAGES

    for name, script, extra_args in todo:
        print(f"\n{'='*70}\n== {name}\n{'='*70}")
        cmd = [sys.executable, os.path.join(HERE, script)] + extra_args
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nStage '{name}' FAILED (exit {result.returncode}).")
            print(f"Fix the error above, then re-run with: --from {name}")
            sys.exit(result.returncode)

    print("\nAll stages complete.")
    print(f"Available stage names: {', '.join(names)}")

if __name__ == "__main__":
    main()
