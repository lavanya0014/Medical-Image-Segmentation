"""
Same idea as run_all.py, but for the ~5M-student novelty experiment. Safe to re-run after
any Colab disconnect: every training stage resumes (--c), finished stages skip themselves.

    !python /content/drive/MyDrive/cas2023/pipeline/run_novelty.py                    # everything
    !python /content/drive/MyDrive/cas2023/pipeline/run_novelty.py --from train_c1    # resume from a stage
    !python /content/drive/MyDrive/cas2023/pipeline/run_novelty.py --stage export_c2
"""
import argparse, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import novelty_env  # noqa: F401  (sets env vars inherited by every subprocess below)

PREPROC = "PREPROC_REUSE"   # handled inline below

STAGES = [
    ("make_student_plan",   "04_make_student_plan.py",  []),
    ("preprocess_student",  PREPROC,                    []),
    # --- B5: identical 5M student, trained from scratch ---
    ("train_b5",            "06_train_student.py",      []),
    ("predict_eval_b5",     "07_predict_evaluate.py",   ["student"]),
    ("stats_b5",            "08_gather_stats.py",       ["student"]),
    ("install_trainers",    "00b_install_trainers.py",  []),
    # --- C1: output-level KD ---
    ("train_c1",            "09_train_kd_variant.py",   ["c1"]),
    ("predict_eval_c1",     "07_predict_evaluate.py",   ["c1"]),
    ("stats_c1",            "08_gather_stats.py",       ["c1"]),
    # --- C2: bottleneck feature matching ---
    ("train_c2",            "09_train_kd_variant.py",   ["c2"]),
    ("predict_eval_c2",     "07_predict_evaluate.py",   ["c2"]),
    ("stats_c2",            "08_gather_stats.py",       ["c2"]),
    # --- ONNX export for phone timing ---
    ("export_b5",           "11b_export_onnx.py",       ["student"]),
    ("export_c1",           "11b_export_onnx.py",       ["c1"]),
    ("export_c2",           "11b_export_onnx.py",       ["c2"]),
    # --- optional: C3 (combined) at 5M ---
    ("train_c3",            "09_train_kd_variant.py",   ["c3"]),
    ("predict_eval_c3",     "07_predict_evaluate.py",   ["c3"]),
    ("stats_c3",            "08_gather_stats.py",       ["c3"]),
    ("export_c3",           "11b_export_onnx.py",       ["c3"]),
]

def reuse_preprocessed():
    """The student plan keeps the teacher's data_identifier, so the already-preprocessed
    data on Drive is reused as-is -- no need to preprocess again."""
    exec(open(os.path.join(HERE, "00_settings.py")).read(), globals())
    import json
    plans = json.load(open(os.path.join(PP_DS, f"{STUDENT_PLANS_NAME}.json")))
    ident = plans["configurations"][CONFIG]["data_identifier"]
    folder = os.path.join(PP_DS, ident)
    if os.path.isdir(folder) and len(os.listdir(folder)) > 0:
        open(os.path.join(PP_DS, f".preprocess_{STUDENT_PLANS_NAME}_done"), "w").close()
        print(f"Reusing existing preprocessed data: {folder}")
        return 0
    print("Preprocessed folder not found -> running 05_preprocess_student.py")
    return subprocess.run([sys.executable, os.path.join(HERE, "05_preprocess_student.py")]).returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage"); ap.add_argument("--from", dest="from_stage")
    a = ap.parse_args()
    names = [s[0] for s in STAGES]
    todo = [s for s in STAGES if s[0] == a.stage] if a.stage else (STAGES[names.index(a.from_stage):] if a.from_stage else STAGES)
    print("5M experiment settings:", {k: os.environ[k] for k in
          ("STUDENT_WIDTH_SCALE", "STUDENT_PLANS_NAME", "RUN_TAG", "KD_WARM_START", "KD_WIDTH_FACTOR", "KD_NUM_EPOCHS")})
    for name, script, extra in todo:
        print(f"\n{'='*70}\n== {name}\n{'='*70}")
        rc = reuse_preprocessed() if script == PREPROC else subprocess.run(
            [sys.executable, os.path.join(HERE, script)] + extra).returncode
        if rc != 0:
            print(f"\nStage '{name}' FAILED (exit {rc}). Fix the error, then re-run with: --from {name}")
            sys.exit(rc)
    print("\nAll requested stages complete.\nStages:", ", ".join(names))

if __name__ == "__main__":
    main()
