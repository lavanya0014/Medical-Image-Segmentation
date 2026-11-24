"""
Single place that defines the 5M-student experiment. Every new script imports this
BEFORE 00_settings.py so all of them agree on the plan name / width / tag.
Change STUDENT_WIDTH_SCALE here (after running 04b_count_params.py) if you want
the parameter count closer to 5.0M.
"""
import os

os.environ.setdefault("STUDENT_WIDTH_SCALE", "0.25")              # ~4x fewer params than the 0.5 student
os.environ.setdefault("STUDENT_PLANS_NAME",  "nnUNetPlansStudent5M")  # separate plan -> old results untouched
os.environ.setdefault("RUN_TAG",             "_5m")               # suffix on prediction/summary/stats files
os.environ.setdefault("KD_WARM_START",       "0")                 # cold start: every model learns from scratch
os.environ.setdefault("KD_WIDTH_FACTOR",     "1.0")               # plan is already narrow; no 2nd shrink
os.environ.setdefault("KD_NUM_EPOCHS",       "5")                 # same budget as before
