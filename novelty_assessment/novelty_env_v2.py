"""
Settings for the SECOND experiment: ~5M-parameter student (width scale 0.4).
Does not touch the first (scale 0.25 / 1.9M) run, which used novelty_env.py and the "_5m" tag.
Run 04b_count_params_v2.py first and change STUDENT_WIDTH_SCALE if another scale is closer to 5.0M.
"""
import os

os.environ.setdefault("STUDENT_WIDTH_SCALE", "0.4")
os.environ.setdefault("STUDENT_PLANS_NAME",  "nnUNetPlansStudentW40")
os.environ.setdefault("RUN_TAG",             "_w40")
os.environ.setdefault("KD_WARM_START",       "0")     # cold start for every model
os.environ.setdefault("KD_WIDTH_FACTOR",     "1.0")
os.environ.setdefault("KD_NUM_EPOCHS",       "5")
