"""C3: supervised and output-level KD, with optional soft clDice."""
from .nnUNetTrainerKDStudent import nnUNetTrainerKDStudent


class nnUNetTrainerKD_C3_Combined(nnUNetTrainerKDStudent):
    DEFAULT_ALPHA = 0.7
    DEFAULT_USE_CLDICE = True
