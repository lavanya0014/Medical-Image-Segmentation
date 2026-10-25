"""C1: output-level distillation only."""
from .nnUNetTrainerKDStudent import nnUNetTrainerKDStudent


class nnUNetTrainerKD_C1_OutputOnly(nnUNetTrainerKDStudent):
    DEFAULT_ALPHA = 0.0
    DEFAULT_USE_CLDICE = False
