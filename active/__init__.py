from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .entropy_selector import EntropySelector
from .sfm_uncertainty_selector import SFMUncertaintySelector

methods_dict = {"rand": RandSelector, "H_reg": HRegSelector, "entropy": EntropySelector, "sfm_uncertainty": SFMUncertaintySelector}