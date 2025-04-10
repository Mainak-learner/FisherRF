from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .entropy_selector import EntropySelector
from .var_uncertainty import VarUncertaintySelector

methods_dict = {"rand": RandSelector, "H_reg": HRegSelector, "entropy": EntropySelector, "var_uncertainty": VarUncertaintySelector}