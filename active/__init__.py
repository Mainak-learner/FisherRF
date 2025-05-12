from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .entropy_selector import EntropySelector
from .gp_predictor import GPFisherNBVSelector

methods_dict = {"rand": RandSelector, "H_reg": HRegSelector, "entropy": EntropySelector, "gp_fisher": GPFisherNBVSelector}