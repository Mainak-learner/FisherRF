from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .entropy_selector import EntropySelector
from .primitive_selector import PrimitiveSelector

methods_dict = {"rand": RandSelector, "H_reg": HRegSelector, "entropy": EntropySelector, "uncertainty": PrimitiveSelector}