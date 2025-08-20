from .base_subgradient import BaseSubgradient
import numpy as np


class SubgradientSquareSummable(BaseSubgradient):

    def get_step_size(
        self,
        current_iteration,
        *args,
        **kwargs,
    ):
        FIXED_A = 1
        FIXED_B = 0
        return FIXED_A / (FIXED_B + current_iteration)
