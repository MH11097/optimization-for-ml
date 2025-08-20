from .base_subgradient import BaseSubgradient
import numpy as np


class SubgradientConstantStepSize(BaseSubgradient):

    def get_step_size(
        self,
        *args,
        **kwargs,
    ):
        FIXED_STEP_SIZE = 0.05
        return FIXED_STEP_SIZE
