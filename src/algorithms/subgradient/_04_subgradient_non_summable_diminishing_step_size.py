from .base_subgradient import BaseSubgradient
import numpy as np


class SubgradientNonSummableDiminishingStepSize(BaseSubgradient):

    def get_step_size(
        self,
        current_iteration,
        *args,
        **kwargs,
    ):
        FIXED_A = 1
        return FIXED_A / np.sqrt(current_iteration)
