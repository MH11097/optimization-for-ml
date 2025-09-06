from .base_subgradient import BaseSubgradient
import numpy as np


class SubgradientConstantStepLength(BaseSubgradient):

    def get_step_size(
        self,
        current_subgradient_vector,
        *args,
        **kwargs,
    ):
        subgrad_norm = np.linalg.norm(x=current_subgradient_vector)
        FIXED_STEP_LENGTH = 0.04
        return FIXED_STEP_LENGTH / subgrad_norm
