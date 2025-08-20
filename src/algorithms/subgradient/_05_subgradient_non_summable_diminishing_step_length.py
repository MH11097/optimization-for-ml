from .base_subgradient import BaseSubgradient
import numpy as np


class SubgradientNonSummableDiminishingStepLength(BaseSubgradient):

    def get_step_size(
        self,
        current_subgradient_vector,
        current_iteration,
        *args,
        **kwargs,
    ):
        GAMMA_0 = 1
        # diminishing sequence gamma_k
        gamma_k = GAMMA_0 / np.sqrt(current_iteration)

        # avoid division by zero
        subgradient_norm = np.dot(
            current_subgradient_vector, current_subgradient_vector
        )
        if subgradient_norm == 0:
            step_size = 0.0
        else:
            step_size = gamma_k / subgradient_norm

        return step_size
