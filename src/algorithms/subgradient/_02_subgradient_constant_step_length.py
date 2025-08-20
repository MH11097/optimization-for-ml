from ._01_subgradient_constant_step_size import SubgradientConstantStepSize
import matplotlib.pyplot as plt
import numpy as np


class SubgradientConstantStepLength(SubgradientConstantStepSize):

    def get_step_size(self, current_subgradient_vector):
        subgrad_norm = np.linalg.norm(x=current_subgradient_vector)
        FIXED_STEP_LENGTH = 0.05
        return FIXED_STEP_LENGTH / subgrad_norm
