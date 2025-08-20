import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.optimization_utils import (
    tinh_gradient_hoi_quy_tuyen_tinh,
    tinh_ma_tran_hessian_hoi_quy_tuyen_tinh,
    giai_he_phuong_trinh_tuyen_tinh,
    kiem_tra_positive_definite,
    tinh_condition_number,
    in_thong_tin_ma_tran,
    in_thong_tin_gradient,
    du_doan,
    tinh_mse,
)


class SubgradientConstantStepSize:
    """
    Bộ tối ưu hóa Subgradient cho Hồi quy tuyến tính
    """

    def __init__(
        self,
        lambda_penalty: float = 0.5,
        regularization: float = 1e-8,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        armijo_c1: float = 1e-4,
        backtrack_rho: float = 0.8,
        max_line_search_iter: int = 50,
        verbose: bool = False,
    ):
        self.lambda_penalty = lambda_penalty
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.armijo_c1 = armijo_c1
        self.backtrack_rho = backtrack_rho
        self.max_line_search_iter = max_line_search_iter
        self.verbose = verbose

        # Tracking results
        self.cost_history: List[float] = []
        self.gradient_norms: List[float] = []
        self.step_sizes: List[float] = []
        self.line_search_iterations: List[int] = []
        self.condition_numbers: List[float] = []
        self.convergence_info: Dict = {}

    def _compute_cost(
        self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
    ) -> float:
        """Tính cost function (MSE với regularization)"""
        predictions = du_doan(X, weights, bias)
        mse = np.mean((y - predictions) ** 2) / 2

        # Thêm regularization term
        regularization_term = self.lambda_penalty * np.linalg.norm(weights, 1)

        return mse + regularization_term

    def get_step_size(self, current_subgradient_vector, current_iteration: int):
        FIXED_STEP_SIZE = 0.05
        return FIXED_STEP_SIZE

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_weights: Optional[np.ndarray] = None,
        initial_bias: float = 0.0,
    ) -> Dict:
        # Init
        n_samples, n_features = X.shape
        if initial_weights is None:
            weights = np.zeros(n_features)
        else:
            weights = initial_weights.copy()
        bias = initial_bias

        losses = []

        # Main optimization loop
        for iteration in range(1, self.max_iterations + 1):
            # Gradient of squared loss
            grad = (1 / n_samples) * X.T @ (X @ weights - y)

            # Subgradient of L1 norm
            subgrad = np.sign(weights)
            # (np.sign(0) = 0, which is valid since subgradient at 0 is [-1,1])

            # Full subgradient
            full_subgrad = grad + self.lambda_penalty * subgrad

            # Step size
            step_size = self.get_step_size(
                current_subgradient_vector=full_subgrad, current_iteration=iteration
            )

            # Update weights
            weights = weights - step_size * full_subgrad

            # Tính cost
            current_cost = self._compute_cost(X, y, weights, bias)
            losses.append(current_cost)

        results = {
            "weights": weights,
            "min_loss": min(losses),
            "losses": losses,
        }
        return results


if __name__ == "__main__":
    # Test với data giả
    np.random.seed(42)

    # Generate test data
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    noise = 0.1 * np.random.randn(n_samples)
    y = X @ true_weights + true_bias + noise

    results = SubgradientConstantStepSize().optimize(X=X, y=y)

    # Plot losses over iterations
    losses = results["losses"]
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Iteration")
    plt.grid(True)
    plt.show()

    print(results)
