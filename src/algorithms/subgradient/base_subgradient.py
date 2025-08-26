import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
import os
import json
from abc import ABC, abstractmethod
from copy import deepcopy

# Add the src directory to path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.optimization_utils import (
    du_doan,
    danh_gia_mo_hinh,
    in_ket_qua_danh_gia,
)
from utils.visualization_utils import (
    ve_duong_hoi_tu,
    ve_duong_dong_muc_optimization,
    ve_du_doan_vs_thuc_te,
)


class BaseSubgradient(ABC):
    """
    Bộ tối ưu hóa Subgradient cho Hồi quy tuyến tính
    """

    def __init__(
        self,
        lambda_penalty: float = 0.1,
        max_iterations: int = 750,
        tolerance: float = 1e-8,
    ):
        self.lambda_penalty = lambda_penalty
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Chọn loss function và gradient function
        self.loss_func = self._compute_cost
        self.grad_func = self._compute_gradient

        # Khởi tạo các thuộc tính lưu kết quả
        self.weights = None
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []
        self.training_time = 0
        self.converged = False
        self.final_iteration = 0

    def _compute_cost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        bias: float = 0,
    ) -> float:
        """Tính cost function (MSE với regularization)"""
        predictions = du_doan(X, weights, bias)
        mse = np.mean((y - predictions) ** 2) / 2

        # Thêm regularization term
        regularization_term = self.lambda_penalty * np.linalg.norm(weights, 1)

        return mse + regularization_term

    def _compute_gradient(
        self,
        n_samples: int,
        X: np.ndarray,
        y: np.ndarray,
        weights: list,
    ):
        # Gradient of squared loss
        grad = (1 / n_samples) * X.T @ (X @ weights - y)

        # Subgradient of L1 norm
        subgrad = np.sign(weights)
        # (np.sign(0) = 0, which is valid since subgradient at 0 is [-1,1])

        # Full subgradient
        full_subgrad = grad + self.lambda_penalty * subgrad

        return full_subgrad

    @abstractmethod
    def get_step_size(self, *args, **kwargs):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Huấn luyện model với dữ liệu X, y

        Returns:
        - dict: Kết quả training bao gồm weights, loss_history, etc.
        """
        print(f"Training Subgradient Descent - method {type(self).__name__}")
        print(f"   Lambda penalty: {self.lambda_penalty}")
        print(f"   Max iterations: {self.max_iterations}")
        print(f"   Tolerance: {self.tolerance}")

        # Initialize weights
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        # For subgradient, loss value can increase in later iteration.
        # As a result, we store minimum loss value.
        # We store both lowest value (min_loss_1) and second lowest value (min_loss_2)
        BASE_LOSS_VALUE = 1000
        min_loss_1 = {"iteration": 0, "loss_value": BASE_LOSS_VALUE, "weights": None}
        min_loss_2 = {"iteration": 0, "loss_value": BASE_LOSS_VALUE, "weights": None}

        # Reset histories
        self.loss_history = []
        self.gradient_norms = []
        self.weights_history = []

        start_time = time.time()

        # Main optimization loop
        for iteration in range(1, self.max_iterations + 1):
            # Gradient of squared loss
            loss_value = self.loss_func(X=X, y=y, weights=self.weights)
            gradient = self.grad_func(
                n_samples=n_samples, X=X, y=y, weights=self.weights
            )

            # Step size
            step_size = self.get_step_size(
                current_subgradient_vector=gradient,
                current_iteration=iteration,
            )

            # Update weights
            self.weights = self.weights - step_size * gradient

            # Store history
            self.loss_history.append(loss_value)
            gradient_norm = np.linalg.norm(gradient)
            self.gradient_norms.append(gradient_norm)
            self.weights_history.append(self.weights.copy())

            # Update min loss
            if loss_value < min_loss_1["loss_value"]:
                min_loss_2 = deepcopy(min_loss_1)
                min_loss_1 = {
                    "iteration": iteration,
                    "loss_value": loss_value,
                    "weights": deepcopy(self.weights),
                }

            # Check convergence
            if (
                iteration > 1
                and (min_loss_1["loss_value"] != BASE_LOSS_VALUE)
                and abs(min_loss_1["loss_value"] - min_loss_2["loss_value"])
                < self.tolerance
            ):
                print(f"Converged after {iteration} iterations")
                self.converged = True
                self.final_iteration = iteration
                break

            # Progress update
            if iteration % 50 == 0:
                print(
                    f"Iteration {iteration}: Loss = {loss_value:.6f}, Gradient norm = {gradient_norm:.6f}"
                )

        self.training_time = time.time() - start_time

        if not self.converged:
            print(f"Reached maximum iterations ({self.max_iterations})")
            self.final_iteration = self.max_iterations

        print(f"Training time: {self.training_time:.2f} seconds")

        # Get params with min loss
        self.weights = deepcopy(min_loss_1["weights"])
        self.final_iteration = min_loss_1["iteration"]

        results = {
            "weights": self.weights,
            "loss_history": self.loss_history,
            "gradient_norms": self.gradient_norms,
            "weights_history": self.weights_history,
            "training_time": self.training_time,
            "converged": self.converged,
            "final_iteration": self.final_iteration,
            "min_loss": min_loss_1,
        }

        print(f"Minimum loss result: {min_loss_1}")

        return results

    def predict(self, X):
        """Dự đoán với dữ liệu X"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        return du_doan(X, self.weights, 0)

    def evaluate(self, X_test, y_test):
        """Đánh giá model trên test set"""
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")

        print(f"\\nĐánh giá model trên test set")
        metrics = danh_gia_mo_hinh(self.weights, X_test, y_test)
        in_ket_qua_danh_gia(metrics, self.training_time, f"Subgradient Descent")
        return metrics

    def save_results(self, ten_file, base_dir="data/03_algorithms/subgradient"):
        """
        Lưu kết quả model vào file

        Parameters:
        - ten_file: Tên file/folder để lưu kết quả
        - base_dir: Thư mục gốc để lưu
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")

        # Setup results directory
        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save results.json
        print(f"   Lưu kết quả vào {results_dir}/results.json")
        results_data = {
            "algorithm": f"Subgradient Descent",
            "parameters": {
                "lambda_penalty": self.lambda_penalty,
                "max_iterations": self.max_iterations,
                "tolerance": self.tolerance,
            },
            "training_time": self.training_time,
            "convergence": {
                "converged": self.converged,
                "iterations": self.final_iteration,
                "final_loss": float(self.loss_history[-1]),
                "final_gradient_norm": float(self.gradient_norms[-1]),
            },
        }

        with open(results_dir / "results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        # Save training history
        print(f"   Lưu lịch sử training vào {results_dir}/training_history.csv")
        training_df = pd.DataFrame(
            {
                "iteration": range(len(self.loss_history)),
                "loss": self.loss_history,
                "gradient_norm": self.gradient_norms,
            }
        )
        training_df.to_csv(results_dir / "training_history.csv", index=False)

        print(f"\\n Kết quả đã được lưu vào: {results_dir.absolute()}")
        return results_dir

    def plot_results(
        self, X_test, y_test, ten_file, base_dir="data/03_algorithms/subgradient"
    ):
        """
        Tạo các biểu đồ visualization

        Parameters:
        - X_test, y_test: Dữ liệu test để vẽ predictions
        - ten_file: Tên file/folder để lưu biểu đồ
        - base_dir: Thư mục gốc
        """
        if self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")

        results_dir = Path(base_dir) / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\\n Tạo các biểu đồ visualization")

        # 1. Convergence curves
        print("   Vẽ đường hội tụ")
        ve_duong_hoi_tu(
            self.loss_history,
            self.gradient_norms,
            title=f"Subgradient Descent - Convergence Analysis",
            save_path=str(results_dir / "convergence_analysis.png"),
        )

        # 2. Predictions vs Actual
        print("   Vẽ so sánh dự đoán với thực tế")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(
            y_test,
            y_pred_test,
            title=f"Predictions vs Actual",
            save_path=str(results_dir / "predictions_vs_actual.png"),
        )
        print(f"   Biểu đồ đã được lưu vào: {results_dir.absolute()}")
