"""
Additional mixins for optimization algorithms.
This module provides specialized mixins that extend the functionality
beyond the base utils mixins.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from utils.optimization_utils import kiem_tra_dieu_kien_dung
from utils.visualization_utils import (
    ve_duong_hoi_tu,
    ve_du_doan_vs_thuc_te
)
from utils_old.visualization_utils import ve_duong_dong_muc_optimization

class ValidationMixin:
    """
    Mixin cung cấp các phương thức validation cho input data.
    """
    
    @staticmethod
    def validate_input_data(X: np.ndarray, y: np.ndarray) -> None:
        """
        Kiểm tra tính hợp lệ của dữ liệu đầu vào.
        
        Args:
            X: Ma trận đặc trưng
            y: Vector target
            
        Raises:
            ValueError: Nếu dữ liệu không hợp lệ
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X và y phải là numpy arrays")
        
        if X.ndim != 2:
            raise ValueError(f"X phải là ma trận 2D, nhận được {X.ndim}D")
        
        if y.ndim != 1:
            raise ValueError(f"y phải là vector 1D, nhận được {y.ndim}D")
        
        if X.shape[0] != len(y):
            raise ValueError(
                f"Số samples không khớp: X có {X.shape[0]} samples, y có {len(y)} samples"
            )
        
        if X.shape[0] == 0:
            raise ValueError("Dataset không được trống")
        
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Dữ liệu chứa NaN values")
        
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Dữ liệu chứa Inf values")
    
    def validate_parameters(self) -> None:
        """
        Kiểm tra tính hợp lệ của các tham số optimizer.
        
        Raises:
            ValueError: Nếu các tham số không hợp lệ
        """
        if hasattr(self, 'diem_dung') and self.diem_dung <= 0:
            raise ValueError("diem_dung phải > 0")
        
        if hasattr(self, 'max_iterations') and self.max_iterations <= 0:
            raise ValueError("max_iterations phải > 0")
        
        if hasattr(self, 'convergence_check_freq') and self.convergence_check_freq <= 0:
            raise ValueError("convergence_check_freq phải > 0")
        
        if hasattr(self, 'regularization') and self.regularization < 0:
            raise ValueError("regularization phải >= 0")
        
        if hasattr(self, 'ham_loss') and self.ham_loss not in ['ols', 'ridge', 'lasso']:
            raise ValueError("ham_loss phải là 'ols', 'ridge', hoặc 'lasso'")

class ConvergenceMixin:
    """
    Mixin cung cấp các phương thức kiểm tra hội tụ nâng cao.
    """
    
    def check_convergence(self, 
                         gradient_norm: float, 
                         cost_change: float, 
                         iteration: int,
                         loss_value: float) -> Tuple[bool, bool, str]:
        """
        Kiểm tra điều kiện hội tụ sử dụng logic từ utils.
        
        Args:
            gradient_norm: Norm của gradient hiện tại
            cost_change: Sự thay đổi cost so với iteration trước
            iteration: Số iteration hiện tại
            loss_value: Giá trị loss hiện tại
            
        Returns:
            (should_stop, converged, reason)
        """
        return kiem_tra_dieu_kien_dung(
            gradient_norm=gradient_norm,
            cost_change=cost_change,
            iteration=iteration,
            tolerance=getattr(self, 'diem_dung', 1e-5),
            max_iterations=getattr(self, 'max_iterations', 10000),
            loss_value=loss_value,
            weights=getattr(self, 'weights', None)
        )
    
    def check_early_stopping(self, 
                           loss_history: list, 
                           patience: int = 10,
                           min_delta: float = 1e-7) -> bool:
        """
        Kiểm tra early stopping based on loss plateau.
        
        Args:
            loss_history: Lịch sử các giá trị loss
            patience: Số iterations chờ đợi cải thiện
            min_delta: Mức cải thiện tối thiểu
            
        Returns:
            True nếu cần early stopping
        """
        if len(loss_history) < patience + 1:
            return False
        
        recent_losses = loss_history[-(patience + 1):]
        best_recent_loss = min(recent_losses[:-1])
        current_loss = recent_losses[-1]
        
        improvement = best_recent_loss - current_loss
        return improvement < min_delta

class VisualizationMixin:
    """
    Mixin cung cấp các phương thức visualization thống nhất.
    """
    
    def plot_results(self, 
                    X_test: np.ndarray, 
                    y_test: np.ndarray, 
                    setup_name: str,
                    algorithm_dir: str = None,
                    base_dir: str = "data/03_algorithms") -> Path:
        """
        Tạo các biểu đồ visualization thống nhất.
        
        Args:
            X_test: Dữ liệu test features
            y_test: Dữ liệu test targets  
            setup_name: Tên setup (e.g., "130_gd_ols_lr_020")
            algorithm_dir: Tên thư mục thuật toán (e.g., "gradient_descent"). Nếu None, dùng class name.
            base_dir: Thư mục gốc để lưu
            
        Returns:
            Path đến folder chứa các biểu đồ
        """
        if not hasattr(self, 'weights') or self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        # Determine algorithm directory name
        if algorithm_dir is None:
            # Fallback to class name for backward compatibility
            algorithm_dir = getattr(self, '__class__', type(self)).__name__.lower()
        
        # Setup results directory using algorithm_dir/setup_name structure
        results_dir = Path(base_dir) / algorithm_dir / setup_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[PLOTTING] Creating visualizations...")
        
        # Get algorithm name for plot titles
        algorithm_name = algorithm_dir  # Use directory name for consistency
        
        # 1. Convergence curves
        print("   - Creating convergence plot")
        iterations = list(range(0, len(self.loss_history) * 
                               getattr(self, 'convergence_check_freq', 1),
                               getattr(self, 'convergence_check_freq', 1)))
        
        ve_duong_hoi_tu(
            self.loss_history, 
            self.gradient_norms,
            iterations=iterations,
            title=f"{algorithm_name.title()} {getattr(self, 'ham_loss', '').upper()} - Convergence",
            save_path=str(results_dir / "convergence_analysis.png")
        )
        
        # 2. Predictions vs Actual
        print("   - Creating predictions vs actual plot")
        y_pred_test = self.predict(X_test)
        ve_du_doan_vs_thuc_te(
            y_test, 
            y_pred_test,
            title=f"{algorithm_name.title()} {getattr(self, 'ham_loss', '').upper()} - Predictions vs Actual",
            save_path=str(results_dir / "predictions_vs_actual.png")
        )
        
        # 3. Optimization trajectory (if weights history available)
        if hasattr(self, 'weights_history') and len(self.weights_history) > 1:
            print("   - Creating optimization trajectory plot")
            from utils.optimization_utils import add_bias_column
            
            X_test_with_bias = add_bias_column(X_test)
            
            ve_duong_dong_muc_optimization(
                loss_function=getattr(self, 'loss_func'),
                weights_history=self.weights_history,
                X=X_test_with_bias,
                y=y_test,
                title=f"{algorithm_name.title()} {getattr(self, 'ham_loss', '').upper()} - Optimization Path",
                save_path=str(results_dir / "optimization_trajectory.png"),
                original_iterations=getattr(self, 'final_iteration', len(self.weights_history)),
                convergence_check_freq=getattr(self, 'convergence_check_freq', 1),
                max_trajectory_points=50
            )
        
        print(f"   [SUCCESS] Plots saved to: {results_dir}")
        return results_dir
    
    def save_detailed_results(self, 
                            ten_file: str,
                            base_dir: str = "data/03_algorithms",
                            results_data: Optional[Dict[str, Any]] = None) -> Path:
        """
        Lưu kết quả chi tiết vào file.
        
        Args:
            ten_file: Tên file/folder để lưu
            base_dir: Thư mục gốc
            results_data: Dữ liệu kết quả (nếu không có sẽ tự tạo)
            
        Returns:
            Path đến folder chứa kết quả
        """
        if not hasattr(self, 'weights') or self.weights is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        
        algorithm_name = getattr(self, '__class__', type(self)).__name__.lower()
        results_dir = Path(base_dir) / algorithm_name / ten_file
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results data if not provided
        if results_data is None:
            best_results = self._get_best_results()
            results_data = {
                "algorithm": algorithm_name.title(),
                "loss_function": getattr(self, 'ham_loss', '').upper(),
                "training_time": getattr(self, 'training_time', 0.0),
                "converged": getattr(self, 'converged', False),
                "final_iteration": getattr(self, 'final_iteration', 0),
                **best_results
            }
        
        # Save results.json
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save training history
        if hasattr(self, 'loss_history') and self.loss_history:
            training_data = {
                'iteration': range(0, len(self.loss_history) *
                                 getattr(self, 'convergence_check_freq', 1),
                                 getattr(self, 'convergence_check_freq', 1)),
                'loss': self.loss_history,
                'gradient_norm': getattr(self, 'gradient_norms', [])
            }

            # Add learning rate if available, ensuring same sampling frequency
            if hasattr(self, 'learning_rate_history') and self.learning_rate_history:
                # Sample learning rate history at the same frequency as loss history
                convergence_freq = getattr(self, 'convergence_check_freq', 1)
                lr_sampled = []
                for i in range(len(self.loss_history)):
                    lr_index = min((i + 1) * convergence_freq - 1, len(self.learning_rate_history) - 1)
                    lr_sampled.append(self.learning_rate_history[lr_index])
                training_data['learning_rate'] = lr_sampled

            training_df = pd.DataFrame(training_data)
            training_df.to_csv(results_dir / "training_history.csv", index=False)
        
        # Save complexity analysis if available
        if hasattr(self, 'get_complexity_analysis'):
            complexity_analysis = self.get_complexity_analysis()
            if complexity_analysis:
                with open(results_dir / "complexity_analysis.json", 'w') as f:
                    json.dump(complexity_analysis, f, indent=2)
        
        print(f"\\n Ket qua da duoc luu vao: {results_dir.absolute()}")
        return results_dir