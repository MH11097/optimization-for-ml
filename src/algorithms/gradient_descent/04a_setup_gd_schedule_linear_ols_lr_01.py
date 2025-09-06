import sys
import os
from pathlib import Path
import numpy as np

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.gradient_descent.gradient_descent_model import GradientDescentModel
from utils.data_process_utils import load_du_lieu


class LinearScheduleGD(GradientDescentModel):
    """Gradient Descent with Linear Schedule: lr(t) = lr_0 * (1 - t/T)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = self.so_lan_thu
        
    def _get_step_size(self, iteration, gradient, X=None, y=None, weights=None):
        """Override to implement linear schedule"""
        if self.step_size_method == 'linear_schedule':
            # lr(t) = lr_0 * (1 - t/T) where T = max_iterations
            progress = min(iteration / self.max_iterations, 0.99)  # Cap at 99% to avoid lr=0
            return self.learning_rate * (1 - progress)
        else:
            return super()._get_step_size(iteration, gradient, X, y, weights)


def get_experiment_name():
    """Lấy tên experiment từ tên file hiện tại"""
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    model = LinearScheduleGD(
        ham_loss='ols',
        learning_rate=0.1,
        diem_dung=1e-5,
        step_size_method='linear_schedule'
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả
    ten_file = get_experiment_name()
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nLinear Schedule GD completed!")
    print(f"Initial learning rate: {results['step_sizes_history'][0]:.6f}")
    print(f"Final learning rate: {results['step_sizes_history'][-1]:.6f}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()