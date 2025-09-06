import sys
import os
from pathlib import Path
import numpy as np

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.gradient_descent.gradient_descent_model import GradientDescentModel
from utils.data_process_utils import load_du_lieu


class ExponentialScheduleGD(GradientDescentModel):
    """Gradient Descent with Exponential Schedule: lr(t) = lr_0 * gamma^(t/step_size)"""
    
    def __init__(self, decay_step=1000, **kwargs):
        super().__init__(**kwargs)
        self.decay_step = decay_step  # Apply decay every N iterations
        
    def _get_step_size(self, iteration, gradient, X=None, y=None, weights=None):
        """Override to implement exponential schedule with step"""
        if self.step_size_method == 'exponential_schedule':
            # lr(t) = lr_0 * gamma^(t/decay_step)
            # This decays more gradually than per-iteration
            decay_factor = iteration // self.decay_step  # Integer division
            return self.learning_rate * (self.decay_gamma ** decay_factor)
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
    
    model = ExponentialScheduleGD(
        ham_loss='ols',
        learning_rate=0.1,  # Initial lr
        diem_dung=1e-5,
        step_size_method='exponential_schedule',
        decay_gamma=0.95,   # 5% reduction every decay_step
        decay_step=1000     # Decay every 2000 iterations
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
    
    print(f"\nExponential Schedule GD completed!")
    print(f"Initial learning rate: {results['step_sizes_history'][0]:.6f}")
    print(f"Final learning rate: {results['step_sizes_history'][-1]:.6f}")
    
    # Show decay progression
    print("Decay progression:")
    print(f"  0-1999 iterations: lr = {0.1:.4f}")
    print(f"  2000-3999 iterations: lr = {0.1 * 0.95:.4f}")
    print(f"  4000-5999 iterations: lr = {0.1 * 0.95**2:.4f}")
    print(f"  6000+ iterations: lr = {0.1 * 0.95**3:.4f}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()