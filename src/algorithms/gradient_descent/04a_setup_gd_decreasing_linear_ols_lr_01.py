import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.gradient_descent.gradient_descent_model import GradientDescentModel
from utils.data_process_utils import load_du_lieu


def get_experiment_name():
    """Lấy tên experiment từ tên file hiện tại"""
    import inspect
    frame = inspect.currentframe()
    filename = frame.f_back.f_globals['__file__']
    return Path(filename).stem  # Lấy tên file không có extension

def main():
   
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()
    
    model = GradientDescentModel(
        ham_loss='ols',
        learning_rate=0.1,  # Base learning rate α
        diem_dung=1e-5,
        step_size_method='decreasing_linear'  # α / (iteration + 1)
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()  # Sẽ là "setup_gd_decreasing_linear_ols_lr_01"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nDecreasing Linear Step Size training completed!")
    print(f"Initial step size: {results['step_sizes_history'][0]:.6f}")
    print(f"Final step size: {results['step_sizes_history'][-1]:.6f}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()