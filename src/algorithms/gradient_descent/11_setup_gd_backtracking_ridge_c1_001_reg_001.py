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
        ham_loss='ridge',
        learning_rate=0.1,  # Base learning rate cho backtracking
        so_lan_thu=10000,
        diem_dung=1e-5,
        regularization=0.01,  # Ridge regularization parameter
        step_size_method='backtracking',
        backtrack_c1=1e-3,  # Armijo constant (less strict than 1e-4)
        backtrack_rho=0.8   # Reduction factor
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()  # Sẽ là "setup_gd_backtracking_ridge_c1_001_reg_001"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nBacktracking Line Search + Ridge training completed!")
    print(f"Final step size: {results['step_sizes_history'][-1]:.6f}")
    print(f"Regularization: {model.regularization}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()