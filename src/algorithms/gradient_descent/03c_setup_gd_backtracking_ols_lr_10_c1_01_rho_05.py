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
        learning_rate=1.0,  # Base learning rate cho backtracking
        diem_dung=1e-5,
        step_size_method='backtracking',
        backtrack_c1=1e-1,  # Armijo constant (loose condition)
        backtrack_rho=0.5   # Reduction factor (gentle backtrack)
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()  
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nBacktracking Line Search (c1=0.01, rho=0.8) training completed!")
    print(f"Final step size: {results['step_sizes_history'][-1]:.6f}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()