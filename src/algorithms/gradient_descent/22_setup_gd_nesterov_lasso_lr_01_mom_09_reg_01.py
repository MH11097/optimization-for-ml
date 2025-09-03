import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.gradient_descent.nesterov_gd_model import NesterovGDModel
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
    
    model = NesterovGDModel(
        ham_loss='lasso',    # L1 regularization
        learning_rate=0.1,   # Learning rate
        so_lan_thu=1000,
        diem_dung=1e-5,
        momentum=0.9,        # High momentum for acceleration
        regularization=0.1   # L1 regularization parameter
    )
    
    # Huấn luyện model
    results = model.fit(X_train, y_train)
    
    # Đánh giá model
    metrics = model.evaluate(X_test, y_test)
    
    # Lưu kết quả với tên file tự động
    ten_file = get_experiment_name()  # Sẽ là "setup_gd_nesterov_lasso_lr_01_mom_09_reg_01"
    results_dir = model.save_results(ten_file)
    
    # Tạo biểu đồ
    model.plot_results(X_test, y_test, ten_file)
    
    print(f"\nNesterov GD + Lasso training completed!")
    print(f"Momentum parameter: {model.momentum}")
    print(f"L1 regularization: {model.regularization}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()