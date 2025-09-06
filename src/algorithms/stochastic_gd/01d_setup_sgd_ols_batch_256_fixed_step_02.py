import sys
import os
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.stochastic_gd.sgd_model import SGDModel
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
    
    # Fixed Step Length: batch_size=256, step_length=0.02
    model = SGDModel(
        learning_rate=0.01,  # Ignored when using fixed step length
        so_epochs=100,
        random_state=42,
        batch_size=256,
        ham_loss='ols',
        use_fixed_step_length=True,
        step_length=0.02
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
    
    print(f"\nFixed Step Length - Batch 256, Step Length 0.02 completed!")
    print(f"Final Loss: {results['final_loss']:.6f}")
    print(f"Test MSE: {metrics['mse']:.6f}")
    
    return model, results, metrics


if __name__ == "__main__":
    main()