import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.data_process_utils import load_du_lieu
from algorithms.subgradient import (
    SubgradientNonSummableDiminishingStepLength,
    SubgradientNonSummableDiminishingStepSize,
    SubgradientConstantStepLength,
    SubgradientConstantStepSize,
    SubgradientSquareSummable,
)


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()

    model = SubgradientConstantStepLength()

    # Add static value to X_test
    X_train_updated = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_updated = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    # Huấn luyện model
    results = model.fit(X=X_train_updated, y=y_train)

    # Đánh giá model
    metrics = model.evaluate(X_test_updated, y_test)

    # Lưu kết quả với tên file tự động
    ten_file = "toan_test"
    results_dir = model.save_results(ten_file)

    # # Tạo biểu đồ
    # model.plot_results(X_test_updated, y_test, ten_file)

    print(f"\\nTraining and visualization completed!")
    print(f"Results saved to: {results_dir.absolute()}")

    return model, results, metrics


if __name__ == "__main__":
    main()
