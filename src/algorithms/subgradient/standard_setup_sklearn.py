import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.linear_model import Lasso

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.data_process_utils import load_du_lieu


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_du_lieu()

    # Fit Lasso regression

    lasso = Lasso(alpha=0.1, max_iter=750)
    lasso.fit(X_train, y_train)

    # Assess the model
    score = lasso.score(X_test, y_test)
    print(f"Test R^2 score: {score:.4f}")
    print(f"Coefficients: {lasso.coef_}")
    print(f"Intercept: {lasso.intercept_}")
    d = 1


if __name__ == "__main__":
    main()
