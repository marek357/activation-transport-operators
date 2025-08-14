from sklearn.linear_model import Ridge
import numpy as np
from typing import Dict, Any, Tuple, Optional


def train_ridge_regression(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0
) -> Ridge:
    """
    Train a Ridge regression model.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target values.
        alpha (float): Regularization strength.

    Returns:
        Ridge: Trained Ridge regression model.
    """
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model
