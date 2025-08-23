from src.activation_loader import (
    ActivationDataset,
    ActivationLoader,
    EfficientActivationDataset,
    get_train_val_test_datasets,
)
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_r2_ceiling(X, y):
    """Compute the R² ceiling for predicting y from X using linear regression."""
    sigma_x_x = (X.T @ X) / X.shape[0]
    sigma_x_y = (X.T @ y) / X.shape[0]
    sigma_y = (y.T @ y) / y.shape[0]
    r2_ceiling = (sigma_x_y.T @ np.linalg.inv(sigma_x_x) @ sigma_x_y) / sigma_y
    return r2_ceiling.item()


def _inv_sqrt_psd(S, eps=1e-12, ridge=0.0):
    """Real inverse square-root for (near-)PSD S via eigh."""
    S = 0.5 * (S + S.T)  # enforce symmetry
    if ridge > 0:
        S = S + ridge * np.eye(S.shape[0], dtype=S.dtype)
    w, V = np.linalg.eigh(S)  # real eigenpairs
    w = np.clip(w, eps, None)  # kill tiny/neg eigenvals
    return (V * (w**-0.5)) @ V.T  # V @ diag(w^-1/2) @ V^T


def r2_ceiling_from_cca(X, Y, rs, *, center=True, ridge=0.0, eps=1e-12):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N, dx = X.shape
    Ny, dy = Y.shape
    assert N == Ny, "X and Y must have the same number of rows (samples)."
    assert dx == dy, "This implementation assumes d_model matches for X and Y."

    if center:
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

    # Covariances with 1/N factor
    Sigma_xx = (X.T @ X) / N
    Sigma_yy = (Y.T @ Y) / N
    Sigma_yx = (Y.T @ X) / N

    # Inverse square roots (real, stable)
    Sxx_mhalf = _inv_sqrt_psd(Sigma_xx, eps=eps, ridge=ridge)
    Syy_mhalf = _inv_sqrt_psd(Sigma_yy, eps=eps, ridge=ridge)

    C = Syy_mhalf @ Sigma_yx @ Sxx_mhalf

    _, rho, _ = np.linalg.svd(C, full_matrices=False)  # rho sorted desc

    ceilings = {}
    for r in rs:
        # Clip r to available modes
        k = min(r, len(rho))
        R2_ceiling_r = float(np.sum(rho[:k] ** 2) / dx)
        ceilings[r] = R2_ceiling_r

    return ceilings, rho


def dataset_to_arrays(
    dataset: ActivationDataset | EfficientActivationDataset,
    max_samples: int | None = None,
):
    """Convert ActivationDataset | EfficientActivationDataset to numpy arrays."""
    logger.info(
        f"Converting ActivationDataset | EfficientActivationDataset to arrays (max_samples={max_samples})..."
    )

    x_list = []
    y_list = []

    # Since ActivationDataset | EfficientActivationDataset is an IterableDataset, we need to iterate through it
    samples_collected = 0

    for x, y in dataset:
        x_list.append(x.numpy() if hasattr(x, "numpy") else x.cpu().numpy())
        y_list.append(y.numpy() if hasattr(y, "numpy") else y.cpu().numpy())
        samples_collected += 1

        # Stop if we've collected enough samples
        if max_samples is not None and samples_collected >= max_samples:
            break

    logger.info(f"Collected {samples_collected} samples from dataset")

    if len(x_list) == 0:
        raise ValueError("No samples collected from dataset")

    return np.array(x_list), np.array(y_list)


if __name__ == "__main__":
    activation_loader = ActivationLoader(
        files_to_download=[
            "activations-gemma2-2b-slimpajama-250k/activations_part_0000.zarr.zip"
        ]
    )
    # activation_loader = ActivationLoader("./activations-gemma2-2b-slimpajama-250k")
    # load X and Y from src.activation_loader
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        5, 5, activation_loader, "j==i"
    )
    x_train, y_train = dataset_to_arrays(train_dataset, 10000)
    x_val, y_val = dataset_to_arrays(val_dataset, 10000)
    x_test, y_test = dataset_to_arrays(test_dataset, 10000)

    print(r2_ceiling_from_cca(x_test, y_test, range(1, 3101, 100)))
