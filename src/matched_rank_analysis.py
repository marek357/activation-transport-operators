"""
Matched-rank curves analysis: Compare CCA ceiling vs rank-r transport operators.

This module implements the core matched-rank analysis that compares:
1) CCA ceiling: use CCA to find the optimal rank-r relationship between X and Y
2) Rank-r transport: fit TransportOperator + SVD truncation to rank r, predict Y_test from X_test
3) Compare R²(r) curves and compute efficiency ratios

The key insight is that R²_T(r) ≤ R²_CCA(r) is expected (CCA gives the theoretical upper bound
for any rank-r linear relationship between X and Y). The gap shows how much of the achievable
variance is captured by the transport operator.

This module provides tools to analyze the relationship between the variance explained by a rank-r
canonical correlation analysis (CCA) reconstruction of target data (the "CCA ceiling") and the
variance explained by a rank-r transport operator mapping from source to target data. The key
mathematical relationship is:

    R²_T(r) ≤ R²_CCA(r)

where:
    - R²_CCA(r): The variance explained (R²) by the best possible rank-r linear relationship between X and Y.
    - R²_T(r): The variance explained (R²) by predicting Y_test from X_test using a transport operator (linear regression) constrained to rank r (via SVD truncation).

**Key Concepts:**
- *CCA ceiling*: The best possible rank-r linear relationship between X and Y, computed using canonical correlation analysis.
- *Rank-r transport operator*: A linear mapping from X to Y, fit on training data, and truncated to rank r.
- *Matched-rank analysis*: For each rank r, compare R²_T(r) to R²_CCA(r). The gap between the two curves quantifies how much of the achievable variance in the X→Y relationship is captured by the transport operator.
- *Efficiency ratio*: The ratio R²_T(r) / R²_CCA(r) for each r, indicating the fraction of the best possible (achievable) variance that is captured by the transport operator.

**Expected Usage Pattern:**
1. Compute R²_CCA(r) for a range of ranks r using canonical correlation analysis between X_train and Y_train.
2. Fit a transport operator (e.g., linear regression) from X_train to Y_train, truncate to rank r, and compute R²_T(r) by predicting Y_test from X_test.
3. For each r, compare R²_T(r) and R²_CCA(r). The efficiency ratio R²_T(r) / R²_CCA(r) summarizes how much of the achievable variance in the X→Y relationship is captured by the transport operator.

**Example:**

```python
from src.matched_rank_analysis import cca_ceiling, fit_transport_rank_r

# Assume x_train, y_train, x_test, y_test are numpy arrays
ranks = [1, 2, 5, 10, 20]

# Compute CCA ceiling
cca_results = cca_ceiling(x_train, y_train, ranks)
# cca_results[r] gives R²_CCA(r)

# Compute transport operator R² for each rank
transport_results = fit_transport_rank_r(
    x_train, y_train, x_train, y_train, x_test, y_test, ranks, alpha_grid=[0.0]
)
# transport_results[r]['r2'] gives R²_T(r)

# Compare R²_T(r) and R²_CCA(r)
for r in ranks:
    print(f"Rank {r}: R²_T(r) = {transport_results[r]['r2']:.3f}, R²_CCA(r) = {cca_results[r]:.3f}, Efficiency = {transport_results[r]['r2']/cca_results[r]:.2%}")
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.activation_loader import ActivationDataset, EfficientActivationDataset
from src.transport_operator import TransportOperator, load_transport_operator
from src.transport_efficiency import _inv_sqrt_psd, r2_ceiling_from_cca

logger = logging.getLogger(__name__)


class RankConstrainedTransportOperator(TransportOperator):
    """
    Transport operator with rank constraint via SVD truncation.

    This extends the base TransportOperator to support rank constraints by
    truncating the learned transport matrix to a specified rank using SVD.
    """

    def __init__(self, L: int, k: int, rank: int, **kwargs):
        """
        Initialize rank-constrained transport operator.

        Args:
            L: Layer number
            k: Offset for target layer
            rank: Rank constraint for transport matrix
            **kwargs: Additional parameters for TransportOperator
        """
        super().__init__(L, k, **kwargs)
        self.rank = rank
        self._full_transport_matrix = None
        self._truncated_transport_matrix = None
        self._truncated_bias = None

    def fit(
        self, dataset: ActivationDataset | EfficientActivationDataset
    ) -> "RankConstrainedTransportOperator":
        """
        Fit the transport operator and apply rank constraint via SVD truncation.

        Args:
            dataset: ActivationDataset | EfficientActivationDataset containing upstream-downstream vector pairs

        Returns:
            self: Fitted rank-constrained transport operator
        """
        # First fit the base transport operator
        super().fit(dataset)

        # Get the full transport matrix
        # Shape: [d_in, d_out]
        self._full_transport_matrix = self.get_transport_matrix()
        full_bias = self.get_bias()  # Shape: [d_out]

        # Apply SVD truncation to enforce rank constraint
        # Transport matrix is stored as coef_.T, so we need to transpose
        T = self._full_transport_matrix.T  # Shape: [d_out, d_in]

        logger.debug(
            f"Applying rank-{self.rank} constraint via SVD to transport matrix {T.shape}"
        )

        # Perform SVD
        U, S, Vt = np.linalg.svd(T, full_matrices=False)

        # Truncate to specified rank
        r_eff = min(self.rank, S.shape[0])
        S_trunc = np.zeros_like(S)
        S_trunc[:r_eff] = S[:r_eff]

        # Reconstruct rank-constrained transport matrix
        T_r = (U * S_trunc) @ Vt  # Shape: [d_out, d_in]

        # Store truncated matrices
        # Store as [d_in, d_out] to match get_transport_matrix
        self._truncated_transport_matrix = T_r.T
        self._truncated_bias = full_bias.copy()

        # Update the internal model's coefficients to use truncated version
        # We need to be careful here since the model might be different types
        if hasattr(self.model, "coef_"):
            self.model.coef_ = T_r  # Shape: [d_out, d_in]

        logger.debug(f"Rank constraint applied: effective rank = {r_eff}")

        return self

    @classmethod
    def from_pretrained(cls, transport_operator, rank):
        instance = cls(
            L=transport_operator.L,
            k=transport_operator.k,
            rank=rank,  # full rank
        )
        instance._full_transport_matrix = transport_operator.get_transport_matrix()
        instance.is_fitted_ = True
        full_bias = transport_operator.get_bias()  # Shape: [d_out]

        # Apply SVD truncation to enforce rank constraint
        # Transport matrix is stored as coef_.T, so we need to transpose
        T = instance._full_transport_matrix.T  # Shape: [d_out, d_in]

        logger.debug(
            f"Applying rank-{instance.rank} constraint via SVD to transport matrix {T.shape}"
        )

        # Perform SVD
        U, S, Vt = np.linalg.svd(T, full_matrices=False)

        # Truncate to specified rank
        r_eff = min(instance.rank, S.shape[0])
        S_trunc = np.zeros_like(S)
        S_trunc[:r_eff] = S[:r_eff]

        # Reconstruct rank-constrained transport matrix
        T_r = (U * S_trunc) @ Vt  # Shape: [d_out, d_in]

        # Store truncated matrices
        # Store as [d_in, d_out] to match get_transport_matrix
        instance._truncated_transport_matrix = T_r.T
        instance._truncated_bias = full_bias.copy()

        # Update the internal model's coefficients to use truncated version
        # We need to be careful here since the model might be different types
        if hasattr(instance.model, "coef_"):
            instance.model.coef_ = T_r  # Shape: [d_out, d_in]

        logger.debug(f"Rank constraint applied: effective rank = {r_eff}")

        return instance

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the rank-constrained transport matrix.

        Args:
            X: Input features [n_samples, n_input_features]

        Returns:
            Predictions [n_samples, n_output_features]
        """
        if not self.is_fitted_:
            raise ValueError("Transport operator must be fitted first")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        # Use the truncated transport matrix directly
        if (
            self._truncated_transport_matrix is not None
            and self._truncated_bias is not None
        ):
            # Apply: Y = X @ T_r.T + b
            # where T_r is shape [d_out, d_in] and T_r.T is [d_in, d_out]
            T_r = self._truncated_transport_matrix.T  # [d_out, d_in]
            return X @ T_r.T + self._truncated_bias
        else:
            # Fallback to parent implementation
            return super().predict(X)

    def get_transport_matrix(self) -> np.ndarray:
        """
        Get the rank-constrained transport matrix.

        Returns:
            Rank-constrained transport matrix, shape (n_features_upstream, n_features_downstream)
        """
        if self._truncated_transport_matrix is not None:
            return self._truncated_transport_matrix
        else:
            return super().get_transport_matrix()

    def get_effective_rank(self) -> int:
        """Get the effective rank after truncation."""
        return min(
            self.rank,
            self._full_transport_matrix.shape[0]
            if self._full_transport_matrix is not None
            else self.rank,
        )


def variance_weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Variance-weighted multi-output R^2 (robust to low-variance dims).

    Args:
        y_true: True target values [n_samples, n_features]
        y_pred: Predicted values [n_samples, n_features]

    Returns:
    Compute a variance-weighted multi-output R² score, robust to low-variance dimensions.

    This metric is designed for multi-output regression tasks where each output dimension
    may have a different variance. Standard R² (as in sklearn) averages R² across dimensions,
    which can cause low-variance dimensions to dominate the mean R², even if they are not
    meaningful. This function instead weights each dimension's R² by its variance in the
    true data, so that high-variance (more informative) outputs contribute more to the
    overall score.

    Mathematical formulation:
        For each output dimension d:
            R²_d = 1 - SSE_d / (SST_d + eps)
            where SSE_d = sum_i (y_true[i, d] - y_pred[i, d])²
                  SST_d = sum_i (y_true[i, d] - mean(y_true[:, d]))²
        The variance weight for dimension d is:
            w_d = SST_d / sum_j SST_j
        The final score is:
            weighted_R² = sum_d w_d * R²_d
        Only dimensions with SST_d > eps are included (to avoid division by zero).

    When to use:
        - Prefer this metric over standard R² when output dimensions have very different
          variances, or when you want the score to reflect performance on high-variance
          (more important) outputs.
        - This is especially useful in neuroscience, genomics, or other domains where
          some outputs are much noisier or less informative than others.
        - Use standard R² if all outputs are equally important and have similar variance.

    Args:
        y_true: True target values [n_samples, n_features]
        y_pred: Predicted values [n_samples, n_features]

    Returns:
        Variance-weighted R² score (float)
    """
    y_true = np.asarray(y_true, dtype=np.float64)  # Use double precision
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_true.shape == y_pred.shape

    # Check for extreme values that could cause overflow
    max_val = max(np.abs(y_true).max(), np.abs(y_pred).max())
    if max_val > 1e6:
        # Scale down to prevent overflow
        scale_factor = 1e6 / max_val
        y_true = y_true * scale_factor
        y_pred = y_pred * scale_factor

    sse = np.sum((y_true - y_pred) ** 2, axis=0)
    mu = np.mean(y_true, axis=0, keepdims=True)

    # More robust variance calculation
    y_centered = y_true - mu
    sst = np.sum(y_centered**2, axis=0)

    # Handle near-zero variance dimensions
    eps = 1e-12
    valid_dims = sst > eps

    if not np.any(valid_dims):
        # All dimensions have zero variance
        return 0.0

    # Only compute R² for dimensions with sufficient variance
    r2_dims = np.zeros_like(sst)
    r2_dims[valid_dims] = 1.0 - sse[valid_dims] / (sst[valid_dims] + eps)

    # Clip extreme R² values
    r2_dims = np.clip(r2_dims, -1e6, 1.0)

    # Weight by variance (only valid dimensions)
    weights = np.zeros_like(sst)
    if np.sum(sst[valid_dims]) > 0:
        weights[valid_dims] = sst[valid_dims] / np.sum(sst[valid_dims])

    # Compute weighted average
    weighted_r2 = np.sum(weights * r2_dims)

    # Handle NaN/inf results
    if not np.isfinite(weighted_r2):
        return 0.0

    return float(weighted_r2)


def calibrate_intercept_whitened(Y_tr, Yhat_tr, muY_tr, W):
    # Compute the u that minimizes ||(Y_tr - (Yhat_tr + 1 c^T)) W||_F^2
    Ew_tr = (Y_tr - Yhat_tr) @ W
    u = Ew_tr.mean(axis=0, keepdims=True)  # [1,d]
    c = u @ np.linalg.inv(W)  # [1,d]  (W is SPD so inv exists)
    return c


def apply_intercept(Yhat, c):
    return Yhat + c  # broadcasts over rows


def whitened_r2(Y, Yhat, muY, W):
    Yw = (Y - muY) @ W
    Yhatw = (Yhat - muY) @ W
    sse = np.linalg.norm(Yw - Yhatw, "fro") ** 2
    sst = np.linalg.norm(Yw, "fro") ** 2
    return 0.0 if sst < 1e-12 else 1.0 - sse / sst


def fit_transport_rank_r(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    L: int,
    k: int,
    ranks: list[int],
) -> dict[int, dict[str, float]]:
    """
    Fit rank-r transport operators using TransportOperator + SVD truncation.

    For each rank r, fit TransportOperator with different regularization strengths,
    then apply SVD truncation to rank r. Choose alpha per r by validation R^2.

    Args:
        x_train: Training inputs [n_train, n_input_features]
        y_train: Training targets [n_train, n_output_features]
        x_val: Validation inputs [n_val, n_input_features]
        y_val: Validation targets [n_val, n_output_features]
        x_test: Test inputs [n_test, n_input_features]
        y_test: Test targets [n_test, n_output_features]
        ranks: List of ranks to evaluate
        alpha_grid: List of regularization strengths to try

    Returns:
        Dictionary mapping rank -> {"R2_val": float, "R2_test": float, "alpha": float, "effective_rank": int}
    """
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    base_transport_op = load_transport_operator(L, k, "./cache")

    results = {}
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    Sigma_yy = (y_train.T @ y_train) / y_train.shape[0]
    cov_yy_inv_sqrt = _inv_sqrt_psd(Sigma_yy)

    for r in tqdm(ranks):
        best_result = {
            "R2_val": -1e9,
            "R2_test": None,
            "alpha": None,
            "effective_rank": 0,
        }

        try:
            transport_op = RankConstrainedTransportOperator.from_pretrained(
                base_transport_op, rank=r
            )

            # Validate on validation set
            y_test_hat = transport_op.predict(x_test)
            muY_tr = y_train.mean(axis=0, keepdims=True)
            Syy_tr = np.cov((y_train - muY_tr).T, bias=True)
            W = _inv_sqrt_psd(Syy_tr)

            y_train_hat = transport_op.predict(x_train)
            # Calibrate intercept in the whitened metric on TRAIN ONLY
            c = calibrate_intercept_whitened(y_train, y_train_hat, muY_tr, W)

            # Adjust predictions everywhere
            # Yhat_tr_adj = apply_intercept(y_train_hat, c)
            Yhat_te_adj = apply_intercept(y_test_hat, c)

            # Evaluate whitened R^2 (aligned with the CCA ceiling)
            # r2w_train = whitened_r2(y_train, Yhat_tr_adj, muY_tr, W)
            r2_val = whitened_r2(y_test, Yhat_te_adj, muY_tr, W)

            # r2_val = whitened_r2(y_test, y_test_hat, y_mean, cov_yy_inv_sqrt)

            if r2_val > best_result["R2_val"]:
                # Evaluate on test set
                y_test_hat = transport_op.predict(x_test)
                r2_test = whitened_r2(y_test, Yhat_te_adj, y_mean, cov_yy_inv_sqrt)

                best_result = {
                    "R2_val": float(r2_val),
                    "R2_test": float(r2_test),
                    "effective_rank": transport_op.get_effective_rank(),
                    # WARNING: Remove in the future
                    # TODO: This is probably a very dummy decision we made at 4 am when debugging this code...
                    "alpha": 1500.0,
                }

        except Exception as e:
            logger.exception(f"Failed to fit transport rank {r}: {e}", stack_info=True)
            raise Exception from e

        results[int(r)] = best_result

        # Safe logging with None checks
        r2_test = best_result["R2_test"]
        effective_rank = best_result["effective_rank"]

        alpha = best_result["alpha"]

        r2_str = f"{r2_test:.4f}" if r2_test is not None else "None"
        alpha_str = f"{alpha}" if alpha is not None else "None"

        logger.debug(
            f"Transport rank {r}: R2_test={r2_str}, "
            f"alpha={alpha_str}, effective_rank={effective_rank}"
        )

    return results


class _NumpyActivationDataset(ActivationDataset):
    """Simple dataset wrapper for numpy arrays to work with TransportOperator."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.idx_list = list(range(len(x)))

        # Set required attributes to make it compatible with ActivationDataset
        self.dataset_id = "numpy_dataset"
        self.j_policy = "j==i"  # Required by the parent class iterator
        self.L = 0
        self.k = 1
        self.activation_loader = None  # Not used in our simple case

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __iter__(self):
        """Simple iterator that yields (x, y) pairs directly."""
        for i in range(len(self.x)):
            yield self.x[i], self.y[i]


def _create_numpy_dataset(x: np.ndarray, y: np.ndarray) -> _NumpyActivationDataset:
    """Create a dataset wrapper for numpy arrays."""
    return _NumpyActivationDataset(x, y)


def compare_cca_vs_transport(
    L: int,
    k: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ranks: list[int],
) -> dict[str, Any]:
    """
    Main function to compare CCA ceiling vs rank-r transport operators.
    Returns:
        Dictionary with all results and metrics
    """

    # ranks = list(range(1, 2300, 50))
    # ranks.append(2304)  # the full rank is 2304

    ranks = list(sorted(set(int(r) for r in ranks)))

    logger.info(f"Starting matched-rank analysis for ranks: {ranks}")
    logger.info(f"Data shapes: X_test={x_test.shape}, Y_test={y_test.shape}, ")
    logger.info(f"Data shapes: X_train={x_train.shape}, Y_train={y_train.shape}, ")

    # 1) CCA ceiling
    logger.info("Computing CCA ceiling...")
    cca_dict, rhos = r2_ceiling_from_cca(x_train, y_train, ranks)
    dim_lin = sum(rhos) ** 2 / (sum(rho**2 for rho in rhos) + 1e-12)

    # 2) Transport rank-r
    logger.info("Computing rank-r transport operators...")
    trans_dict = fit_transport_rank_r(x_train, y_train, x_test, y_test, L, k, ranks)

    # 3) Efficiency (Transport R2 / CCA R2)
    efficiency = {}
    for r in ranks:
        cca_r2 = cca_dict[int(r)]
        trans_r2 = trans_dict[r]["R2_test"]
        if cca_r2 > 1e-8 and trans_r2 is not None:
            efficiency[int(r)] = float(trans_r2 / cca_r2)
        else:
            efficiency[int(r)] = float("nan")

    print(efficiency)

    # Generate plots if requested
    try:
        plt.figure(figsize=(12, 5))

        # Main comparison plot
        plt.subplot(1, 2, 1)
        cca_r2_values = [cca_dict[r] for r in ranks]
        trans_r2_values = [
            trans_dict[r]["R2_test"] if trans_dict[r]["R2_test"] is not None else 0
            for r in ranks
        ]

        plt.plot(ranks, cca_r2_values, marker="o", label="CCA ceiling", linewidth=2)
        plt.plot(
            ranks,
            trans_r2_values,
            marker="s",
            label="Transport (rank-r)",
            linewidth=2,
        )
        plt.xlabel("Rank r")
        plt.ylabel("Variance-weighted $R^2$ (test)")
        plt.title("CCA vs Transport ($X \\to Y$)")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        # Efficiency plot
        plt.subplot(1, 2, 2)
        eff_values = [efficiency[r] for r in ranks]
        plt.plot(ranks, eff_values, marker="o", linewidth=2)
        plt.xlabel("Rank r")
        plt.ylabel("Efficiency = $R^2_T(r) / R^2_{CCA}(r)$")
        plt.title("Transport Efficiency vs Rank")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.ylim(0, 1.1)

        plt.tight_layout()
        # plt.show()

    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    # Compile results
    results = {
        "ranks": ranks,
        "dim_lin": dim_lin,
        "cca_R2": {int(r): float(cca_dict[int(r)]) for r in ranks},
        "transport": trans_dict,  # contains R2_val, R2_test, alpha per rank
        "efficiency": {int(r): float(efficiency[int(r)]) for r in ranks},
        "summary_stats": {
            "max_cca_r2": max(cca_dict[r] for r in ranks),
            "max_transport_r2": max(
                trans_dict[r]["R2_test"]
                for r in ranks
                if trans_dict[r]["R2_test"] is not None
            ),
            "mean_efficiency": np.nanmean([efficiency[r] for r in ranks]),
            "best_rank_cca": max(ranks, key=lambda r: cca_dict[r]),
            "best_rank_transport": max(
                ranks,
                key=lambda r: trans_dict[r]["R2_test"]
                if trans_dict[r]["R2_test"] is not None
                else -1,
            ),
        },
    }

    logger.info("Matched-rank analysis completed successfully")
    logger.info("Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    return results


def run_matched_CCA_rank_analysis_from_datasets(
    train_dataset: ActivationDataset | EfficientActivationDataset,
    val_dataset: ActivationDataset | EfficientActivationDataset,
    test_dataset: ActivationDataset | EfficientActivationDataset,
    ranks: list[int] | None = None,
    L: int = 0,
    k: int = 1,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Run matched-rank analysis using ActivationDataset | EfficientActivationDataset objects with CCA ceiling.

    Args:
        train_dataset, val_dataset, test_dataset: ActivationDataset | EfficientActivationDataset objects
        ranks: List of ranks to evaluate
        L: Layer number
        k: Offset for target layer
        max_samples: Maximum number of samples to use (for computational efficiency)

    Returns:
        Dictionary with all results and metrics
    """

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

    logger.info("Converting datasets to numpy arrays...")

    # Convert datasets to numpy arrays
    x_train, y_train = dataset_to_arrays(train_dataset, max_samples)
    x_val, y_val = dataset_to_arrays(val_dataset, max_samples)
    x_test, y_test = dataset_to_arrays(test_dataset, max_samples)

    logger.info("Dataset conversion complete. Final shapes:")
    logger.info(f"  Train: X={x_train.shape}, Y={y_train.shape}")
    logger.info(f"  Val: X={x_val.shape}, Y={y_val.shape}")
    logger.info(f"  Test: X={x_test.shape}, Y={y_test.shape}")

    # Run the CCA-based analysis
    return compare_cca_vs_transport(
        L,
        k,
        x_train,
        y_train,
        x_test,
        y_test,
        ranks,
    )


def run_matched_rank_analysis_from_datasets(
    L: int,
    k: int,
    train_dataset: ActivationDataset | EfficientActivationDataset,
    test_dataset: ActivationDataset | EfficientActivationDataset,
    ranks: list[int],
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Run matched-rank analysis using ActivationDataset | EfficientActivationDataset objects.

    Args:
        train_dataset, val_dataset, test_dataset: ActivationDataset | EfficientActivationDataset objects
        ranks: List of ranks to evaluate
        alpha_grid: Ridge regularization strengths to try
        orthogonal_test_ranks: Ranks for orthogonal complement analysis
        plot: Whether to generate plots
        max_samples: Maximum number of samples to use (for computational efficiency)

    Returns:
        Dictionary with all results and metrics
    """

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

    logger.info("Converting datasets to numpy arrays...")

    # Convert datasets to numpy arrays
    x_test, y_test = dataset_to_arrays(test_dataset, max_samples)
    x_train, y_train = dataset_to_arrays(train_dataset, max_samples)

    logger.info("Dataset conversion complete. Final shapes:")
    logger.info(f"  Train: X={x_train.shape}, Y={y_train.shape}")
    logger.info(f"  Test: X={x_test.shape}, Y={y_test.shape}")

    # Run the main analysis
    return compare_cca_vs_transport(L, k, x_train, y_train, x_test, y_test, ranks)
