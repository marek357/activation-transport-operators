"""
Matched-rank curves analysis: Compare PCA ceiling vs rank-r transport operators.

This module implements the core matched-rank analysis that compares:
1) PCA ceiling: learn PCs on Y_train, reconstruct Y_test with rank-r approximation
2) Rank-r transport: fit TransportOperator + SVD truncation to rank r, predict Y_test from X_test
3) Compare R²(r) curves and compute efficiency ratios

The key insight is that R²_T(r) ≤ R²_PCA(r) is expected (PCA is the best possible
rank-r Y-only reconstruction). The gap shows how much of compressible variance
This module provides tools to analyze the relationship between the variance explained by a rank-r principal component analysis (PCA) reconstruction of target data (the "PCA ceiling") and the variance explained by a rank-r transport operator mapping from source to target data. The key mathematical relationship is:

    R²_T(r) ≤ R²_PCA(r)

where:
    - R²_PCA(r): The variance explained (R²) by reconstructing Y_test using the top r principal components learned from Y_train.
    - R²_T(r): The variance explained (R²) by predicting Y_test from X_test using a transport operator (linear regression) constrained to rank r (via SVD truncation).

**Key Concepts:**
- *PCA ceiling*: The best possible rank-r approximation of Y_test, using only information from Y_train.
- *Rank-r transport operator*: A linear mapping from X to Y, fit on training data, and truncated to rank r.
- *Matched-rank analysis*: For each rank r, compare R²_T(r) to R²_PCA(r). The gap between the two curves quantifies how much of the compressible variance in Y is actually predictable from X.
- *Efficiency ratio*: The ratio R²_T(r) / R²_PCA(r) for each r, indicating the fraction of the best possible (compressible) variance that is predictable from X.

**Expected Usage Pattern:**
1. Fit PCA on Y_train, and compute R²_PCA(r) for a range of ranks r by reconstructing Y_test.
2. Fit a transport operator (e.g., linear regression) from X_train to Y_train, truncate to rank r, and compute R²_T(r) by predicting Y_test from X_test.
3. For each r, compare R²_T(r) and R²_PCA(r). The efficiency ratio R²_T(r) / R²_PCA(r) summarizes how much of the compressible variance in Y is predictable from X.

**Example:**

```python
from src.matched_rank_analysis import pca_ceiling, fit_transport_rank_r

# Assume x_train, y_train, x_test, y_test are numpy arrays
ranks = [1, 2, 5, 10, 20]

# Compute PCA ceiling
pca_results = pca_ceiling(y_train, y_test, ranks)
# pca_results[r]['r2'] gives R²_PCA(r)

# Compute transport operator R² for each rank
transport_results = fit_transport_rank_r(
    x_train, y_train, x_train, y_train, x_test, y_test, ranks, alpha_grid=[0.0]
)
# transport_results[r]['r2'] gives R²_T(r)

# Compare R²_T(r) and R²_PCA(r)
for r in ranks:
    print(f"Rank {r}: R²_T(r) = {transport_results[r]['r2']:.3f}, R²_PCA(r) = {pca_results[r]['r2']:.3f}, Efficiency = {transport_results[r]['r2']/pca_results[r]['r2']:.2%}")
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
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


# def whitened_r2(y_gt, y_pred, y_mean, cov_yy_inv_sqrt):
#     # Center the ground truth and predicted values
#     y_gt_centered = y_gt - y_mean
#     y_pred_centered = y_pred - y_mean

#     # Whiten the centered values
#     y_gt_whitened = y_gt_centered @ cov_yy_inv_sqrt
#     y_pred_whitened = y_pred_centered @ cov_yy_inv_sqrt

#     # Compute R² score on the whitened values
#     return (
#         1
#         - np.linalg.norm(y_gt_whitened - y_pred_whitened, ord="fro") ** 2
#         / np.linalg.norm(y_gt_whitened, ord="fro") ** 2
#     )


def inv_sqrt_psd(S, ridge=1e-6, eps=1e-12):
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S + ridge * np.eye(S.shape[0]))
    w = np.clip(w, eps, None)
    return (V * (w**-0.5)) @ V.T


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


def pca_ceiling(
    y_train: np.ndarray, y_test: np.ndarray, ranks: list[int]
) -> dict[int, dict[str, float]]:
    """
    Compute PCA ceiling: fit PCA on Y_train, reconstruct Y_test for each rank.

    Args:
        y_train: Training targets [n_train, n_features]
        y_test: Test targets [n_test, n_features]
        ranks: List of ranks to evaluate

    Returns:
        Dictionary mapping rank -> {"R2": float, "explained_variance_ratio": float}
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    max_r = int(max(ranks))
    max_components = min(max_r, y_train.shape[1], y_train.shape[0] - 1)

    pca = PCA(n_components=max_components, svd_solver="full", whiten=False)
    pca.fit(y_train)  # centers internally

    mu_tr = np.mean(y_train, axis=0, keepdims=True)
    z_te_full = (y_test - mu_tr) @ pca.components_.T

    results = {}
    for r in ranks:
        r = int(r)
        r_eff = min(r, max_components)

        comps_r = pca.components_[:r_eff, :]
        z_te_r = z_te_full[:, :r_eff]
        y_hat = z_te_r @ comps_r + mu_tr

        r2_score = variance_weighted_r2(y_test, y_hat)
        explained_var_ratio = np.sum(pca.explained_variance_ratio_[:r_eff])

        results[r] = {
            "R2": r2_score,
            "explained_variance_ratio": float(explained_var_ratio),
            "effective_rank": r_eff,
        }

        logger.debug(
            f"PCA ceiling rank {r}: R2={r2_score:.4f}, explained_var={explained_var_ratio:.4f}"
        )

    return results


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

    # Create temporary datasets for TransportOperator
    test_dataset = _create_numpy_dataset(x_test, y_test)
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
            # Create rank-constrained transport operator
            # transport_op = RankConstrainedTransportOperator(
            #     L=0,  # Dummy values since we're using it standalone
            #     k=1,
            #     rank=int(r),
            #     method="ridge",
            #     regularization=float(alpha),
            #     auto_tune=False,  # We're manually tuning
            #     use_cache=False,  # Disable caching for this analysis
            #     normalize=False,  # Keep consistent with original implementation
            # )

            # Fit the transport operator
            # transport_op.fit(train_dataset)
            # base_transport_op = transport_op
            transport_op = RankConstrainedTransportOperator.from_pretrained(
                base_transport_op, rank=r
            )

            # Validate on validation set
            y_test_hat = transport_op.predict(x_test)
            muY_tr = y_train.mean(axis=0, keepdims=True)
            Syy_tr = np.cov((y_train - muY_tr).T, bias=True)
            W = inv_sqrt_psd(Syy_tr)

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
                r2_test = whitened_r2(y_test, y_test_hat, y_mean, cov_yy_inv_sqrt)

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


def orthogonal_complement_r2(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_dict: dict[int, np.ndarray],
    ranks_for_pca: list[int],
) -> dict[int, dict[int, float]]:
    """
    Compute R^2 in orthogonal complement of top-r0 PCA components.

    For each r0, compute R^2 in the orthogonal complement of top-r0 PCA(Y_train).
    Y_pred_dict: {rank_r: Y_pred_on_test}.

    Args:
        y_train: Training targets for fitting PCA
        y_test: Test targets
        y_pred_dict: Dictionary mapping rank -> predicted targets on test set
        ranks_for_pca: List of PCA ranks to use for orthogonal complement

    Returns:
        Dictionary mapping pca_rank -> {transport_rank: R2_orthogonal}
    """
    if not ranks_for_pca:
        return {}

    max_r0 = max(ranks_for_pca)
    max_components = min(max_r0, y_train.shape[1], y_train.shape[0] - 1)

    pca = PCA(n_components=max_components, svd_solver="full", whiten=False)
    pca.fit(y_train)
    mu_tr = np.mean(y_train, axis=0, keepdims=True)

    results = {}
    for r0 in ranks_for_pca:
        r0_eff = min(int(r0), max_components)
        comps = pca.components_[:r0_eff, :]
        P = comps.T @ comps  # Projection matrix onto top-r0 subspace
        I = np.eye(P.shape[0], dtype=y_test.dtype)
        Proj_perp = I - P  # Projection onto orthogonal complement

        # Project test targets onto orthogonal complement
        y_test_perp = ((y_test - mu_tr) @ Proj_perp) + mu_tr

        results[int(r0)] = {}
        for r_pred, y_hat in y_pred_dict.items():
            # Project predictions onto orthogonal complement
            y_hat_perp = ((y_hat - mu_tr) @ Proj_perp) + mu_tr
            r2_perp = variance_weighted_r2(y_test_perp, y_hat_perp)
            results[int(r0)][int(r_pred)] = float(r2_perp)

    return results


def compare_pca_vs_transport(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ranks: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    orthogonal_test_ranks: list[int] | None = None,
    plot: bool = True,
) -> dict[str, Any]:
    """
    Main function to compare PCA ceiling vs rank-r transport operators.

    Args:
        x_train, y_train: Training data
        x_val, y_val: Validation data
        x_test, y_test: Test data
        ranks: List of ranks to evaluate
        alpha_grid: Ridge regularization strengths to try
        orthogonal_test_ranks: Ranks for orthogonal complement analysis
        plot: Whether to generate plots

    Returns:
        Dictionary with all results and metrics
    """
    if ranks is None:
        ranks = [8, 16, 32, 64, 128, 256]
    if alpha_grid is None:
        alpha_grid = [0.1, 1.0, 10.0, 100.0]
    if orthogonal_test_ranks is None:
        orthogonal_test_ranks = []

    ranks = list(sorted(set(int(r) for r in ranks)))

    logger.info(f"Starting matched-rank analysis for ranks: {ranks}")
    logger.info(
        f"Data shapes: X_train={x_train.shape}, Y_train={y_train.shape}, "
        f"X_test={x_test.shape}, Y_test={y_test.shape}"
    )

    # 1) PCA ceiling
    logger.info("Computing PCA ceiling...")
    pca_dict = pca_ceiling(y_train, y_test, ranks)

    # 2) Transport rank-r
    logger.info("Computing rank-r transport operators...")
    trans_dict = fit_transport_rank_r(
        x_train, y_train, x_val, y_val, x_test, y_test, ranks, alpha_grid
    )

    # 3) Efficiency (Transport R2 / PCA R2)
    efficiency = {}
    for r in ranks:
        pca_r2 = pca_dict[int(r)]["R2"]
        trans_r2 = trans_dict[int(r)]["R2_test"]
        if pca_r2 > 1e-8 and trans_r2 is not None:
            efficiency[int(r)] = float(trans_r2 / pca_r2)
        else:
            efficiency[int(r)] = float("nan")

    # 4) Optional orthogonal complement analysis
    ortho_results = {}
    if orthogonal_test_ranks and False:
        logger.info("Computing orthogonal complement analysis...")
        # Recompute test predictions for orthogonal analysis
        preds = {}
        for r in ranks:
            alpha = trans_dict[int(r)]["alpha"]
            if alpha is not None:
                # Create and fit transport operator with the best alpha
                transport_op = RankConstrainedTransportOperator(
                    L=0,
                    k=1,
                    rank=int(r),
                    method="ridge",
                    regularization=float(alpha),
                    auto_tune=False,
                    use_cache=False,
                    normalize=False,
                )

                train_dataset = _create_numpy_dataset(x_train, y_train)
                transport_op.fit(train_dataset)
                preds[int(r)] = transport_op.predict(x_test)

        ortho_results = orthogonal_complement_r2(
            y_train, y_test, preds, orthogonal_test_ranks
        )

    # 5) Generate plots if requested
    if plot:
        try:
            plt.figure(figsize=(12, 5))

            # Main comparison plot
            plt.subplot(1, 2, 1)
            pca_r2_values = [pca_dict[r]["R2"] for r in ranks]
            trans_r2_values = [
                trans_dict[r]["R2_test"] if trans_dict[r]["R2_test"] is not None else 0
                for r in ranks
            ]

            plt.plot(ranks, pca_r2_values, marker="o", label="PCA ceiling", linewidth=2)
            plt.plot(
                ranks,
                trans_r2_values,
                marker="s",
                label="Transport (rank-r)",
                linewidth=2,
            )
            plt.xlabel("Rank r")
            plt.ylabel("Variance-weighted $R^2$ (test)")
            plt.title("PCA vs Transport ($X \\to Y$)")
            plt.legend()
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

            # Efficiency plot
            plt.subplot(1, 2, 2)
            eff_values = [efficiency[r] for r in ranks]
            plt.plot(ranks, eff_values, marker="o", linewidth=2)
            plt.xlabel("Rank r")
            plt.ylabel("Efficiency = $R^2_T(r) / R^2_{PCA}(r)$")
            plt.title("Transport Efficiency vs Rank")
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            plt.ylim(0, 1.1)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

    # Compile results
    results = {
        "ranks": ranks,
        "pca_R2": {int(r): float(pca_dict[int(r)]["R2"]) for r in ranks},
        "pca_explained_variance": {
            int(r): float(pca_dict[int(r)]["explained_variance_ratio"]) for r in ranks
        },
        "transport": trans_dict,  # contains R2_val, R2_test, alpha per rank
        "efficiency": {int(r): float(efficiency[int(r)]) for r in ranks},
        "orthogonal_complement_R2": ortho_results,
        "summary_stats": {
            "max_pca_r2": max(pca_dict[r]["R2"] for r in ranks),
            "max_transport_r2": max(
                trans_dict[r]["R2_test"]
                for r in ranks
                if trans_dict[r]["R2_test"] is not None
            ),
            "mean_efficiency": np.nanmean([efficiency[r] for r in ranks]),
            "best_rank_pca": max(ranks, key=lambda r: pca_dict[r]["R2"]),
            "best_rank_transport": max(
                ranks,
                key=lambda r: trans_dict[r]["R2_test"]
                if trans_dict[r]["R2_test"] is not None
                else -1,
            ),
        },
    }

    logger.info("Matched-rank analysis completed successfully")
    return results


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
    cca_dict = r2_ceiling_from_cca(x_train, y_train, ranks)

    # 2) Transport rank-r
    logger.info("Computing rank-r transport operators...")
    trans_dict = fit_transport_rank_r(x_train, y_train, x_test, y_test, L, k, ranks)

    # 3) Efficiency (Transport R2 / PCA R2)
    efficiency = {}
    for r in ranks:
        cca_r2 = cca_dict[int(r)]
        trans_r2 = trans_dict[r]["R2_test"]
        if cca_r2 > 1e-8 and trans_r2 is not None:
            efficiency[int(r)] = float(trans_r2 / cca_r2)
        else:
            efficiency[int(r)] = float("nan")

    print(efficiency)

    # 4) Optional orthogonal complement analysis
    ortho_results = {}
    # 5) Generate plots if requested
    try:
        plt.figure(figsize=(12, 5))

        # Main comparison plot
        plt.subplot(1, 2, 1)
        pca_r2_values = [cca_dict[r] for r in ranks]
        trans_r2_values = [
            trans_dict[r]["R2_test"] if trans_dict[r]["R2_test"] is not None else 0
            for r in ranks
        ]

        plt.plot(ranks, pca_r2_values, marker="o", label="PCA ceiling", linewidth=2)
        plt.plot(
            ranks,
            trans_r2_values,
            marker="s",
            label="Transport (rank-r)",
            linewidth=2,
        )
        plt.xlabel("Rank r")
        plt.ylabel("Variance-weighted $R^2$ (test)")
        plt.title("PCA vs Transport ($X \\to Y$)")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        # Efficiency plot
        plt.subplot(1, 2, 2)
        eff_values = [efficiency[r] for r in ranks]
        plt.plot(ranks, eff_values, marker="o", linewidth=2)
        plt.xlabel("Rank r")
        plt.ylabel("Efficiency = $R^2_T(r) / R^2_{PCA}(r)$")
        plt.title("Transport Efficiency vs Rank")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.ylim(0, 1.1)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    # Compile results
    results = {
        "ranks": ranks,
        "pca_R2": {int(r): float(cca_dict[int(r)]["R2"]) for r in ranks},
        "pca_explained_variance": {
            int(r): float(cca_dict[int(r)]["explained_variance_ratio"]) for r in ranks
        },
        "transport": trans_dict,  # contains R2_val, R2_test, alpha per rank
        "efficiency": {int(r): float(efficiency[int(r)]) for r in ranks},
        "orthogonal_complement_R2": ortho_results,
        "summary_stats": {
            "max_pca_r2": max(cca_dict[r]["R2"] for r in ranks),
            "max_transport_r2": max(
                trans_dict[r]["R2_test"]
                for r in ranks
                if trans_dict[r]["R2_test"] is not None
            ),
            "mean_efficiency": np.nanmean([efficiency[r] for r in ranks]),
            "best_rank_pca": max(ranks, key=lambda r: cca_dict[r]["R2"]),
            "best_rank_transport": max(
                ranks,
                key=lambda r: trans_dict[r]["R2_test"]
                if trans_dict[r]["R2_test"] is not None
                else -1,
            ),
        },
    }

    logger.info("Matched-rank analysis completed successfully")
    return results


def run_matched_PCA_rank_analysis_from_datasets(
    train_dataset: ActivationDataset | EfficientActivationDataset,
    val_dataset: ActivationDataset | EfficientActivationDataset,
    test_dataset: ActivationDataset | EfficientActivationDataset,
    ranks: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    orthogonal_test_ranks: list[int] | None = None,
    plot: bool = True,
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
    x_train, y_train = dataset_to_arrays(train_dataset, max_samples)
    x_val, y_val = dataset_to_arrays(val_dataset, max_samples)
    x_test, y_test = dataset_to_arrays(test_dataset, max_samples)

    logger.info(f"Dataset conversion complete. Final shapes:")
    logger.info(f"  Train: X={x_train.shape}, Y={y_train.shape}")
    logger.info(f"  Val: X={x_val.shape}, Y={y_val.shape}")
    logger.info(f"  Test: X={x_test.shape}, Y={y_test.shape}")

    # Run the main analysis
    return compare_pca_vs_transport(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        ranks=ranks,
        alpha_grid=alpha_grid,
        orthogonal_test_ranks=orthogonal_test_ranks,
        plot=plot,
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

    logger.info(f"Dataset conversion complete. Final shapes:")
    logger.info(f"  Train: X={x_train.shape}, Y={y_train.shape}")
    logger.info(f"  Test: X={x_test.shape}, Y={y_test.shape}")

    # Run the main analysis
    return compare_cca_vs_transport(L, k, x_train, y_train, x_test, y_test, ranks)
