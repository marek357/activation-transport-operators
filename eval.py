from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, cast, Callable

import hydra
import numpy as np
import torch
from tqdm import tqdm
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from src.activation_loader import (
    ActivationLoader,
    get_train_val_test_datasets,
)
from src.sae_loader import load_sae_from_cfg
from src.transport_operator import (
    IdentityBaselineTransportOperator,
    TransportOperator,
    load_transport_operator,
)
from src.matched_rank_analysis import run_matched_rank_analysis_from_datasets

logger = logging.getLogger(__name__)


def _get_eval_cache_filename(
    layer_l: int,
    k: int,
    j_policy: str,
    dataset_id: str = "default",
    dataset_num_sequences: int = 0,
) -> str:
    """Generate a unique cache filename for evaluation data."""
    cache_info = {
        "layer_l": layer_l,
        "k": k,
        "j_policy": j_policy,
        "dataset_id": dataset_id,
        "dataset_num_sequences": dataset_num_sequences,
    }
    cache_str = str(sorted(cache_info.items()))
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:8]
    return f"eval_data_{cache_hash}.pkl"


def _save_eval_cache(
    x_up_all: np.ndarray,
    y_dn_all: np.ndarray,
    y_hat_all: np.ndarray,
    feature_masks_all: dict[int, np.ndarray],
    cache_path: str,
) -> None:
    """Save evaluation data to cache file."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            "x_up_all": x_up_all,
            "y_dn_all": y_dn_all,
            "y_hat_all": y_hat_all,
            "feature_masks_all": feature_masks_all,
            "timestamp": time.time(),
            "x_shape": x_up_all.shape,
            "y_shape": y_dn_all.shape,
            "y_hat_shape": y_hat_all.shape,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"  Cached evaluation data saved to: {cache_path}")
    except Exception as e:
        logger.warning(f"  Warning: Failed to save cache: {e}")


def _load_eval_cache(
    cache_path: str,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]
    | tuple[None, None, None, None]
):
    """Load evaluation data from cache file."""
    try:
        if not os.path.exists(cache_path):
            return None, None, None, None

        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)

        x_up_all = cache_data["x_up_all"]
        y_dn_all = cache_data["y_dn_all"]
        y_hat_all = cache_data["y_hat_all"]
        feature_masks_all = cache_data["feature_masks_all"]
        timestamp = cache_data.get("timestamp", 0)

        # Basic validation
        if x_up_all.ndim != 2 or y_dn_all.ndim != 2 or y_hat_all.ndim != 2:
            logger.warning("  Invalid cached data dimensions")
            return None, None, None, None

        cache_age = time.time() - timestamp
        logger.info(f"  Loaded cached evaluation data from: {cache_path}")
        logger.info(f"  Cache age: {cache_age / 3600:.1f} hours")
        logger.info(
            f"  X shape: {x_up_all.shape}, Y shape: {y_dn_all.shape}, Y_hat shape: {y_hat_all.shape}"
        )

        return x_up_all, y_dn_all, y_hat_all, feature_masks_all

    except Exception as e:
        logger.warning(f"  Warning: Failed to load cache: {e}")
        return None, None, None, None


def compute_all_feature_metrics(
    y_dn_all: np.ndarray,
    y_hat_all: np.ndarray,
    decoder_matrix: torch.Tensor,
    feature_list: list[int],
    feature_masks_all: dict[int, np.ndarray] | None = None,
    *,
    normalize_decoder: bool = True,
) -> dict[int, dict[str, float]]:
    """
    Compute metrics for all features in one go without batching/aggregation.

    Args:
        y_dn_all: All true residual stream activations [n_samples, d_model]
        y_hat_all: All predicted residual stream activations [n_samples, d_model]
        decoder_matrix: SAE decoder matrix [n_features, d_model]
        feature_list: List of feature indices to evaluate
        feature_masks_all: Optional masks indicating which samples to include for each feature
        normalize_decoder: Whether to normalize decoder vectors

    Returns:
        Dictionary mapping feature_idx -> metrics dict
    """
    results = {}

    # Convert to tensors for efficient computation
    y_dn_tensor = torch.from_numpy(y_dn_all).double()
    y_hat_tensor = torch.from_numpy(y_hat_all).double()

    total_samples = y_dn_all.shape[0]

    for feat_idx in tqdm(feature_list, desc="Computing metrics for features"):
        # Get decoder vector
        d_f = decoder_matrix[feat_idx].double()

        if normalize_decoder:
            d_f = d_f / (torch.norm(d_f) + 1e-12)

        # Project residuals onto feature direction
        a_true = torch.matmul(y_dn_tensor, d_f)  # [n_samples]
        a_pred = torch.matmul(y_hat_tensor, d_f)  # [n_samples]

        # Apply feature mask if provided (for filtering inactive features)
        if feature_masks_all is not None and feat_idx in feature_masks_all:
            mask = feature_masks_all[feat_idx]
            if not np.any(mask):
                # Feature never activated
                results[feat_idx] = {
                    "r2_lat": 0.0,
                    "mse_lat": 0.0,
                    "r_pearson": 0.0,
                    "calibration": 0.0,
                    "rms_ratio": 1.0,
                    "mean_abs_ratio": 1.0,
                    "mean_ratio": 1.0,
                    "mad_ratio": 0.0,
                    "rel_error_mean": 0.0,
                    "log_mse_ratio": 0.0,
                    "rms_true": 0.0,
                    "rms_pred": 0.0,
                    "feature_never_activated": True,
                    "activation_count": 0,
                    "total_samples": total_samples,
                    "activation_rate": 0.0,
                }
                continue

            # Filter to activated samples
            mask_tensor = torch.from_numpy(mask)
            a_true = a_true[mask_tensor]
            a_pred = a_pred[mask_tensor]
            activation_count = mask.sum()
        else:
            # Use all samples
            activation_count = total_samples

        n_samples = a_true.shape[0]

        if n_samples == 0:
            # No samples to evaluate
            results[feat_idx] = {
                "r2_lat": 0.0,
                "mse_lat": 0.0,
                "r_pearson": 0.0,
                "calibration": 0.0,
                "rms_ratio": 1.0,
                "mean_abs_ratio": 1.0,
                "mean_ratio": 1.0,
                "mad_ratio": 0.0,
                "rel_error_mean": 0.0,
                "log_mse_ratio": 0.0,
                "rms_true": 0.0,
                "rms_pred": 0.0,
                "feature_never_activated": True,
                "activation_count": 0,
                "total_samples": total_samples,
                "activation_rate": 0.0,
            }
            continue

        # Compute all metrics at once
        metrics = _compute_single_feature_metrics(a_true, a_pred)

        # Add metadata
        metrics.update(
            {
                "feature_never_activated": False,
                "activation_count": int(activation_count),
                "total_samples": total_samples,
                "activation_rate": float(activation_count) / total_samples
                if total_samples > 0
                else 0.0,
            }
        )

        results[feat_idx] = metrics

    return results


def _compute_single_feature_metrics(
    a_true: torch.Tensor, a_pred: torch.Tensor
) -> dict[str, float]:
    """Compute all metrics for a single feature without aggregation."""
    n_samples = a_true.shape[0]

    if n_samples < 2:
        return {
            "r2_lat": 0.0,
            "mse_lat": 0.0,
            "r_pearson": 0.0,
            "calibration": 0.0,
            "rms_ratio": 1.0,
            "mean_abs_ratio": 1.0,
            "mean_ratio": 1.0,
            "mad_ratio": 0.0,
            "rel_error_mean": 0.0,
            "log_mse_ratio": 0.0,
            "rms_true": 0.0,
            "rms_pred": 0.0,
        }

    eps = 1e-12

    # Basic statistics
    mean_true = torch.mean(a_true)
    mean_pred = torch.mean(a_pred)

    # MSE
    mse = torch.mean((a_true - a_pred) ** 2)

    # R-squared
    ss_tot = torch.sum((a_true - mean_true) ** 2)
    if ss_tot < eps:
        r2 = 0.0
    else:
        ss_res = torch.sum((a_true - a_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r2 = max(-1e6, min(1.0, r2))  # Clamp extreme values

    # Pearson correlation
    cov = torch.mean((a_true - mean_true) * (a_pred - mean_pred))
    var_true = torch.mean((a_true - mean_true) ** 2)
    var_pred = torch.mean((a_pred - mean_pred) ** 2)

    denominator = torch.sqrt(var_true * var_pred)
    if denominator < eps:
        r_pearson = 0.0
    else:
        r_pearson = cov / denominator
        r_pearson = max(-1.0, min(1.0, r_pearson))  # Clamp to valid range

    # Calibration metrics
    sum_true_sq = torch.sum(a_true**2)
    sum_pred_sq = torch.sum(a_pred**2)
    sum_true_pred = torch.sum(a_true * a_pred)

    # Geometric mean calibration
    det = sum_true_sq * sum_pred_sq
    if abs(det) < eps:
        calibration = 0.0
    else:
        calibration = float(sum_true_pred / torch.sqrt(det))

    # RMS values and ratio
    rms_true = torch.sqrt(torch.mean(a_true**2))
    rms_pred = torch.sqrt(torch.mean(a_pred**2))
    rms_ratio = rms_pred / (rms_true + eps) if rms_true > eps else 1.0

    # Mean absolute values and ratio
    mean_abs_true = torch.mean(torch.abs(a_true))
    mean_abs_pred = torch.mean(torch.abs(a_pred))
    mean_abs_ratio = (
        mean_abs_pred / (mean_abs_true + eps) if mean_abs_true > eps else 1.0
    )

    # Mean ratio (bias detection)
    mean_ratio = mean_pred / (mean_true + eps) if abs(mean_true) > eps else 1.0

    # Mean Absolute Deviation ratio
    mad_ratio = (
        torch.mean(torch.abs(a_true - a_pred)) / (mean_abs_true + eps)
        if mean_abs_true > eps
        else 0.0
    )

    # Relative error
    eps_rel = 1e-12
    nonzero_mask = torch.abs(a_true) > eps_rel
    if nonzero_mask.any():
        rel_errors = torch.abs(a_true[nonzero_mask] - a_pred[nonzero_mask]) / torch.abs(
            a_true[nonzero_mask]
        )
        rel_error_mean = torch.mean(rel_errors)
    else:
        rel_error_mean = 0.0

    # Log-scale calibration
    both_nonzero = (torch.abs(a_true) > eps_rel) & (torch.abs(a_pred) > eps_rel)
    if both_nonzero.any():
        log_ratios = torch.log(
            torch.abs(a_pred[both_nonzero]) / torch.abs(a_true[both_nonzero])
        )
        log_mse_ratio = torch.mean(log_ratios**2)
    else:
        log_mse_ratio = 0.0

    return {
        "r2_lat": float(r2),
        "mse_lat": float(mse),
        "r_pearson": float(r_pearson),
        "calibration": float(calibration),
        "rms_ratio": float(rms_ratio),
        "mean_abs_ratio": float(mean_abs_ratio),
        "mean_ratio": float(mean_ratio),
        "mad_ratio": float(mad_ratio),
        "rel_error_mean": float(rel_error_mean),
        "log_mse_ratio": float(log_mse_ratio),
        "rms_true": float(rms_true),
        "rms_pred": float(rms_pred),
    }


def load_sae_decoders(
    cfg: DictConfig, activation_dtype: torch.dtype
) -> dict[int, torch.Tensor]:
    """Load SAE decoders for all required layers based on config."""
    sae_decoders = {}

    # Get all unique layers that we need SAEs for (both source and target layers)
    required_layers = set()
    for layer_l in cfg.eval.Ls:
        for k in cfg.eval.ks:
            required_layers.add(layer_l + k)  # Target layer L+k

    logger.info("Loading SAE decoders for layers: %s", sorted(required_layers))

    for layer in required_layers:
        # Create a copy of the SAE config and modify the layer
        cfg.sae.layer = layer
        sae_cfg = OmegaConf.to_container(cfg.sae, resolve=True)

        # Create a temporary config with the modified SAE config
        temp_cfg = OmegaConf.create({"sae": sae_cfg})

        try:
            sae, _, _ = load_sae_from_cfg(temp_cfg)
            # Extract decoder weights (W_dec)
            decoder_weights = (
                sae.W_dec.detach().clone().to(activation_dtype)
            )  # Ensure correct dtype,  Shape: [d_sae, d_model]
            sae_decoders[layer] = decoder_weights
            logger.info(
                "Loaded SAE decoder for layer %d: shape %s and dtype %s",
                layer,
                decoder_weights.shape,
                decoder_weights.dtype,
            )
        except Exception:
            logger.exception("Failed to load SAE for layer %d", layer)
            raise

    return sae_decoders


def load_full_saes(cfg: DictConfig) -> dict[int, Any]:
    """Load full SAE models for all required layers based on config."""
    saes = {}

    # Get all unique layers that we need SAEs for (both source and target layers)
    required_layers = set()
    for layer_l in cfg.eval.Ls:
        for k in cfg.eval.ks:
            required_layers.add(layer_l + k)  # Target layer L+k

    logger.info("Loading full SAEs for layers: %s", sorted(required_layers))

    for layer in required_layers:
        # Create a copy of the SAE config and modify the layer
        cfg.sae.layer = layer
        sae_cfg = OmegaConf.to_container(cfg.sae, resolve=True)

        # Create a temporary config with the modified SAE config
        temp_cfg = OmegaConf.create({"sae": sae_cfg})

        try:
            sae, _, _ = load_sae_from_cfg(temp_cfg)
            saes[layer] = sae
            logger.info(
                "Loaded full SAE for layer %d: d_sae=%s, d_model=%s",
                layer,
                getattr(sae.cfg, "d_sae", "unknown"),
                getattr(sae.cfg, "d_in", "unknown"),
            )
        except Exception:
            logger.exception("Failed to load SAE for layer %d", layer)
            raise

    return saes


def get_sample_feature_activations(
    y_dn_tensor: torch.Tensor,
    sae: Any,
    available_features: list[int],
) -> dict[int, torch.Tensor]:
    """
    Get sample-level feature activation masks for the given tensor.

    For JumpReLU SAEs, this properly handles learnable thresholds.

    Args:
        y_dn_tensor: Residual stream activations [n_samples, d_model]
        sae: SAE model for encoding (supports both ReLU and JumpReLU architectures)
        available_features: List of feature indices to check

    Returns:
        Dictionary mapping feature_idx -> boolean mask [n_samples] indicating
        which samples have that feature activated
    """
    with torch.no_grad():
        # Move SAE to the same device as the tensor
        sae_device = next(iter(sae.parameters())).device
        if sae_device != y_dn_tensor.device:
            sae = sae.to(y_dn_tensor.device)

        # Check SAE architecture to handle different activation functions
        sae_arch = (
            getattr(sae.cfg, "architecture", "standard")
            if hasattr(sae, "cfg")
            else "standard"
        )

        # Get feature activations using the SAE's encode method
        # This should handle JumpReLU, TopK, and other architectures automatically
        try:
            feature_activations = sae.encode(y_dn_tensor)  # [n_samples, d_sae]
        except Exception as e:
            logger.exception("Failed to encode features using SAE")
            raise RuntimeError("SAE encode failed") from e

        # For JumpReLU and TopK SAEs, the encode method already applies the correct
        # activation function with learnable thresholds or sparsity constraints.
        # We create boolean masks indicating which samples have each feature activated.

        feature_masks = {}
        activation_threshold = 1e-6  # Very small threshold for detecting any activation

        for feat_idx in available_features:
            feat_acts = feature_activations[:, feat_idx]  # [n_samples]

            # Boolean mask: True where feature is activated for each sample
            feature_masks[feat_idx] = feat_acts > activation_threshold

        # Log architecture info and activation stats for debugging
        total_activations = sum(mask.sum().item() for mask in feature_masks.values())
        logger.debug(
            f"SAE arch: {sae_arch}, processed {len(available_features)} features, "
            f"total sample activations: {total_activations}"
        )

        return feature_masks


def generate_feature_dict(cfg: DictConfig) -> dict[int, list[int]]:
    """Generate feature dict for each layer based on config."""
    # TODO: dummy implementation, make sure to re-write
    feature_dict = {}

    # Load feature lists from JSON files in the feature_ids_dir
    feature_ids_dir = Path(cfg.feature_ids_dir)
    logger.info(f"Loading feature lists from: {feature_ids_dir}")

    # Get all unique layers that we need features for
    required_layers = set()
    for layer_l in cfg.eval.Ls:
        for k in cfg.eval.ks:
            required_layers.add(layer_l + k)  # Target layer L+k

    for layer_id in required_layers:
        feature_file = feature_ids_dir / f"feature_scores_{layer_id}.json"

        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")

        try:
            with open(feature_file, "r") as f:
                feature_data = json.load(f)

            feature_dict[layer_id] = feature_data["high_quality_feature_ids"][
                : min(
                    cfg.eval.max_features, len(feature_data["high_quality_feature_ids"])
                )
            ]

            logger.info(
                f"Loaded {len(feature_dict[layer_id])} features for layer {layer_id}"
            )

        except Exception as e:
            logger.error(f"Failed to load features from {feature_file}: {e}")
            feature_dict[layer_id] = []

    return feature_dict


def create_baseline_transport_operators(
    cfg: DictConfig, activation_loader: ActivationLoader
) -> dict[
    tuple[int, int, str],
    TransportOperator
    | IdentityBaselineTransportOperator,
]:
    """Create baseline transport operators for evaluation."""
    baseline_operators = {}

    baseline_configs = cfg.get("baselines", {})

    for layer_l in cfg.eval.Ls:
        for k in cfg.eval.ks:
            for j_policy in cfg.eval.j_policy:
                # Get training dataset for fitting baselines
                train_dataset, _, _ = get_train_val_test_datasets(
                    layer_l, k, activation_loader, j_policy
                )

                # Create Identity baseline if enabled
                if baseline_configs.get("enable_identity", False):
                    identity_op = IdentityBaselineTransportOperator(L=layer_l, k=k)
                    # No actual fitting needed, but for consistency
                    identity_op.fit(train_dataset)
                    baseline_operators[(layer_l, k, f"{j_policy}_identity")] = (
                        identity_op
                    )

                    logger.info(
                        "Created Identity baseline for L=%d, k=%d, j_policy=%s",
                        layer_l,
                        k,
                        j_policy,
                    )

    return baseline_operators


def load_trained_transport_operators(
    cfg: DictConfig, activation_loader: ActivationLoader
) -> dict[tuple[int, int, str], TransportOperator]:
    """
    Load trained transport operators from saved models.

    This function should be implemented to load your trained transport operators.
    """
    transport_operators = {}

    # TODO: Implement loading of trained transport operators
    # Example implementation:

    # Option 1: Load from pickle files
    # for layer_l in cfg.eval.Ls:
    #     for k in cfg.eval.ks:
    #         for j_policy in cfg.eval.j_policy:
    #             model_file = Path(cfg.get("model_dir", "models")) / f"transport_L{layer_l}_k{k}_{j_policy}.pkl"
    #             if model_file.exists():
    #                 with open(model_file, 'rb') as f:
    #                     transport_op = pickle.load(f)
    #                 transport_operators[(layer_l, k, j_policy)] = transport_op
    #                 logger.info(f"Loaded transport operator from {model_file}")
    #             else:
    #                 logger.warning(f"Model file not found: {model_file}")

    # Option 2: Train operators on-demand
    # for layer_l in cfg.eval.Ls:
    #     for k in cfg.eval.ks:
    #         for j_policy in cfg.eval.j_policy:
    #             train_dataset, _, _ = get_train_val_test_datasets(
    #                 layer_l, k, activation_loader, j_policy
    #             )
    #             transport_op = TransportOperator(L=layer_l, k=k, method="ridge", )
    #             transport_op.fit(train_dataset)
    #             transport_operators[(layer_l, k, j_policy)] = transport_op
    #             logger.info(
    #                 f"Trained transport operator for L={layer_l}, k={k}, j_policy={j_policy}"
    #             )

    # logger.info("Loading trained transport operators...")

    # for layer_l in cfg.eval.Ls:
    #     for k in cfg.eval.ks:
    #         for j_policy in cfg.eval.j_policy:
    #             # TODO: Replace with actual loading from saved .pkl files
    #             # For now, create a dummy trained operator
    #             transport_op = TransportOperator(L=layer_l, k=k, method="ridge")
    #             # In a real implementation, the operator would already be fitted

    #             transport_operators[(layer_l, k, j_policy)] = transport_op

    #             logger.info(
    #                 "Created placeholder transport operator for L=%d, k=%d, j_policy=%s",
    #                 layer_l,
    #                 k,
    #                 j_policy,
    #             )

    for layer_l in cfg.eval.Ls:
        for k in cfg.eval.ks:
            for j_policy in cfg.eval.j_policy:
                # Note, that currently loading does not support choosing the j_policy because it is not hashed
                transport_operators[(layer_l, k, j_policy)] = load_transport_operator(
                    layer_l, k, "./cache"
                )

    return transport_operators


def create_dummy_transport_operators(
    cfg: DictConfig, activation_loader: ActivationLoader
) -> dict[tuple[int, int, str], TransportOperator]:
    """
    Create dummy transport operators for testing/demonstration.

    This function creates identity transport operators as placeholders.
    Replace this with actual loading logic for your trained operators.
    """
    transport_operators = {}

    logger.info("Creating dummy transport operators for testing...")

    for L in cfg.eval.Ls:
        for k in cfg.eval.ks:
            for j_policy in cfg.eval.j_policy:
                # Create a dummy transport operator
                # You could use different methods here: "linear", "ridge", "lasso", etc.
                transport_op = TransportOperator(
                    L=L,
                    k=k,
                    method="linear",  # Simple linear regression as dummy
                    normalize=False,
                    auto_tune=False,
                )

                # TODO: In a real implementation, you would:
                # 1. Load the operator from a saved file, or
                # 2. Train it on your data here, or
                # 3. Load pre-computed transport matrices and create a custom operator

                # For now, we'll leave it untrained (it will act as identity when predict() is called)
                transport_operators[(L, k, j_policy)] = transport_op

                logger.info(
                    "Created dummy transport operator for L=%d, k=%d, j_policy=%s",
                    L,
                    k,
                    j_policy,
                )

    return transport_operators


def score_latent_default(
    a_true: torch.Tensor, a_pred: torch.Tensor
) -> tuple[float, float, float]:
    """Default scoring function for latent activations."""
    # R-squared
    ss_res = torch.sum((a_true - a_pred) ** 2)
    ss_tot = torch.sum((a_true - torch.mean(a_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-12))

    # MSE
    mse = torch.mean((a_true - a_pred) ** 2)

    # Pearson correlation
    correlation_matrix = torch.corrcoef(torch.stack([a_true, a_pred]))
    r_pearson = correlation_matrix[0, 1]

    return float(r2), float(mse), float(r_pearson)


def score_residual_default(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Score residual stream predictions with multiple metrics."""
    # Convert to PyTorch tensors with double precision for numerical stability
    y_true_t = torch.from_numpy(y_true).double()
    y_pred_t = torch.from_numpy(y_pred).double()

    # Check for extremely large values that might cause overflow
    max_val = max(torch.abs(y_true_t).max().item(), torch.abs(y_pred_t).max().item())
    if max_val > 1e6:
        # Scale down values to prevent overflow
        scale_factor = 1e6 / max_val
        y_true_t = y_true_t * scale_factor
        y_pred_t = y_pred_t * scale_factor

    # R-squared (mean across dimensions) - using more stable computation
    diff = y_true_t - y_pred_t
    ss_res = torch.sum(diff**2, dim=0)  # Per dimension

    # Use more stable variance computation
    y_true_mean = torch.mean(y_true_t, dim=0, keepdim=True)
    y_true_centered = y_true_t - y_true_mean
    ss_tot = torch.sum(y_true_centered**2, dim=0)

    # Avoid division by zero and handle near-zero variance
    eps = 1e-12
    r2_per_dim = 1 - (ss_res / (ss_tot + eps))
    r2_per_dim = torch.where(
        ss_tot < eps, torch.tensor(0.0, dtype=torch.float64), r2_per_dim
    )
    r2_per_dim = torch.clamp(r2_per_dim, -1e6, 1.0)  # Clip extreme values

    # Only use finite values for mean
    finite_mask = torch.isfinite(r2_per_dim)
    r2_mean = float(torch.mean(r2_per_dim[finite_mask])) if finite_mask.any() else 0.0

    # MSE (mean across all elements)
    mse = float(torch.mean(diff**2))

    # Cosine similarity (mean across samples) - more stable computation
    # Normalize each sample
    y_true_norms = torch.norm(y_true_t, dim=1, keepdim=True)
    y_pred_norms = torch.norm(y_pred_t, dim=1, keepdim=True)

    # Handle zero norms
    y_true_norms = torch.where(
        y_true_norms < eps, torch.tensor(eps, dtype=torch.float64), y_true_norms
    )
    y_pred_norms = torch.where(
        y_pred_norms < eps, torch.tensor(eps, dtype=torch.float64), y_pred_norms
    )

    y_true_norm = y_true_t / y_true_norms
    y_pred_norm = y_pred_t / y_pred_norms

    cos_sim = torch.sum(y_true_norm * y_pred_norm, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # Ensure valid cosine values
    cos_mean = float(torch.mean(cos_sim))

    return {
        "r2_res": r2_mean,
        "mse_res": mse,
        "cos_res": cos_mean,
    }


def interpret_calibration_metrics(calib_metrics: dict[str, float]) -> dict[str, str]:
    """Interpret calibration metrics and provide qualitative assessments."""
    interpretation = {}

    # RMS ratio interpretation
    rms_ratio = calib_metrics.get("rms_ratio", 1.0)
    if 0.9 <= rms_ratio <= 1.1:
        interpretation["rms_assessment"] = "excellent"
    elif 0.8 <= rms_ratio <= 1.25:
        interpretation["rms_assessment"] = "good"
    elif 0.6 <= rms_ratio <= 1.67:
        interpretation["rms_assessment"] = "fair"
    else:
        interpretation["rms_assessment"] = "poor"

    # Mean absolute ratio interpretation
    mean_abs_ratio = calib_metrics.get("mean_abs_ratio", 1.0)
    if 0.9 <= mean_abs_ratio <= 1.1:
        interpretation["magnitude_assessment"] = "excellent"
    elif 0.8 <= mean_abs_ratio <= 1.25:
        interpretation["magnitude_assessment"] = "good"
    elif 0.6 <= mean_abs_ratio <= 1.67:
        interpretation["magnitude_assessment"] = "fair"
    else:
        interpretation["magnitude_assessment"] = "poor"

    # Bias detection from mean ratio
    mean_ratio = calib_metrics.get("mean_ratio", 1.0)
    if abs(mean_ratio - 1.0) < 0.1:
        interpretation["bias_assessment"] = "minimal"
    elif abs(mean_ratio - 1.0) < 0.3:
        interpretation["bias_assessment"] = "moderate"
    else:
        interpretation["bias_assessment"] = "significant"
        interpretation["bias_direction"] = (
            "overestimation" if mean_ratio > 1.0 else "underestimation"
        )

    # Relative error interpretation
    rel_error = calib_metrics.get("rel_error_mean", 0.0)
    if rel_error < 0.2:
        interpretation["precision_assessment"] = "excellent"
    elif rel_error < 0.5:
        interpretation["precision_assessment"] = "good"
    elif rel_error < 1.0:
        interpretation["precision_assessment"] = "fair"
    else:
        interpretation["precision_assessment"] = "poor"

    # Log-scale calibration (multiplicative consistency)
    log_mse = calib_metrics.get("log_mse_ratio", 0.0)
    if log_mse < 0.1:
        interpretation["consistency_assessment"] = "excellent"
    elif log_mse < 0.5:
        interpretation["consistency_assessment"] = "good"
    elif log_mse < 1.0:
        interpretation["consistency_assessment"] = "fair"
    else:
        interpretation["consistency_assessment"] = "poor"

    return interpretation


def run_matched_rank_experiment(
    cfg: DictConfig,
    activation_loader: ActivationLoader,
) -> dict[str, Any]:
    """
    Run matched-rank curves analysis comparing CCA ceiling vs rank-r transport.

    The key insight is that R²_T(r) ≤ R²_CCA(r) is expected (CCA is the best possible
    rank-r Y-only reconstruction). The gap shows how much of compressible variance
    is actually predictable from X.
    """
    logger.info("Running matched-rank curves analysis...")

    matched_rank_results = {}

    # Get configuration for matched-rank analysis
    matched_rank_cfg = cfg.get("matched_rank", {})
    # Allow ranks to be specified directly, or via start/stop/step/max
    if "ranks" in matched_rank_cfg:
        ranks = matched_rank_cfg["ranks"]
    else:
        rank_start = matched_rank_cfg.get("rank_start", 1)
        rank_stop = matched_rank_cfg.get("rank_stop", 2300)
        rank_step = matched_rank_cfg.get("rank_step", 100)
        rank_max = matched_rank_cfg.get("rank_max", 2304)
        ranks = list(range(rank_start, rank_stop, rank_step))
        if rank_max not in ranks:
            ranks.append(rank_max)  # the full rank
    alpha_grid = matched_rank_cfg.get("alpha_grid", [0.1, 1.0, 10.0, 100.0])
    orthogonal_test_ranks = matched_rank_cfg.get("orthogonal_test_ranks", [16, 32, 64])
    # Limit samples for computational efficiency
    max_samples = matched_rank_cfg.get("max_samples", None)

    logger.info("Matched-rank configuration:")
    logger.info(f"  Ranks to evaluate: {ranks}")
    logger.info(f"  Alpha grid: {alpha_grid}")
    logger.info(f"  Orthogonal test ranks: {orthogonal_test_ranks}")
    logger.info(f"  Max samples per dataset: {max_samples}")

    for layer_l in cfg.eval.Ls:
        for k in cfg.eval.ks:
            for j_policy in cfg.eval.j_policy:
                logger.info(
                    f"Matched-rank analysis for L={layer_l}, k={k}, j_policy={j_policy}"
                )

                try:
                    # Get train/val/test datasets
                    train_dataset, val_dataset, test_dataset = (
                        get_train_val_test_datasets(
                            layer_l, k, activation_loader, j_policy
                        )
                    )

                    # Run matched-rank analysis
                    results = run_matched_rank_analysis_from_datasets(
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        ranks=ranks,
                        max_samples=max_samples,
                        L=layer_l,
                        k=k,
                    )

                    # Store results
                    key = (layer_l, k, j_policy)
                    matched_rank_results[key] = results

                    # Log summary
                    summary = results["summary_stats"]
                    logger.info(
                        f"  Max CCA R²: {summary['max_cca_r2']:.4f} at rank {summary['best_rank_cca']}"
                    )
                    logger.info(
                        f"  Max Transport R²: {summary['max_transport_r2']:.4f} at rank {summary['best_rank_transport']}"
                    )
                    logger.info(f"  Mean efficiency: {summary['mean_efficiency']:.4f}")

                    # Log rank-by-rank comparison
                    logger.info("  Rank-by-rank comparison:")
                    for r in ranks:
                        cca_r2 = results["cca_R2"][r]
                        trans_r2 = results["transport"][r]["R2_test"]
                        efficiency = results["efficiency"][r]
                        alpha = results["transport"][r]["alpha"]

                        if trans_r2 is not None:
                            alpha_str = f"{alpha}" if alpha is not None else "None"
                            logger.info(
                                f"    r={r:3d}: CCA={cca_r2:.3f}, Transport={trans_r2:.3f} "
                                f"(α={alpha_str}), Efficiency={efficiency:.3f}"
                            )
                        else:
                            logger.info(
                                f"    r={r:3d}: CCA={cca_r2:.3f}, Transport=FAILED"
                            )

                except Exception as e:
                    logger.exception(
                        f"Matched-rank analysis failed for L={layer_l}, k={k}, j_policy={j_policy}: {e}",
                        stack_info=True,
                    )
                    raise Exception from e

    return matched_rank_results


def summarize_matched_rank_results(matched_rank_results: dict) -> dict[str, Any]:
    """Summarize matched-rank analysis results across all configurations."""
    if not matched_rank_results:
        return {}

    summary = {
        "configurations": [],
        "aggregate_stats": {
            "max_cca_r2_overall": 0.0,
            "max_transport_r2_overall": 0.0,
            "mean_efficiency_overall": 0.0,
            "best_configs": {},
        },
        "rank_performance": {},
    }

    all_efficiencies = []
    all_cca_r2 = []
    all_transport_r2 = []

    for key, results in matched_rank_results.items():
        layer_l, k, j_policy = key

        # Store configuration results
        config_summary = {
            "layer_l": layer_l,
            "k": k,
            "j_policy": j_policy,
            "max_cca_r2": results["summary_stats"]["max_cca_r2"],
            "max_transport_r2": results["summary_stats"]["max_transport_r2"],
            "mean_efficiency": results["summary_stats"]["mean_efficiency"],
            "best_rank_cca": results["summary_stats"]["best_rank_cca"],
            "best_rank_transport": results["summary_stats"]["best_rank_transport"],
        }
        summary["configurations"].append(config_summary)

        # Collect values for aggregate stats
        for r in results["ranks"]:
            all_cca_r2.append(results["cca_R2"][r])
            if results["transport"][r]["R2_test"] is not None:
                all_transport_r2.append(results["transport"][r]["R2_test"])
            if not np.isnan(results["efficiency"][r]):
                all_efficiencies.append(results["efficiency"][r])

        # Track best overall performance
        if (
            results["summary_stats"]["max_cca_r2"]
            > summary["aggregate_stats"]["max_cca_r2_overall"]
        ):
            summary["aggregate_stats"]["max_cca_r2_overall"] = results["summary_stats"][
                "max_cca_r2"
            ]
            summary["aggregate_stats"]["best_configs"]["cca"] = key

        if (
            results["summary_stats"]["max_transport_r2"]
            > summary["aggregate_stats"]["max_transport_r2_overall"]
        ):
            summary["aggregate_stats"]["max_transport_r2_overall"] = results[
                "summary_stats"
            ]["max_transport_r2"]
            summary["aggregate_stats"]["best_configs"]["transport"] = key

    # Compute aggregate statistics
    if all_efficiencies:
        summary["aggregate_stats"]["mean_efficiency_overall"] = float(
            np.mean(all_efficiencies)
        )
        summary["aggregate_stats"]["std_efficiency_overall"] = float(
            np.std(all_efficiencies)
        )
        summary["aggregate_stats"]["median_efficiency_overall"] = float(
            np.median(all_efficiencies)
        )

    if all_cca_r2:
        summary["aggregate_stats"]["mean_cca_r2_overall"] = float(np.mean(all_cca_r2))

    if all_transport_r2:
        summary["aggregate_stats"]["mean_transport_r2_overall"] = float(
            np.mean(all_transport_r2)
        )

    # Performance by rank
    if matched_rank_results:
        first_result = next(iter(matched_rank_results.values()))
        for r in first_result["ranks"]:
            rank_cca_r2 = []
            rank_transport_r2 = []
            rank_efficiency = []

            for results in matched_rank_results.values():
                rank_cca_r2.append(results["cca_R2"][r])
                if results["transport"][r]["R2_test"] is not None:
                    rank_transport_r2.append(results["transport"][r]["R2_test"])
                if not np.isnan(results["efficiency"][r]):
                    rank_efficiency.append(results["efficiency"][r])

            summary["rank_performance"][r] = {
                "mean_cca_r2": float(np.mean(rank_cca_r2)) if rank_cca_r2 else 0.0,
                "mean_transport_r2": float(np.mean(rank_transport_r2))
                if rank_transport_r2
                else 0.0,
                "mean_efficiency": float(np.mean(rank_efficiency))
                if rank_efficiency
                else 0.0,
                "std_efficiency": float(np.std(rank_efficiency))
                if rank_efficiency
                else 0.0,
            }

    return summary


def run_experiment(
    transport_operators: dict[
        tuple[int, int, str],
        TransportOperator
        | IdentityBaselineTransportOperator,
    ],
    chosen_layers: list[int],
    activation_loader: ActivationLoader,
    k_list: list[int],
    j_policy_list: list[str],
    features_dict: dict[int, list[int]],
    sae_decoders: dict[int, torch.Tensor],
    saes: dict[int, Any],
    score_residual: Callable | None = None,
    *,
    decoder_normalize: bool = True,
    filter_inactive_features: bool = True,
    device: str = "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu",
) -> dict[tuple[int, int, str, int], dict[str, float]]:
    """Run evaluation of transport operators on SAE features."""
    # Input validation
    if not transport_operators:
        raise ValueError("transport_operators must be provided and non-empty")

    results = {}

    logger.info("Using device: %s", device)

    for layer_l in chosen_layers:
        for k in k_list:
            for j_policy in j_policy_list:
                logger.info("Evaluating L=%d, k=%d, j_policy=%s", layer_l, k, j_policy)

                # Check if transport operator exists
                if (layer_l, k, j_policy) not in transport_operators:
                    logger.warning(
                        "Transport operator not found for (L=%d, k=%d, j_policy=%s)",
                        layer_l,
                        k,
                        j_policy,
                    )
                    continue

                transport_op = transport_operators[(layer_l, k, j_policy)]

                _, _, dataset = get_train_val_test_datasets(
                    layer_l,
                    k,
                    activation_loader,
                    j_policy,
                )
                logger.info("Evaluating on %d test samples", len(dataset.idx_list))

                # Prepare decoders for the target layer L+k
                target_layer = layer_l + k
                if target_layer not in sae_decoders:
                    logger.warning(
                        "SAE decoder not found for target layer %d", target_layer
                    )
                    continue

                if target_layer not in features_dict:
                    logger.warning(
                        "Feature list not found for target layer %d", target_layer
                    )
                    continue

                if filter_inactive_features and target_layer not in saes:
                    logger.warning(
                        "Full SAE not found for target layer %d (needed for feature filtering)",
                        target_layer,
                    )
                    continue

                decoder_matrix = sae_decoders[target_layer].to(device)
                target_sae = (
                    saes.get(target_layer) if filter_inactive_features else None
                )

                # Use cache directory similar to transport operator
                cache_dir = "cache"

                # Check for cached evaluation data first
                dataset_id = getattr(dataset, "dataset_id", "default")
                dataset_num_sequences = len(dataset.idx_list)
                cache_filename = _get_eval_cache_filename(
                    layer_l, k, j_policy, dataset_id, dataset_num_sequences
                )
                cache_path = os.path.join(cache_dir, cache_filename)

                logger.info(f"Checking for cached evaluation data: {cache_path}")
                x_up_all, y_dn_all, y_hat_all, feature_masks_all = _load_eval_cache(
                    cache_path
                )

                # If no cached data found, load from dataset
                if x_up_all is None:
                    logger.info("Loading evaluation data from dataset...")

                    # Load all data at once using DataLoader
                    dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=256,  # Larger batch size for efficient loading
                        num_workers=0,
                    )

                    x_up_list = []
                    y_dn_list = []
                    y_hat_list = []

                    total_samples = 0
                    load_start_time = time.time()

                    for batch_idx, (x_up_batch, y_dn_batch) in enumerate(dataloader):
                        x_up_batch = x_up_batch.to(device)
                        y_dn_batch = y_dn_batch.to(device)

                        # Predict using transport operator
                        with torch.no_grad():
                            x_np = x_up_batch.cpu().numpy()
                            y_hat_np = transport_op.predict(x_np)
                            y_hat_batch = (
                                torch.from_numpy(y_hat_np)
                                .to(y_dn_batch.dtype)
                                .to(device)
                            )

                        # Collect data
                        x_up_list.append(x_up_batch.cpu().numpy())
                        y_dn_list.append(y_dn_batch.cpu().numpy())
                        y_hat_list.append(y_hat_batch.cpu().numpy())

                        total_samples += x_up_batch.shape[0]

                        # Log progress
                        if (batch_idx + 1) % 50 == 0:
                            elapsed_time = time.time() - load_start_time
                            samples_per_sec = total_samples / elapsed_time
                            logger.info(
                                f"  Loaded {batch_idx + 1} batches ({total_samples} samples) - "
                                f"{samples_per_sec:.1f} samples/sec"
                            )

                    # Stack all data
                    x_up_all = np.concatenate(x_up_list, axis=0)
                    y_dn_all = np.concatenate(y_dn_list, axis=0)
                    y_hat_all = np.concatenate(y_hat_list, axis=0)

                    load_time = time.time() - load_start_time
                    logger.info(
                        f"Data loading complete: {total_samples:,} samples in {load_time:.2f}s"
                    )
                    logger.info(
                        f"  X shape: {x_up_all.shape}, Y shape: {y_dn_all.shape}, Y_hat shape: {y_hat_all.shape}"
                    )

                    # Compute feature masks if filtering is enabled
                    if filter_inactive_features and target_sae is not None:
                        logger.info("Computing feature activation masks...")

                        # Convert to tensors for SAE processing
                        y_dn_tensor = torch.from_numpy(y_dn_all).to(device)

                        # Get feature masks for all samples
                        feature_masks_all = get_sample_feature_activations(
                            y_dn_tensor,
                            target_sae,
                            features_dict[target_layer],
                        )

                        # Convert masks to numpy for caching
                        feature_masks_all = {
                            feat_idx: mask.cpu().numpy()
                            for feat_idx, mask in feature_masks_all.items()
                        }

                        # Log activation statistics
                        total_activations = sum(
                            mask.sum() for mask in feature_masks_all.values()
                        )
                        logger.info(
                            f"  Total feature activations: {total_activations:,}"
                        )

                    else:
                        # No filtering - all features are "activated" for all samples
                        feature_masks_all = {
                            feat_idx: np.ones(total_samples, dtype=bool)
                            for feat_idx in features_dict[target_layer]
                        }

                    # Save to cache
                    _save_eval_cache(
                        x_up_all, y_dn_all, y_hat_all, feature_masks_all, cache_path
                    )

                else:
                    logger.info("Using cached evaluation data")
                    total_samples = y_dn_all.shape[0]

                # Compute residual-level metrics if needed
                res_metrics = {}
                if score_residual is not None:
                    logger.info("Computing residual-level metrics...")
                    res_metrics = score_residual(y_dn_all, y_hat_all)

                # Compute feature-level metrics all at once
                logger.info(
                    f"Computing metrics for {len(features_dict[target_layer])} features..."
                )
                metrics_start_time = time.time()

                feature_results = compute_all_feature_metrics(
                    y_dn_all,
                    y_hat_all,
                    decoder_matrix.cpu(),
                    features_dict[target_layer],
                    feature_masks_all,
                    normalize_decoder=decoder_normalize,
                )

                metrics_time = time.time() - metrics_start_time
                logger.info(f"Feature metrics computed in {metrics_time:.2f}s")

                # Store results with interpretation
                for feat_idx, metrics in feature_results.items():
                    # Add residual metrics and calibration interpretation
                    calib_interpretation = interpret_calibration_metrics(metrics)

                    results[(layer_l, k, j_policy, feat_idx)] = {
                        **metrics,
                        **res_metrics,
                        "calib_interpretation": calib_interpretation,
                    }

                # Log feature activation summary
                if filter_inactive_features:
                    activation_counts = [
                        feature_masks_all[feat_idx].sum()
                        for feat_idx in features_dict[target_layer]
                    ]
                    never_activated = sum(
                        1 for count in activation_counts if count == 0
                    )
                    always_activated = sum(
                        1 for count in activation_counts if count == total_samples
                    )

                    logger.info(
                        f"Feature activation summary: {never_activated} never activated, "
                        f"{always_activated} always activated, "
                        f"{len(activation_counts) - never_activated - always_activated} partially activated"
                    )

    return results


def summarize_calibration_across_features(results: dict) -> dict[str, Any]:
    """Summarize calibration performance across all features and conditions."""
    calibration_summary = {
        "rms_ratios": [],
        "mean_abs_ratios": [],
        "rel_errors": [],
        "log_mse_ratios": [],
        "assessments": {"excellent": 0, "good": 0, "fair": 0, "poor": 0},
        "bias_directions": {"overestimation": 0, "underestimation": 0, "minimal": 0},
        "activation_summary": {
            "total_features": 0,
            "never_activated_features": 0,
            "activated_features": 0,
            "activation_rates": [],  # List of activation rates for all features (0-1 scale, not percentages)
            "activation_counts": [],  # List of activation counts for all features
            "total_samples": 0,  # Total number of samples processed
        },
    }

    for key, metrics in results.items():
        layer_l, k, j_policy, feat_idx = key

        # Track feature activation statistics
        calibration_summary["activation_summary"]["total_features"] += 1

        # Collect activation statistics for all features
        activation_count = metrics.get("activation_count", 0)
        activation_rate = metrics.get("activation_rate", 0.0)
        total_samples_feat = metrics.get("total_samples", 0)

        calibration_summary["activation_summary"]["activation_counts"].append(
            activation_count
        )
        calibration_summary["activation_summary"]["activation_rates"].append(
            activation_rate
        )

        # Track total samples (should be consistent across features)
        if calibration_summary["activation_summary"]["total_samples"] == 0:
            calibration_summary["activation_summary"]["total_samples"] = (
                total_samples_feat
            )

        if metrics.get("feature_never_activated", False):
            calibration_summary["activation_summary"]["never_activated_features"] += 1
            # Skip metrics collection for never-activated features
            continue
        else:
            calibration_summary["activation_summary"]["activated_features"] += 1

        # Regular feature metrics (only for activated features)
        # Collect numerical metrics
        calibration_summary["rms_ratios"].append(metrics.get("rms_ratio", 1.0))
        calibration_summary["mean_abs_ratios"].append(
            metrics.get("mean_abs_ratio", 1.0)
        )
        calibration_summary["rel_errors"].append(metrics.get("rel_error_mean", 0.0))
        calibration_summary["log_mse_ratios"].append(metrics.get("log_mse_ratio", 0.0))

        # Count qualitative assessments
        interpretation = metrics.get("calib_interpretation", {})
        rms_assessment = interpretation.get("rms_assessment", "unknown")
        if rms_assessment in calibration_summary["assessments"]:
            calibration_summary["assessments"][rms_assessment] += 1

        bias_assessment = interpretation.get("bias_assessment", "minimal")
        bias_direction = interpretation.get("bias_direction", "minimal")
        if bias_direction in calibration_summary["bias_directions"]:
            calibration_summary["bias_directions"][bias_direction] += 1
        elif bias_assessment == "minimal":
            calibration_summary["bias_directions"]["minimal"] += 1

    # Compute aggregate statistics
    import numpy as np

    if calibration_summary["rms_ratios"]:
        calibration_summary["rms_ratio_mean"] = float(
            np.mean(calibration_summary["rms_ratios"])
        )
        calibration_summary["rms_ratio_std"] = float(
            np.std(calibration_summary["rms_ratios"])
        )
        calibration_summary["mean_abs_ratio_mean"] = float(
            np.mean(calibration_summary["mean_abs_ratios"])
        )
        calibration_summary["mean_abs_ratio_std"] = float(
            np.std(calibration_summary["mean_abs_ratios"])
        )
        calibration_summary["rel_error_mean"] = float(
            np.mean(calibration_summary["rel_errors"])
        )
        calibration_summary["rel_error_std"] = float(
            np.std(calibration_summary["rel_errors"])
        )

        # Overall calibration quality score (0-1, higher is better)
        total_features = len(calibration_summary["rms_ratios"])
        excellent_ratio = (
            calibration_summary["assessments"]["excellent"] / total_features
        )
        good_ratio = calibration_summary["assessments"]["good"] / total_features
        calibration_summary["overall_quality_score"] = (
            excellent_ratio + 0.7 * good_ratio
        )

    # Compute activation summary statistics
    activation_rates = calibration_summary["activation_summary"]["activation_rates"]

    if activation_rates:
        import numpy as np

        calibration_summary["activation_summary"]["mean_activation_rate"] = float(
            np.mean(activation_rates)
        )
        calibration_summary["activation_summary"]["std_activation_rate"] = float(
            np.std(activation_rates)
        )
        calibration_summary["activation_summary"]["median_activation_rate"] = float(
            np.median(activation_rates)
        )
        calibration_summary["activation_summary"]["min_activation_rate"] = float(
            np.min(activation_rates)
        )
        calibration_summary["activation_summary"]["max_activation_rate"] = float(
            np.max(activation_rates)
        )

        # Add metadata to clarify units
        calibration_summary["activation_summary"]["activation_rate_units"] = "ratio"
        calibration_summary["activation_summary"]["activation_rate_description"] = (
            "Values are in 0-1 range, not percentages"
        )

        # Categorize features by activation frequency
        always_active = sum(1 for rate in activation_rates if rate >= 0.99)
        often_active = sum(1 for rate in activation_rates if 0.5 <= rate < 0.99)
        sometimes_active = sum(1 for rate in activation_rates if 0.01 < rate < 0.5)
        rarely_active = sum(1 for rate in activation_rates if 0 < rate <= 0.01)

        calibration_summary["activation_summary"]["always_active_features"] = (
            always_active
        )
        calibration_summary["activation_summary"]["often_active_features"] = (
            often_active
        )
        calibration_summary["activation_summary"]["sometimes_active_features"] = (
            sometimes_active
        )
        calibration_summary["activation_summary"]["rarely_active_features"] = (
            rarely_active
        )

    return calibration_summary


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> dict:
    """Run evaluation function using Hydra configuration."""
    load_dotenv()

    # Set random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    set_seed(cfg.get("seed", 42))

    # Initialize wandb
    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.experiment_name,
        config=cast("dict[str, Any] | None", OmegaConf.to_container(cfg, resolve=True)),
        mode=cfg.logger.wandb_mode,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Loaded evaluation configuration:")
    logger.info(cfg)

    if cfg.activation_dtype == "float16":
        activation_dtype = torch.float16
    elif cfg.activation_dtype == "float32":
        activation_dtype = torch.float32
    else:
        msg = f"Unsupported activation dtype: {cfg.activation_dtype}"
        raise NotImplementedError(msg)

    logger.info("Using activation dtype: %s", cfg.activation_dtype)

    # Load SAE decoders for all required layers
    logger.info("Loading SAE decoders...")
    sae_decoders = load_sae_decoders(cfg, activation_dtype=activation_dtype)

    # Load full SAEs for feature activation filtering (if enabled)
    filter_inactive_features = cfg.eval.get("filter_inactive_features", True)

    if filter_inactive_features:
        logger.info("Loading full SAEs for feature activation filtering...")
        saes = load_full_saes(cfg)
        logger.info(
            "Feature activation filtering enabled",
        )

    else:
        saes = {}
        logger.info(
            "Feature activation filtering disabled - will evaluate all features"
        )

    # Load activation loader
    logger.info("Loading activation data...")
    activation_loader = ActivationLoader(cfg.activation_dir)

    # Generate feature lists
    logger.info("Generating feature lists...")
    feature_dict = generate_feature_dict(cfg)

    # Check if matched-rank analysis is requested
    if cfg.get("run_matched_rank", False):
        logger.info("Running matched-rank analysis...")
        matched_rank_results = run_matched_rank_experiment(cfg, activation_loader)
        matched_rank_summary = summarize_matched_rank_results(matched_rank_results)

        # Log matched-rank summary
        logger.info("=== MATCHED-RANK ANALYSIS SUMMARY ===")
        aggregate_stats = matched_rank_summary.get("aggregate_stats", {})
        logger.info(
            f"Best overall CCA R²: {aggregate_stats.get('max_cca_r2_overall', 0):.4f}"
        )
        logger.info(
            f"Best overall Transport R²: {aggregate_stats.get('max_transport_r2_overall', 0):.4f}"
        )
        logger.info(
            f"Mean efficiency across all configs: {aggregate_stats.get('mean_efficiency_overall', 0):.4f} ± {aggregate_stats.get('std_efficiency_overall', 0):.4f}"
        )

        # Log per-rank performance
        rank_performance = matched_rank_summary.get("rank_performance", {})
        if rank_performance:
            logger.info("Performance by rank:")
            for rank in sorted(rank_performance.keys()):
                perf = rank_performance[rank]
                logger.info(
                    f"  Rank {rank}: CCA R²={perf['mean_cca_r2']:.3f}, "
                    f"Transport R²={perf['mean_transport_r2']:.3f}, "
                    f"Efficiency={perf['mean_efficiency']:.3f}±{perf['std_efficiency']:.3f}"
                )

        # Save matched-rank results
        output_dir = Path(cfg.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        matched_rank_file = (
            output_dir / f"{cfg.experiment_name}_matched_rank_results.json"
        )

        # Convert to JSON-serializable format
        json_matched_rank = {"summary": matched_rank_summary, "detailed_results": {}}
        for key, results in matched_rank_results.items():
            layer_l, k, j_policy = key
            key_str = f"L{layer_l}_k{k}_{j_policy}"
            json_matched_rank["detailed_results"][key_str] = results

        with matched_rank_file.open("w") as f:
            json.dump(json_matched_rank, f, indent=2)

        logger.info(f"Matched-rank results saved to: {matched_rank_file}")

        # Log to wandb
        wandb.log(
            {
                "matched_rank/max_cca_r2": aggregate_stats.get("max_cca_r2_overall", 0),
                "matched_rank/max_transport_r2": aggregate_stats.get(
                    "max_transport_r2_overall", 0
                ),
                "matched_rank/mean_efficiency": aggregate_stats.get(
                    "mean_efficiency_overall", 0
                ),
                "matched_rank/std_efficiency": aggregate_stats.get(
                    "std_efficiency_overall", 0
                ),
                "matched_rank/num_configurations": len(
                    matched_rank_summary.get("configurations", [])
                ),
            }
        )

        logger.info("=== END MATCHED-RANK ANALYSIS ===")

        # Return early if only running matched-rank analysis
        if cfg.get("matched_rank_only", False):
            wandb.finish()
            return json_matched_rank

    # Load or create transport mechanisms
    transport_operators = {}

    # Determine evaluation mode
    # baselines, pretrained, or dummy
    eval_mode = cfg.get("eval_mode", "baselines")

    if (
        eval_mode == "baselines"
        or cfg.get("baselines", {}).get("enable_identity", False)
    ):
        logger.info("Creating baseline transport operators...")
        transport_operators = create_baseline_transport_operators(
            cfg, activation_loader
        )

    elif eval_mode == "pretrained":
        logger.info("Loading trained transport operators...")
        transport_operators = load_trained_transport_operators(cfg, activation_loader)

    else:
        # Default mode: create dummy transport operators for testing
        logger.info("Creating dummy transport operators...")
        transport_operators = create_dummy_transport_operators(cfg, activation_loader)

    # Set up scoring functions
    score_residual_fn = (
        score_residual_default if cfg.eval.scoring.include_residual_metrics else None
    )

    # Run evaluation
    logger.info("Running evaluation...")

    results = run_experiment(
        transport_operators=transport_operators,
        chosen_layers=cfg.eval.Ls,
        activation_loader=activation_loader,
        k_list=cfg.eval.ks,
        j_policy_list=cfg.eval.j_policy,
        features_dict=feature_dict,
        sae_decoders=sae_decoders,
        saes=saes,
        score_residual=score_residual_fn,
        decoder_normalize=cfg.eval.decoders.normalize,
        filter_inactive_features=filter_inactive_features,
    )

    logger.info("Evaluation completed. Generated %d result entries.", len(results))

    # Analyze calibration performance across all features
    calibration_summary = summarize_calibration_across_features(results)
    logger.info("=== ANALYSIS SUMMARY ===")

    # Report feature activation statistics if available
    activation_summary = calibration_summary.get("activation_summary", {})
    if activation_summary.get("total_features", 0) > 0:
        logger.info("=== FEATURE ACTIVATION ANALYSIS ===")
        total_features = activation_summary["total_features"]
        activated_features = activation_summary["activated_features"]
        never_activated = activation_summary["never_activated_features"]

        activation_rate = (
            (activated_features / total_features) * 100 if total_features > 0 else 0
        )

        logger.info("Total features evaluated: %d", total_features)
        logger.info(
            "Features with activation data: %d (%.1f%%)",
            activated_features,
            activation_rate,
        )
        logger.info(
            "Features never activated: %d (%.1f%%)",
            never_activated,
            (never_activated / total_features) * 100 if total_features > 0 else 0,
        )

        if filter_inactive_features:
            logger.info(
                "Feature filtering was enabled",
            )
        else:
            logger.info("Feature filtering was disabled")

        logger.info("=== END FEATURE ACTIVATION ANALYSIS ===")

    # Report calibration metrics only if we have prediction-based results
    if calibration_summary["rms_ratios"]:
        logger.info("=== PREDICTION CALIBRATION ANALYSIS ===")
        logger.info(
            "Overall Quality Score: %.3f",
            calibration_summary.get("overall_quality_score", 0.0),
        )
        logger.info(
            "RMS Ratio: %.3f ± %.3f",
            calibration_summary.get("rms_ratio_mean", 1.0),
            calibration_summary.get("rms_ratio_std", 0.0),
        )
        logger.info(
            "Mean Abs Ratio: %.3f ± %.3f",
            calibration_summary.get("mean_abs_ratio_mean", 1.0),
            calibration_summary.get("mean_abs_ratio_std", 0.0),
        )
        logger.info(
            "Relative Error: %.3f ± %.3f",
            calibration_summary.get("rel_error_mean", 0.0),
            calibration_summary.get("rel_error_std", 0.0),
        )

        assessments = calibration_summary.get("assessments", {})
        logger.info(
            "Assessment distribution: Excellent=%d, Good=%d, Fair=%d, Poor=%d",
            assessments.get("excellent", 0),
            assessments.get("good", 0),
            assessments.get("fair", 0),
            assessments.get("poor", 0),
        )

        bias_dirs = calibration_summary.get("bias_directions", {})
        logger.info(
            "Bias analysis: Overest=%d, Underest=%d, Minimal=%d",
            bias_dirs.get("overestimation", 0),
            bias_dirs.get("underestimation", 0),
            bias_dirs.get("minimal", 0),
        )
    else:
        logger.info(
            "No prediction-based calibration metrics available"
        )

    # Save results to JSON

    # Convert results to hierarchical JSON-serializable format
    json_results = {"calibration_summary": calibration_summary}
    for key, metrics in results.items():
        layer_l, k, j_policy, feat_idx = key

        # Create nested structure: layer -> k -> policy -> features
        if f"layer_{layer_l}" not in json_results:
            json_results[f"layer_{layer_l}"] = {}

        if f"k_{k}" not in json_results[f"layer_{layer_l}"]:
            json_results[f"layer_{layer_l}"][f"k_{k}"] = {}

        if j_policy not in json_results[f"layer_{layer_l}"][f"k_{k}"]:
            json_results[f"layer_{layer_l}"][f"k_{k}"][j_policy] = {}

        json_results[f"layer_{layer_l}"][f"k_{k}"][j_policy][f"{feat_idx}"] = metrics

    # Create output directory if it doesn't exist
    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    output_file = output_dir / cfg.output_file_name
    with output_file.open("w") as f:
        json.dump(json_results, f, indent=2)

    logger.info("Results saved to: %s", output_file)

    # Log summary to wandb
    wandb.log(
        {
            "num_results": len(results),
            "calibration_quality_score": calibration_summary.get(
                "overall_quality_score", 0.0
            ),
            "rms_ratio_mean": calibration_summary.get("rms_ratio_mean", 1.0),
            "rms_ratio_std": calibration_summary.get("rms_ratio_std", 0.0),
            "mean_abs_ratio_mean": calibration_summary.get("mean_abs_ratio_mean", 1.0),
            "rel_error_mean": calibration_summary.get("rel_error_mean", 0.0),
            "excellent_calibrations": calibration_summary.get("assessments", {}).get(
                "excellent", 0
            ),
            "poor_calibrations": calibration_summary.get("assessments", {}).get(
                "poor", 0
            ),
        }
    )
    wandb.finish()

    return json_results


if __name__ == "__main__":
    main()
