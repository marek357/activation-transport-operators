from __future__ import annotations

import json
import logging
import random
import time
import warnings
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from src.activation_loader import ActivationDataset, ActivationLoader, partition_loader
from src.sae_loader import load_sae_from_cfg

logger = logging.getLogger(__name__)


def load_sae_decoders(cfg: DictConfig, activation_dtype: torch.dtype) -> dict[int, torch.Tensor]:
    """Load SAE decoders for all required layers based on config."""
    sae_decoders = {}

    # Get all unique layers that we need SAEs for (both source and target layers)
    required_layers = set()
    for layer_l in cfg.eval.Ls:
        required_layers.add(layer_l)  # Source layer
        for k in cfg.eval.ks:
            required_layers.add(layer_l + k)  # Target layer L+k

    logger.info("Loading SAE decoders for layers: %s", sorted(required_layers))

    for layer in required_layers:
        # Create a copy of the SAE config and modify the layer
        sae_cfg = OmegaConf.to_container(cfg.sae, resolve=True)
        sae_cfg["layer"] = layer
        sae_cfg = OmegaConf.create(sae_cfg)

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


def generate_feature_dict(cfg: DictConfig) -> dict[int, list[int]]:
    """Generate feature dict for each layer based on config."""
    # TODO: dummy implementation, make sure to re-write
    feature_dict = {}

    for layer_l in cfg.eval.Ls:
        for k in cfg.eval.ks:
            feature_dict[layer_l + k] = [0, 1, 2]

    return feature_dict


def score_latent_default(a_true: torch.Tensor, a_pred: torch.Tensor) -> tuple[float, float, float]:
    """Default scoring function for latent activations."""
    # R-squared
    ss_res = torch.sum((a_true - a_pred) ** 2)
    ss_tot = torch.sum((a_true - torch.mean(a_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

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
    r2_per_dim = torch.where(ss_tot < eps, torch.tensor(0.0, dtype=torch.float64), r2_per_dim)
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
    y_true_norms = torch.where(y_true_norms < eps, torch.tensor(eps, dtype=torch.float64), y_true_norms)
    y_pred_norms = torch.where(y_pred_norms < eps, torch.tensor(eps, dtype=torch.float64), y_pred_norms)

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


class MetricAggregator:
    """Aggregate metrics across batches for efficient computation."""

    def __init__(self) -> None:
        """Initialize the metric aggregator."""
        self.reset()

    def reset(self) -> None:
        """Reset all aggregated statistics."""
        self.n_samples = 0
        self.sum_true = 0.0
        self.sum_pred = 0.0
        self.sum_true_sq = 0.0
        self.sum_pred_sq = 0.0
        self.sum_true_pred = 0.0
        self.sum_squared_error = 0.0

    def update(self, a_true: torch.Tensor, a_pred: torch.Tensor) -> None:
        """Update aggregated statistics with a batch."""
        batch_size = a_true.shape[0]
        self.n_samples += batch_size

        # Convert to double precision for numerical stability
        a_true_f = a_true.double()
        a_pred_f = a_pred.double()

        # Update sums with overflow protection
        # Note: We don't scale here because metrics like R² and correlation are scale-invariant
        # and MSE should reflect the actual scale of the data
        try:
            self.sum_true += torch.sum(a_true_f).item()
            self.sum_pred += torch.sum(a_pred_f).item()
            self.sum_true_sq += torch.sum(a_true_f**2).item()
            self.sum_pred_sq += torch.sum(a_pred_f**2).item()
            self.sum_true_pred += torch.sum(a_true_f * a_pred_f).item()
            self.sum_squared_error += torch.sum((a_true_f - a_pred_f) ** 2).item()
        except (OverflowError, RuntimeError):
            # Handle overflow by using smaller chunks or different precision
            # For now, skip this batch and warn
            warnings.warn("Numerical overflow detected in metric aggregation, skipping batch", stacklevel=2)
            self.n_samples -= batch_size

    def compute_metrics(self) -> dict[str, float]:
        """Compute final metrics from aggregated statistics."""
        if self.n_samples == 0:
            return {"r2_lat": 0.0, "mse_lat": 0.0, "r_pearson": 0.0}

        # Constants for numerical stability
        EPS = 1e-12
        MIN_SAMPLES = 2

        if self.n_samples < MIN_SAMPLES:
            return {"r2_lat": 0.0, "mse_lat": 0.0, "r_pearson": 0.0}

        # Means
        mean_true = self.sum_true / self.n_samples
        mean_pred = self.sum_pred / self.n_samples

        # MSE
        mse = self.sum_squared_error / self.n_samples

        # R-squared with numerical stability
        ss_tot = self.sum_true_sq - self.n_samples * mean_true**2
        if ss_tot < EPS:
            r2 = 0.0  # No variance in true values
        else:
            r2 = 1.0 - (self.sum_squared_error / ss_tot)
            r2 = max(-1e6, min(1.0, r2))  # Clip extreme values

        # Pearson correlation with numerical stability
        cov = self.sum_true_pred - self.n_samples * mean_true * mean_pred
        var_true = self.sum_true_sq - self.n_samples * mean_true**2
        var_pred = self.sum_pred_sq - self.n_samples * mean_pred**2

        denominator = (var_true * var_pred) ** 0.5
        if denominator < EPS:
            r_pearson = 0.0
        else:
            r_pearson = cov / denominator
            r_pearson = max(-1.0, min(1.0, r_pearson))  # Clip to valid range

        return {
            "r2_lat": float(r2),
            "mse_lat": float(mse),
            "r_pearson": float(r_pearson),
        }


class CalibrationAggregator:
    """Aggregate calibration metrics across batches."""

    def __init__(self) -> None:
        """Initialize the calibration aggregator."""
        self.reset()

    def reset(self) -> None:
        """Reset all aggregated statistics."""
        self.n_samples = 0
        self.sum_sq_true = 0.0
        self.sum_sq_pred = 0.0

    def update(self, a_true: torch.Tensor, a_pred: torch.Tensor) -> None:
        """Update aggregated statistics with a batch."""
        batch_size = a_true.shape[0]
        self.n_samples += batch_size

        # Convert to double precision for numerical stability
        a_true_f = a_true.double()
        a_pred_f = a_pred.double()

        # Update sums with overflow protection
        # Note: No scaling needed as calibration metrics are typically scale-invariant
        try:
            self.sum_sq_true += torch.sum(a_true_f**2).item()
            self.sum_sq_pred += torch.sum(a_pred_f**2).item()
        except (OverflowError, RuntimeError):
            # Handle overflow by skipping this batch and warn
            warnings.warn("Numerical overflow in calibration aggregation, skipping batch", stacklevel=2)
            self.n_samples -= batch_size

    def compute_metrics(self) -> dict[str, float]:
        """Compute final calibration metrics from aggregated statistics."""
        if self.n_samples < 2:
            return {"calibration": 0.0}

        eps = 1e-12

        # Determinant for projection matrix calibration
        det = self.sum_sq_true * self.sum_sq_pred
        if abs(det) < eps:
            calibration = 0.0
        else:
            calibration = float(det**0.5 / self.n_samples)

        return {"calibration": calibration}


def _evaluate_batch_feature(
    y_dn_batch: torch.Tensor,
    y_hat_batch: torch.Tensor,
    decoder_vector: torch.Tensor,
    metric_aggregator: MetricAggregator,
    calib_aggregator: CalibrationAggregator,
    *,
    normalize_decoder: bool = True,
):
    """Evaluate a single feature on a batch and update aggregators."""
    d_f = decoder_vector

    if normalize_decoder:
        d_f = d_f / (torch.norm(d_f) + 1e-8)

    # Project residuals onto feature direction
    a_true = torch.matmul(y_dn_batch, d_f)  # [batch_size]
    a_pred = torch.matmul(y_hat_batch, d_f)  # [batch_size]

    # Update aggregators
    metric_aggregator.update(a_true, a_pred)
    calib_aggregator.update(a_true, a_pred)


def run_experiment(
    transport_maps: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]],
    chosen_layers: list[int],
    activation_loader: ActivationLoader,
    k_list: list[int],
    j_policy_list: list[str],
    features_dict: dict[int, list[int]],
    sae_decoders: dict[int, torch.Tensor],
    score_residual: callable | None = None,
    *,
    decoder_normalize: bool = True,
    val_batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
) -> dict[tuple[int, int, str, int], dict[str, float]]:
    """Run evaluation of transport operators on SAE features."""
    results = {}
    _, _, test_idx = partition_loader(
        len(activation_loader),
        *(0.8, 0.1, 0.1),
    )

    logger.info("Using device: %s", device)
    logger.info("Evaluating on %d test samples", len(test_idx))

    for layer_l in chosen_layers:
        for k in k_list:
            for j_policy in j_policy_list:
                logger.info("Evaluating L=%d, k=%d, j_policy=%s", layer_l, k, j_policy)

                # Check if transport map exists
                if (layer_l, k, j_policy) not in transport_maps:
                    logger.warning("Transport map not found for (L=%d, k=%d, j_policy=%s)", layer_l, k, j_policy)
                    continue

                dataset = ActivationDataset(
                    activation_loader,
                    test_idx,
                    j_policy,
                    layer_l,
                    k,
                )
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=val_batch_size,
                    num_workers=8,
                    persistent_workers=True,
                )

                transport_t, bias_b = transport_maps[(layer_l, k, j_policy)]
                transport_t = transport_t.to(device)
                bias_b = bias_b.to(device)

                # Prepare decoders for the target layer L+k
                target_layer = layer_l + k
                if target_layer not in sae_decoders:
                    logger.warning("SAE decoder not found for target layer %d", target_layer)
                    continue

                if target_layer not in features_dict:
                    logger.warning("Feature list not found for target layer %d", target_layer)
                    continue

                decoder_matrix = sae_decoders[target_layer].to(device)

                # Initialize aggregators for each feature
                feature_aggregators = {}
                feature_calibrators = {}
                for feat_idx in features_dict[target_layer]:
                    feature_aggregators[feat_idx] = MetricAggregator()
                    feature_calibrators[feat_idx] = CalibrationAggregator()

                # For residual metrics (if needed)
                residual_y_true_all = []
                residual_y_pred_all = []
                collect_residual = score_residual is not None

                # Process batches one by one
                total_samples = 0
                batch_start_time = time.time()
                for batch_idx, (x_up_batch, y_dn_batch) in enumerate(dataloader):
                    x_up_batch: torch.Tensor = x_up_batch.to(device)
                    y_dn_batch: torch.Tensor = y_dn_batch.to(device)

                    # Predict downstream residuals using transport operator
                    with torch.no_grad():
                        y_hat_batch = torch.addmm(bias_b, x_up_batch, transport_t.T)

                    # Collect for residual metrics if needed (small memory overhead)
                    if collect_residual:
                        residual_y_true_all.append(y_dn_batch.cpu())
                        residual_y_pred_all.append(y_hat_batch.cpu())

                    # Evaluate each feature on this batch
                    for feat_idx in features_dict[target_layer]:
                        decoder_vector = decoder_matrix[feat_idx].cpu()
                        _evaluate_batch_feature(
                            y_dn_batch.cpu(),
                            y_hat_batch.cpu(),
                            decoder_vector,
                            feature_aggregators[feat_idx],
                            feature_calibrators[feat_idx],
                            normalize_decoder=decoder_normalize,
                        )

                    total_samples += x_up_batch.shape[0]

                    # Log progress every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        elapsed_time = time.time() - batch_start_time
                        batches_per_sec = (batch_idx + 1) / elapsed_time
                        samples_per_sec = total_samples / elapsed_time
                        logger.info(
                            "L=%d, k=%d, j_policy=%s: Processed %d batches (%d samples) - "
                            "%.2f batches/sec, %.1f samples/sec, %.1fs elapsed",
                            layer_l,
                            k,
                            j_policy,
                            batch_idx + 1,
                            total_samples,
                            batches_per_sec,
                            samples_per_sec,
                            elapsed_time,
                        )

                logger.info("Processed %d samples for L=%d, k=%d, j_policy=%s", total_samples, layer_l, k, j_policy)

                # Compute residual-level metrics if needed
                res_metrics = {}
                if collect_residual:
                    y_dn_all = torch.cat(residual_y_true_all, dim=0).numpy()
                    y_hat_all = torch.cat(residual_y_pred_all, dim=0).numpy()
                    res_metrics = score_residual(y_dn_all, y_hat_all)

                # Finalize metrics for each feature
                for feat_idx in features_dict[target_layer]:
                    # Get aggregated metrics
                    feat_metrics = feature_aggregators[feat_idx].compute_metrics()
                    calib_metrics = feature_calibrators[feat_idx].compute_metrics()

                    results[(layer_l, k, j_policy, feat_idx)] = {
                        **feat_metrics,
                        **calib_metrics,
                        **res_metrics,
                    }

    return results


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
    else:
        msg = f"Unsupported activation dtype: {cfg.activation_dtype}"
        raise NotImplementedError(msg)

    logger.info("Using activation dtype: %s", cfg.activation_dtype)

    # Load SAE decoders for all required layers
    logger.info("Loading SAE decoders...")
    sae_decoders = load_sae_decoders(cfg, activation_dtype=activation_dtype)

    # Load activation loader
    logger.info("Loading activation data...")
    activation_loader = ActivationLoader(cfg.activation_dir)

    # Generate feature lists
    logger.info("Generating feature lists...")
    feature_dict = generate_feature_dict(cfg)

    # NOTE: You need to implement loading of transport_maps
    # This would typically come from your transport operator training/loading code
    transport_maps = {}  # dict[(L,k,j_policy)] -> (T:[d,d], b:[d])

    # TODO: Replace with proper loading
    for L in cfg.eval.Ls:
        for k in cfg.eval.ks:
            transport_maps[(L, k, "j==i")] = (
                torch.eye(2304, dtype=activation_dtype),
                torch.zeros(2304, dtype=activation_dtype),
            )

    # Set up scoring functions
    score_residual_fn = score_residual_default if cfg.eval.scoring.include_residual_metrics else None

    # Run evaluation
    logger.info("Running evaluation...")

    results = run_experiment(
        transport_maps=transport_maps,
        chosen_layers=cfg.eval.Ls,
        activation_loader=activation_loader,
        k_list=cfg.eval.ks,
        j_policy_list=cfg.eval.j_policy,
        features_dict=feature_dict,
        sae_decoders=sae_decoders,
        score_residual=score_residual_fn,
        decoder_normalize=cfg.eval.decoders.normalize,
        val_batch_size=cfg.eval.val_batch_size,
    )

    logger.info("Evaluation completed. Generated %d result entries.", len(results))

    # Save results to JSON

    # Convert results to hierarchical JSON-serializable format
    json_results = {}
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
    output_file = output_dir / f"{cfg.experiment_name}_results.json"
    with output_file.open("w") as f:
        json.dump(json_results, f, indent=2)

    logger.info("Results saved to: %s", output_file)

    # Log summary to wandb
    wandb.log({"num_results": len(results)})
    wandb.finish()

    return json_results


if __name__ == "__main__":
    main()
