import logging
import random
import socket
from typing import Any, cast
from src.transport_operator import TransportOperator
from src.activation_loader import ActivationLoader, get_train_val_test_datasets

import hydra
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    load_dotenv()
    print(socket.gethostname())
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_seed(cfg.get("seed", 42))
    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.experiment_name,
        config=cast(dict[str, Any] | None,
                    OmegaConf.to_container(cfg, resolve=True)),
        mode=cfg.logger.wandb_mode,  # NOTE: disabled by default
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Loaded configuration:")
    logging.info(cfg)

    transport_operators = {}
    loader = ActivationLoader(
        activation_dir_path="activations-gemma2-2b-slimpajama-250k",
        # files_to_download=[
        #     "activations-gemma2-2b-slimpajama-500k/activations_part_0000.zarr.zip"
        # ]
    )
    for L in cfg.experiment.L:
        for k in cfg.experiment.k:
            logging.info(f"Training transport operator for L={L}, k={k}")
            train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
                L, k, loader
            )

            # Initialize transport operator with better convergence settings
            transport_operator = TransportOperator(
                L=L,
                k=k,
                # Ridge is more stable than ElasticNet
                method=cfg.get("method", "ridge"),
                # Higher regularization for stability
                normalize=cfg.get("normalize", False),
                regularization=cfg.get("regularization", 10.0),
                # Less L1, more L2 for ElasticNet
                l1_ratio=cfg.get("l1_ratio", 0.1),
                # normalize=True,  # Enable normalization to prevent overflow
                auto_tune=cfg.get("auto_tune", True),
                # Reduce CV folds for faster training
                cv_folds=cfg.get("cv_folds", 5),
                random_state=cfg.seed,
                # Increase iterations for convergence
                max_iter=cfg.get("max_iter", 500),
                tol=cfg.get("tol", 1e-3),  # Relax tolerance slightly
                n_proc_cv=cfg.get("n_proc_cv", 2),  # Use 2 processes for CV
            )

            try:
                # Fit the transport operator
                transport_operator.fit(train_dataset)

                # Evaluate on validation set
                val_metrics = transport_operator.evaluate_dataset(val_dataset)
                logging.info(f"Validation metrics for L={L}, k={k}:")
                logging.info(f"  R² Score: {val_metrics['r2_score']:.4f}")
                logging.info(f"  RMSE: {val_metrics['rmse']:.6f}")
                if 'num_outputs' in val_metrics:
                    logging.info(
                        f"  Per-output R² range: [{val_metrics['r2_per_output_min']:.4f}, {val_metrics['r2_per_output_max']:.4f}]")

                # Log key metrics to wandb
                wandb_metrics = {
                    f"val_r2_L{L}_k{k}": val_metrics['r2_score'],
                    f"val_mse_L{L}_k{k}": val_metrics['mse'],
                    f"val_rmse_L{L}_k{k}": val_metrics['rmse'],
                    "L": L,
                    "k": k
                }

                # Add per-output summary metrics if available
                if 'num_outputs' in val_metrics:
                    wandb_metrics.update({
                        f"val_r2_mean_per_output_L{L}_k{k}": val_metrics['r2_per_output_mean'],
                        f"val_r2_std_per_output_L{L}_k{k}": val_metrics['r2_per_output_std'],
                        f"val_r2_min_per_output_L{L}_k{k}": val_metrics['r2_per_output_min'],
                        f"val_r2_max_per_output_L{L}_k{k}": val_metrics['r2_per_output_max'],
                    })

                wandb.log(wandb_metrics)

                transport_operators[(L, k)] = transport_operator

            except Exception as e:
                logging.error(
                    f"Error training transport operator for L={L}, k={k}: {e}")
                continue  # Continue with next L,k pair instead of crashing

    logging.info(
        f"Successfully trained {len(transport_operators)} transport operators")
    wandb.finish()


if __name__ == "__main__":
    main()
