import logging
import random
import socket
from typing import Any, cast, List, Dict, Tuple
from src.transport_operator import TransportOperator
from src.activation_loader import ActivationLoader, ActivationDataset

import hydra
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from pathlib import Path


def create_scaled_datasets(
    loader: ActivationLoader,
    L: int,
    k: int,
    data_fractions: List[float],
    j_policy: str = "j==i"
) -> Dict[float, Tuple[ActivationDataset, ActivationDataset, ActivationDataset]]:
    """
    Create datasets with different amounts of training data while keeping validation/test fixed.

    Args:
        loader: ActivationLoader instance
        L: Layer number
        k: Layer offset
        data_fractions: List of fractions of training data to use (e.g., [0.1, 0.25, 0.5, 0.75, 1.0])
        j_policy: Policy for data generation

    Returns:
        Dict mapping data fraction to (train_dataset, val_dataset, test_dataset)
    """
    total_samples = len(loader)

    # Fixed validation and test sizes (10% each)
    val_size = int(total_samples * 0.1)
    test_size = int(total_samples * 0.1)
    available_train_samples = total_samples - val_size - test_size

    # Fixed validation and test indices
    val_indices = list(range(total_samples - val_size -
                       test_size, total_samples - test_size))
    test_indices = list(range(total_samples - test_size, total_samples))

    datasets = {}

    for fraction in data_fractions:
        # Calculate training size for this fraction
        train_size = int(available_train_samples * fraction)
        train_indices = list(range(train_size))

        # Create datasets
        train_dataset = ActivationDataset(
            loader, train_indices, j_policy, L, k, f"train_frac_{fraction}_L{L}_k{k}")
        val_dataset = ActivationDataset(
            loader, val_indices, j_policy, L, k, f"val_L{L}_k{k}")
        test_dataset = ActivationDataset(
            loader, test_indices, j_policy, L, k, f"test_L{L}_k{k}")

        datasets[fraction] = (train_dataset, val_dataset, test_dataset)

        logging.info(
            f"Created datasets for fraction {fraction}: train={train_size}, val={len(val_indices)}, test={len(test_indices)}")

    return datasets


def run_scaling_experiment(
    datasets: Dict[float, Tuple[ActivationDataset, ActivationDataset, ActivationDataset]],
    L: int,
    k: int,
    cfg: DictConfig
) -> Dict[float, Dict[str, float]]:
    """
    Run scaling experiment for a single (L, k) pair.

    Returns:
        Dict mapping data fraction to metrics
    """
    results = {}

    for fraction in sorted(datasets.keys()):
        train_dataset, val_dataset, test_dataset = datasets[fraction]

        logging.info(
            f"Training with {fraction*100:.1f}% of data for L={L}, k={k}")

        # Initialize transport operator with same hyperparameters
        transport_operator = TransportOperator(
            L=L,
            k=k,
            method=cfg.get('method', 'ridge'),
            normalize=cfg.get('normalize', False),
            regularization=cfg.get('regularization', 10.0),
            l1_ratio=cfg.get('l1_ratio', 0.1),
            auto_tune=cfg.get('auto_tune', True),
            cv_folds=cfg.get('cv_folds', 5),
            random_state=cfg.seed,
            max_iter=cfg.get('max_iter', 500),
            tol=cfg.get('tol', 1e-3)
        )

        try:
            # Fit the transport operator
            transport_operator.fit(train_dataset)

            # Evaluate on validation set
            val_metrics = transport_operator.evaluate_dataset(val_dataset)

            # Also evaluate on test set for final comparison
            test_metrics = transport_operator.evaluate_dataset(test_dataset)

            # Store results
            results[fraction] = {
                'val_r2': val_metrics['r2_score'],
                'val_mse': val_metrics['mse'],
                'val_rmse': val_metrics['rmse'],
                'test_r2': test_metrics['r2_score'],
                'test_mse': test_metrics['mse'],
                'test_rmse': test_metrics['rmse'],
                'best_params': transport_operator.best_params_,
                'L': L,
                'k': k
            }

            logging.info(
                f"  Fraction {fraction}: Val R² = {val_metrics['r2_score']:.4f}, Test R² = {test_metrics['r2_score']:.4f}")

            # Log to wandb
            wandb_metrics = {
                f"scaling_val_r2_L{L}_k{k}_frac_{fraction}": val_metrics['r2_score'],
                f"scaling_test_r2_L{L}_k{k}_frac_{fraction}": test_metrics['r2_score'],
                f"scaling_val_rmse_L{L}_k{k}_frac_{fraction}": val_metrics['rmse'],
                f"scaling_test_rmse_L{L}_k{k}_frac_{fraction}": test_metrics['rmse'],
                "data_fraction": fraction,
                "L": L,
                "k": k
            }
            wandb.log(wandb_metrics)

        except Exception as e:
            logging.error(
                f"Error training with fraction {fraction} for L={L}, k={k}: {e}")
            results[fraction] = {
                'val_r2': np.nan,
                'val_mse': np.nan,
                'val_rmse': np.nan,
                'test_r2': np.nan,
                'test_mse': np.nan,
                'test_rmse': np.nan,
                'best_params': None,
                'L': L,
                'k': k,
                'error': str(e)
            }

    return results


def plot_scaling_curves(
    all_results: Dict[Tuple[int, int], Dict[float, Dict[str, float]]],
    output_dir: Path
):
    """
    Create scaling curve plots for all (L, k) pairs.
    """
    # Set up the plotting style
    plt.style.use('default')
    if HAS_SEABORN:
        sns.set_palette("husl")

    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Transport Operator Scaling Study', fontsize=16)

    metrics = ['val_r2', 'test_r2', 'val_rmse', 'test_rmse']
    metric_titles = ['Validation R²', 'Test R²',
                     'Validation RMSE', 'Test RMSE']

    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx // 2, idx % 2]

        for (L, k), results in all_results.items():
            fractions = sorted(results.keys())
            values = [results[frac][metric]
                      for frac in fractions if not np.isnan(results[frac][metric])]
            valid_fractions = [
                frac for frac in fractions if not np.isnan(results[frac][metric])]

            if values:  # Only plot if we have valid data
                ax.plot(valid_fractions, values, marker='o',
                        label=f'L={L}, k={k}', linewidth=2)

        ax.set_xlabel('Data Fraction')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set y-axis limits appropriately
        if 'r2' in metric:
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_curves.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scaling_curves.pdf', bbox_inches='tight')
    plt.close()

    # Create individual plots for each (L, k) pair
    for (L, k), results in all_results.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Scaling Curves for L={L}, k={k}', fontsize=14)

        fractions = sorted(results.keys())

        # R² plot
        val_r2 = [results[frac]['val_r2'] for frac in fractions]
        test_r2 = [results[frac]['test_r2'] for frac in fractions]

        axes[0].plot(fractions, val_r2, marker='o',
                     label='Validation', linewidth=2)
        axes[0].plot(fractions, test_r2, marker='s', label='Test', linewidth=2)
        axes[0].set_xlabel('Data Fraction')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('R² vs Data Amount')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # RMSE plot
        val_rmse = [results[frac]['val_rmse'] for frac in fractions]
        test_rmse = [results[frac]['test_rmse'] for frac in fractions]

        axes[1].plot(fractions, val_rmse, marker='o',
                     label='Validation', linewidth=2)
        axes[1].plot(fractions, test_rmse, marker='s',
                     label='Test', linewidth=2)
        axes[1].set_xlabel('Data Fraction')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('RMSE vs Data Amount')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(
            output_dir / f'scaling_L{L}_k{k}.png', dpi=300, bbox_inches='tight')
        plt.close()


def analyze_scaling_trends(
    all_results: Dict[Tuple[int, int], Dict[float, Dict[str, float]]],
    output_dir: Path
):
    """
    Analyze and summarize scaling trends.
    """
    summary = []

    for (L, k), results in all_results.items():
        fractions = sorted(results.keys())

        # Get final performance (highest data fraction)
        final_frac = max(fractions)
        final_val_r2 = results[final_frac]['val_r2']
        final_test_r2 = results[final_frac]['test_r2']

        # Calculate improvement from smallest to largest dataset
        min_frac = min(fractions)
        initial_val_r2 = results[min_frac]['val_r2']
        initial_test_r2 = results[min_frac]['test_r2']

        val_improvement = final_val_r2 - initial_val_r2
        test_improvement = final_test_r2 - initial_test_r2

        # Calculate average slope (simple linear fit)
        valid_fractions = [
            f for f in fractions if not np.isnan(results[f]['val_r2'])]
        if len(valid_fractions) > 1:
            val_r2_values = [results[f]['val_r2'] for f in valid_fractions]
            val_slope = np.polyfit(valid_fractions, val_r2_values, 1)[0]

            test_r2_values = [results[f]['test_r2'] for f in valid_fractions]
            test_slope = np.polyfit(valid_fractions, test_r2_values, 1)[0]
        else:
            val_slope = test_slope = np.nan

        summary.append({
            'L': L,
            'k': k,
            'final_val_r2': final_val_r2,
            'final_test_r2': final_test_r2,
            'val_improvement': val_improvement,
            'test_improvement': test_improvement,
            'val_slope': val_slope,
            'test_slope': test_slope,
            'data_points': len(valid_fractions)
        })

    # Save summary
    import pandas as pd
    df = pd.DataFrame(summary)
    df.to_csv(output_dir / 'scaling_summary.csv', index=False)

    # Print key insights
    logging.info("=== Scaling Study Summary ===")
    for row in summary:
        logging.info(f"L={row['L']}, k={row['k']}: "
                     f"Final R² = {row['final_test_r2']:.4f}, "
                     f"Improvement = {row['test_improvement']:.4f}, "
                     f"Slope = {row['test_slope']:.4f}")

    return df


@hydra.main(version_base=None, config_path="configs", config_name="scaling_study")
def main(cfg: DictConfig):
    load_dotenv()
    print(socket.gethostname())
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_seed(cfg.get("seed", 42))

    # Initialize wandb
    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=f"{cfg.experiment_name}_scaling_study",
        config=cast(dict[str, Any] | None,
                    OmegaConf.to_container(cfg, resolve=True)),
        mode=cfg.logger.wandb_mode,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Starting scaling study...")
    logging.info(cfg)

    # Create output directory for plots
    output_dir = Path("outputs/scaling_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define data fractions to test
    data_fractions = cfg.get('scaling_study', {}).get(
        'data_fractions', [0.1, 0.25, 0.5, 0.75, 1.0])
    logging.info(f"Testing data fractions: {data_fractions}")

    # Load activation data
    loader = ActivationLoader(
        files_to_download=[
            "activations-gemma2-2b-slimpajama-500k/activations_part_0000.zarr.zip"
        ]
    )

    all_results = {}

    # Run scaling study for each (L, k) pair
    for L in cfg.experiment.L:
        for k in cfg.experiment.k:
            logging.info(f"\n=== Running scaling study for L={L}, k={k} ===")

            # Create scaled datasets
            datasets = create_scaled_datasets(loader, L, k, data_fractions)

            # Run scaling experiment
            results = run_scaling_experiment(datasets, L, k, cfg)
            all_results[(L, k)] = results

            # Log key results
            logging.info(f"Results for L={L}, k={k}:")
            for fraction, metrics in results.items():
                if not np.isnan(metrics['val_r2']):
                    logging.info(
                        f"  {fraction*100:5.1f}% data: Val R² = {metrics['val_r2']:.4f}, Test R² = {metrics['test_r2']:.4f}")

    # Create visualizations
    logging.info("Creating scaling curve plots...")
    plot_scaling_curves(all_results, output_dir)

    # Analyze trends
    logging.info("Analyzing scaling trends...")
    summary_df = analyze_scaling_trends(all_results, output_dir)

    # Log final summary to wandb
    wandb.log({
        "scaling_study_complete": True,
        "num_layer_pairs": len(all_results),
        "data_fractions_tested": len(data_fractions)
    })

    logging.info("Scaling study complete!")
    logging.info(f"Results saved to: {output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
