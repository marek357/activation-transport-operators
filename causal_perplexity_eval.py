"""Perplexity evaluation script for transport operator interventions on language models."""

import json
import logging
import random
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import (
    set_seed,
)

from src.transport_operator import load_transport_operator
from src.causal_eval.hooks import (
    create_j_hook_family,
    create_zero_hook_family,
)
from src.causal_eval.perplexity_evaluator import PerplexityEvaluator, PerplexityResult

logger = logging.getLogger(__name__)


def perplexity_result_to_dict(result: PerplexityResult) -> Dict:
    """Convert PerplexityResult to a JSON-serializable dictionary."""
    result_dict = {
        "log_perplexity": float(result.log_perplexity),
        "loss": float(result.loss),
        "total_tokens": int(result.total_tokens),
        "num_sequences": int(result.num_sequences),
        "model_name": result.model_name,
        "modification": result.modification,
    }

    # Add per-sequence log perplexities if available
    if result.per_sequence_log_ppls is not None:
        result_dict["per_sequence_log_ppls"] = [
            float(x) for x in result.per_sequence_log_ppls
        ]

    return result_dict


def save_results_to_json(
    results: Dict[str, PerplexityResult],
    cfg: DictConfig,
    L: int,
    k: int,
    js: list,
) -> None:
    """Save evaluation results to a JSON file."""
    # Create causal_results directory if it doesn't exist
    results_dir = Path("causal_results")
    results_dir.mkdir(exist_ok=True)

    # Create filename with L, k, and config name
    filename = f"perplexity_eval_L{L}_k{k}_{len(js[0])}.json"
    filepath = results_dir / filename

    # Convert results to JSON-serializable format
    json_results = {}
    for name, result in results.items():
        json_results[name] = perplexity_result_to_dict(result)

    # Add metadata
    json_data = {
        "metadata": {
            "L": L,
            "k": k,
            "js": OmegaConf.to_container(js, resolve=True),
            "experiment_name": cfg.get("experiment_name", "unknown"),
            "model_name": cfg.model.get("name", "unknown"),
        },
        "results": json_results,
    }

    # Save to JSON file
    with open(filepath, "w") as f:
        json.dump(json_data, f, indent=2)

    # Log summary of per-sequence data
    total_per_seq_values = sum(
        len(result_data.get("per_sequence_log_ppls", []))
        for result_data in json_results.values()
    )
    logger.info(f"Results saved to {filepath}")
    logger.info(
        f"Total per-sequence log perplexity values saved: {total_per_seq_values}"
    )


def print_evaluation_results(results: Dict[str, PerplexityResult]) -> None:
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)

    # Print header
    print(
        f"{'Method':<40} {'Log PPL':<10} {'Change':<10} {'Sequences':<10} {'Tokens':<12} {'Avg Len':<8}"
    )
    print("-" * 100)

    baseline_log_ppl = results["baseline"].log_perplexity

    for name, result in results.items():
        log_ppl_change = (
            0.0 if name == "baseline" else result.log_perplexity - baseline_log_ppl
        )

        avg_seq_length = (
            result.total_tokens / result.num_sequences
            if result.num_sequences > 0
            else 0
        )

        # Format change with appropriate sign and color coding intent
        change_str = "   -   " if name == "baseline" else f"{log_ppl_change:+7.4f}"

        print(
            f"{name:<40} {result.log_perplexity:8.4f}   {change_str:<10} "
            f"{result.num_sequences:>8,}   {result.total_tokens:>10,}   {avg_seq_length:6.1f}"
        )

    print("=" * 100)


@hydra.main(version_base=None, config_path="configs", config_name="causal_eval")
def main(cfg: DictConfig):
    """Main evaluation function."""
    load_dotenv()

    # Set seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_seed(cfg.seed)

    # Initialize wandb
    wandb.init(
        project=cfg.logger.get("project", "perplexity-eval"),
        entity=cfg.logger.get("entity", None),
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logger.get("wandb_mode", "disabled"),
    )

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting perplexity evaluation")

    # Create evaluator
    evaluator = PerplexityEvaluator(cfg)

    # Define the two hooks to test
    # Hook 1: Transport intervention (capture from source layer, transport, substitute into target layer)
    L = cfg.causal_eval.get("L", 0)
    k = cfg.causal_eval.get("k", 1)
    js = cfg.causal_eval.get("js", [[1], [50, 100]])

    logger.info(f"Configuration: L={L}, k={k}, js={js}")

    transport_operator = load_transport_operator(
        L=L,
        k=k,
        operators_dir=cfg.causal_eval.get("cache_dir", "./cache"),
    )

    intervention_hooks = create_j_hook_family(
        transport_operator,
        source_layer=f"model.layers.{L}",
        target_layer=f"model.layers.{L + k}",
        js=js,
        prefix="intervention",
    )
    ablation_hooks = create_j_hook_family(
        transport_operator,
        source_layer=None,
        target_layer=f"model.layers.{L + k}",
        js=js,
        prefix="ablation",
    )

    # Create zero hooks for comparison
    zero_hooks = create_zero_hook_family(
        layer_name=f"model.layers.{L + k}",
        js=js,
        prefix="zero",
    )

    # Run baseline evaluation
    baseline_result = evaluator.evaluate_baseline()

    # Run evaluations with hooks
    intervention_results = evaluator.evaluate_with_hooks(intervention_hooks)
    ablation_results = evaluator.evaluate_with_hooks(ablation_hooks)
    zero_results = evaluator.evaluate_with_hooks(zero_hooks)

    # Calculate averaged results
    def average_results(
        results_dict: Dict[str, PerplexityResult], prefix: str
    ) -> PerplexityResult:
        """Calculate averaged results for a group of hooks with the same prefix."""
        filtered_results = [
            result for name, result in results_dict.items() if name.startswith(prefix)
        ]

        if not filtered_results:
            raise ValueError(f"No results found with prefix '{prefix}'")

        # Combine all per-sequence log perplexities
        all_per_sequence_log_ppls = []
        for result in filtered_results:
            if result.per_sequence_log_ppls is not None:
                all_per_sequence_log_ppls.append(result.per_sequence_log_ppls)

        arr = np.array(all_per_sequence_log_ppls)
        all_per_sequence_log_ppls = arr.mean(axis=0).tolist()

        avg_loss = sum(r.loss for r in filtered_results) / len(filtered_results)
        avg_log_perplexity = avg_loss
        total_tokens = sum(r.total_tokens for r in filtered_results)
        total_sequences = sum(r.num_sequences for r in filtered_results)

        return PerplexityResult(
            log_perplexity=avg_log_perplexity,
            loss=avg_loss,
            total_tokens=total_tokens,
            num_sequences=total_sequences,
            model_name=filtered_results[0].model_name,
            modification=f"avg_{prefix}",
            per_sequence_log_ppls=all_per_sequence_log_ppls
            if all_per_sequence_log_ppls
            else None,
        )

    # Create averaged results
    avg_intervention_result = average_results(intervention_results, "intervention")
    avg_ablation_result = average_results(ablation_results, "ablation")
    avg_zero_result = average_results(zero_results, "zero")

    # Combine all results including individual and averaged
    hook_results = {}
    hook_results.update(intervention_results)
    hook_results.update(ablation_results)
    hook_results.update(zero_results)
    hook_results["avg_intervention"] = avg_intervention_result
    hook_results["avg_ablation"] = avg_ablation_result
    hook_results["avg_zero"] = avg_zero_result

    # Combine all results
    results = {"baseline": baseline_result}
    results.update(hook_results)

    # Print and log results
    print_evaluation_results(results)

    # Save results to JSON file
    save_results_to_json(results, cfg, L, k, js)

    logger.info(
        "Evaluation completed! Per-sequence log perplexities have been collected and saved for statistical analysis."
    )


if __name__ == "__main__":
    main()
