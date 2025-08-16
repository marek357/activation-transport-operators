"""
Perplexity evaluator for HuggingFace models.
Computes log perplexity.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict

import hydra
import numpy as np
import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    set_seed,
)
from src.activation_loader import ActivationDataset

from src.transport_operator import TransportOperator
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerplexityResult:
    """Container for perplexity evaluation results."""

    log_perplexity: float
    loss: float
    total_tokens: int
    num_sequences: int
    model_name: str
    modification: str


class PerplexityEvaluator:
    """Perplexity evaluator."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = self._get_device()
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _get_device(self):
        """Get computation device."""
        device_cfg = self.cfg.model.get("device", "auto")
        if device_cfg == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_cfg)

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        model_name = self.cfg.model.name
        model_path = self.cfg.model.get("path", model_name)

        dtype = getattr(torch, self.cfg.model.get("dtype", "float16"))

        logger.info(f"Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Model {model_name} loaded successfully with dtype {dtype}.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
        )

        if self.device.type != "cuda":
            model = model.to(self.device)

        model.eval()
        return model, tokenizer

    def _load_dataset(self):
        """Load dataset with streaming."""
        dataset_cfg = self.cfg.datasets
        dataset = load_dataset(
            dataset_cfg.dataset_name,
            dataset_cfg.get("dataset_config", None),
            split=dataset_cfg.get("dataset_split", "test"),
            streaming=True,
        )
        return dataset

    def _compute_log_perplexity_single(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        """Compute log perplexity for a single sequence."""
        with torch.no_grad():
            # Create labels with -100 for padding tokens (ignored in loss computation)
            labels = input_ids.clone()
            # Set padding tokens to -100 so they're ignored in loss calculation
            labels[attention_mask == 0] = -100

            # Use model's built-in loss computation
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            # Get the loss (already averaged over valid tokens)
            sequence_loss = outputs.loss.item()

            # Count the number of valid tokens for this sequence
            # Subtract 1 from sequence length since we predict next tokens
            valid_tokens = (attention_mask.sum() - 1).clamp(min=0).item()

            # Convert back to total loss (loss * num_tokens)
            total_loss = sequence_loss * valid_tokens

            return total_loss, valid_tokens

    def _evaluate_single(self, modification_name: str):
        """Evaluate with current model state."""
        dataset = self._load_dataset()
        text_column = self.cfg.datasets.get("text_column_name", "text")
        max_samples = self.cfg.eval.get("max_samples", 1000)
        max_length = self.cfg.eval.get("max_length", 1024)

        total_loss = 0.0
        total_tokens = 0
        sample_count = 0
        total_samples = 0

        logger.info(f"Evaluating {modification_name}...")

        with tqdm(desc=f"Eval {modification_name}", unit="samples") as pbar:
            for i, sample in enumerate(dataset):
                if total_samples >= max_samples:
                    break

                text = sample[text_column]
                if not text or not text.strip():
                    continue

                # Process single sequence
                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                )

                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)

                # Check if sequence is long enough for the required positions
                max_j = max([max(j) for j in self.cfg.eval.js])
                if input_ids.size(1) <= max_j:
                    # logging.warning(
                    #     f"Input sequence is shorter than required: {input_ids.size(1)} <= {max_j}. Skipping."
                    # )
                    continue

                try:
                    sequence_loss, sequence_tokens = (
                        self._compute_log_perplexity_single(input_ids, attention_mask)
                    )

                    if sequence_tokens > 0:  # Only add if we have valid tokens
                        total_loss += sequence_loss
                        total_tokens += sequence_tokens
                        sample_count += 1

                        current_log_ppl = total_loss / total_tokens
                        pbar.set_postfix({"log_ppl": f"{current_log_ppl:.4f}"})

                except Exception as e:
                    logger.warning(f"Error processing sequence: {e}")

                total_samples += 1
                pbar.update(1)

        if total_tokens == 0:
            raise ValueError("No valid tokens found")

        avg_loss = total_loss / total_tokens
        log_perplexity = avg_loss

        return PerplexityResult(
            log_perplexity=log_perplexity,
            loss=avg_loss,
            total_tokens=total_tokens,
            num_sequences=sample_count,
            model_name=self.cfg.model.name,
            modification=modification_name,
        )

    def evaluate_baseline(self) -> PerplexityResult:
        """Evaluate baseline model without any hooks."""
        logger.info("=== Baseline Evaluation ===")
        return self._evaluate_single("baseline")

    def evaluate_with_hooks(
        self, hooks: Dict[str, object]
    ) -> Dict[str, PerplexityResult]:
        """Evaluate model with multiple hooks applied sequentially.

        Args:
            hooks: Dictionary mapping hook names to hook objects with apply() and remove() methods

        Returns:
            Dictionary mapping hook names to PerplexityResult objects
        """
        results = {}

        for hook_name, hook in hooks.items():
            logger.info(f"=== {hook_name} Evaluation ===")
            hook.apply(self.model)
            try:
                results[hook_name] = self._evaluate_single(hook_name)
            finally:
                hook.remove()

        return results


class TransportHook:
    """Hook for transport operator interventions that captures from source and modifies target layer."""

    def __init__(
        self,
        name: str,
        source_layer: str,
        target_layer: str,
        transport_operator: TransportOperator,
        target_j_positions: list[int],
    ):
        self.name = name
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.transport_operator = transport_operator
        self.captured_activation = None
        self.source_hook_handle = None
        self.target_hook_handle = None
        self.target_j_positions = target_j_positions

    def apply(self, model: PreTrainedModel):
        """Apply both source capture and target modification hooks."""
        if self.source_layer is not None:
            # Hook for capturing from source layer
            source_module = model
            for attr in self.source_layer.split("."):
                source_module = getattr(source_module, attr)

            def capture_hook(module, input_tensors, output):
                # Store the hidden states for transport operator
                if isinstance(output, torch.Tensor):
                    self.captured_activation = output[
                        :, self.target_j_positions, :
                    ].detach()
                elif isinstance(output, tuple):
                    # For layers that return tuples (like attention), take the first element
                    self.captured_activation = output[0][
                        :, self.target_j_positions, :
                    ].detach()
                return output

            self.source_hook_handle = source_module.register_forward_hook(capture_hook)

        # Hook for modifying target layer
        target_module = model
        for attr in self.target_layer.split("."):
            target_module = getattr(target_module, attr)

        def transport_hook(module, input_tensors, output):
            if self.source_layer is None:
                # Use null vector as input to transport operator
                if isinstance(output, torch.Tensor):
                    batch_size, _, _ = output.shape
                    # Get input features from transport matrix shape
                    input_features = (
                        self.transport_operator.get_transport_matrix().shape[0]
                    )
                    null_input = torch.zeros(
                        batch_size,
                        len(self.target_j_positions),
                        input_features,
                        dtype=output.dtype,
                        device=output.device,
                    )
                else:
                    batch_size, seq_len, hidden_size = output[0].shape

                    input_features = (
                        self.transport_operator.get_transport_matrix().shape[0]
                    )
                    null_input = torch.zeros(
                        batch_size,
                        input_features,
                        len(self.target_j_positions),
                        dtype=output[0].dtype,
                        device=output[0].device,
                    )

                transport_input = null_input
            else:
                # Use captured activation from source layer
                if self.captured_activation is None:
                    raise RuntimeError("No captured activation available for transport")
                transport_input = self.captured_activation

            try:
                # Apply transport operator
                original_shape = transport_input.shape
                # Reshape to (batch_size * seq_len, hidden_size) for transport operator
                flat_input = transport_input.view(-1, transport_input.shape[-1])

                # Convert to numpy for transport operator
                flat_input_np = flat_input.float().cpu().numpy()

                # Apply transport operator
                transported_np = self.transport_operator.predict(flat_input_np)

                # Convert back to torch and reshape
                transported = torch.from_numpy(transported_np.astype(np.float32))
                transported = transported.to(transport_input.device)
                transported = transported.to(transport_input.dtype)
                transported = transported.view(
                    original_shape[0], len(self.target_j_positions), -1
                )

                # Replace the output
                if isinstance(output, torch.Tensor):
                    for j, j_position in enumerate(self.target_j_positions):
                        output[:, j_position, :] = transported[:, j, :]
                    return output
                elif isinstance(output, tuple):
                    modified_output = list(output)
                    modified_output[0] = output[0]
                    for j, j_position in enumerate(self.target_j_positions):
                        modified_output[0][:, j_position, :] = transported[:, j, :]
                    return tuple(modified_output)
                else:
                    raise RuntimeError("Unsupported output type")

            except Exception as e:
                logger.exception(
                    f"Transport operation failed: {e}, returning original output"
                )

        self.target_hook_handle = target_module.register_forward_hook(transport_hook)
        logger.info(
            f"Applied transport hook '{self.name}': {self.source_layer} -> {self.target_layer}"
        )

    def remove(self):
        """Remove both hooks."""
        if self.source_hook_handle:
            self.source_hook_handle.remove()
            self.source_hook_handle = None
        if self.target_hook_handle:
            self.target_hook_handle.remove()
            self.target_hook_handle = None
        self.captured_activation = None


def load_transport_operator(
    L: int,
    k: int,
    operators_dir: str,
) -> TransportOperator:
    """Load the transport operator from the cache or create a new one."""
    operator = TransportOperator(
        L,
        k,
        regularization=10.0,
        max_iter=500,
    )
    dummy_ds = ActivationDataset(None, [], "", 0, 0)
    file_name = operator._get_model_cache_filename(dummy_ds)
    is_loaded = operator._load_model_cache(Path(operators_dir) / file_name)
    if not is_loaded:
        raise FileNotFoundError(f"Transport operator model not found: {file_name}")
    return operator


def create_j_hook_family(
    transport_operator,
    source_layer: str,
    target_layer: str,
    js: list[list[int]],
    prefix: str,
) -> dict[int, TransportHook]:
    """Create a family of transport operator hooks for a specific layer."""
    hooks = {}
    for j in js:
        hooks[f"{prefix}_{str(j)}"] = create_transport_hook(
            transport_operator,
            source_layer=source_layer,
            target_layer=target_layer,
            j_position=j,
        )
    return hooks


def create_transport_hook(
    transport_operator, source_layer: str, target_layer: str, j_position: int
) -> TransportHook:
    """Create a transport operator hook that captures from source and transports to target."""
    return TransportHook(
        "transport_intervention",
        source_layer,
        target_layer,
        transport_operator,
        j_position,
    )


@hydra.main(
    version_base=None, config_path="configs", config_name="simple_perplexity_eval"
)
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

    logger.info("Starting simplified perplexity evaluation")

    # Create evaluator
    evaluator = PerplexityEvaluator(cfg)

    # Define the two hooks to test
    # Hook 1: Transport intervention (capture from source layer, transport, substitute into target layer)
    L = cfg.eval.get("L", 0)
    k = cfg.eval.get("k", 1)
    js = cfg.eval.get("js", [[1], [50, 100]])

    transport_operator = load_transport_operator(
        L=L,
        k=k,
        operators_dir=cfg.eval.get("cache_dir", "./cache"),
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
        source_layer=f"model.layers.{L + k}",
        target_layer=f"model.layers.{L + k}",
        js=js,
        prefix="ablation",
    )

    # Run baseline evaluation
    baseline_result = evaluator.evaluate_baseline()

    # Run evaluations with hooks
    intervention_results = evaluator.evaluate_with_hooks(intervention_hooks)
    ablation_results = evaluator.evaluate_with_hooks(ablation_hooks)

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

        avg_log_perplexity = sum(r.log_perplexity for r in filtered_results) / len(
            filtered_results
        )
        avg_loss = sum(r.loss for r in filtered_results) / len(filtered_results)
        total_tokens = sum(r.total_tokens for r in filtered_results)
        total_sequences = sum(r.num_sequences for r in filtered_results)

        return PerplexityResult(
            log_perplexity=avg_log_perplexity,
            loss=avg_loss,
            total_tokens=total_tokens,
            num_sequences=total_sequences,
            model_name=filtered_results[0].model_name,
            modification=f"avg_{prefix}",
        )

    # Create averaged results
    avg_intervention_result = average_results(intervention_results, "intervention")
    avg_ablation_result = average_results(ablation_results, "ablation")

    # Combine all results including individual and averaged
    hook_results = {}
    hook_results.update(intervention_results)
    hook_results.update(ablation_results)
    hook_results["avg_intervention"] = avg_intervention_result
    hook_results["avg_ablation"] = avg_ablation_result

    # Combine all results
    results = {"baseline": baseline_result}
    results.update(hook_results)

    # Print and log results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    # Print header
    logger.info(
        f"{'Method':<20} {'Log PPL':<10} {'Change':<10} {'Sequences':<10} {'Tokens':<12} {'Avg Len':<8}"
    )
    logger.info("-" * 80)

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

        logger.info(
            f"{name:<20} {result.log_perplexity:8.4f}   {change_str:<10} "
            f"{result.num_sequences:>8,}   {result.total_tokens:>10,}   {avg_seq_length:6.1f}"
        )

    logger.info("=" * 80)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
