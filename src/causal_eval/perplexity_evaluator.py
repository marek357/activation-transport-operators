"""
Perplexity evaluator for HuggingFace models.
Computes log perplexity.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset

from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

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
    per_sequence_log_ppls: List[float] = (
        None  # Log perplexity for each individual sequence (for statistical analysis)
    )


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

        logger.info(f"Model {model_name} loaded successfully with dtype {dtype}.")
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
        if (
            self.cfg.causal_eval.get("skip_initial_n_samples_from_dataset", None)
            is not None
        ):
            dataset = dataset.skip(
                self.cfg.causal_eval.skip_initial_n_samples_from_dataset
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
        max_samples = self.cfg.causal_eval.get("max_samples", 1000)
        max_length = self.cfg.causal_eval.get("max_length", 1024)

        total_loss = 0.0
        total_tokens = 0
        sample_count = 0
        total_samples = 0
        per_sequence_log_ppls = []  # Store per-sequence log perplexities

        logger.info(f"Evaluating {modification_name}...")

        with tqdm(desc="Eval", unit="samples") as pbar:
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

                # We only want to process full sequences
                if input_ids.size(1) < max_length:
                    continue

                try:
                    sequence_loss, sequence_tokens = (
                        self._compute_log_perplexity_single(input_ids, attention_mask)
                    )

                    if sequence_tokens > 0:  # Only add if we have valid tokens
                        # Calculate per-sequence log perplexity
                        sequence_log_ppl = sequence_loss / sequence_tokens
                        per_sequence_log_ppls.append(sequence_log_ppl)

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

        logger.info(
            f"Evaluation complete for {modification_name}: "
            f"log_perplexity={log_perplexity:.4f}, "
            f"total_tokens={total_tokens}, "
            f"num_sequences={sample_count}, "
            f"per_sequence_log_ppls_collected={len(per_sequence_log_ppls)}"
        )

        return PerplexityResult(
            log_perplexity=log_perplexity,
            loss=avg_loss,
            total_tokens=total_tokens,
            num_sequences=sample_count,
            model_name=self.cfg.model.name,
            modification=modification_name,
            per_sequence_log_ppls=per_sequence_log_ppls,
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
