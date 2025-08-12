import gc
import logging
import random
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import wandb
import zarr
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from src.sae_loader import pick_device


class ActivationCollector:
    """Efficient activation collector for transformer models."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = pick_device(cfg.model.device)
        self.model = None  # Will be loaded in load_model_and_tokenizer
        self.tokenizer: AutoTokenizer | None = (
            None  # Will be loaded in load_model_and_tokenizer
        )

        # Storage configuration
        # Add date to output directory for uniqueness
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d-%H%M%S")
        base_output_dir = cfg.activation_collection.output_dir
        self.output_dir = Path(f"{base_output_dir}_{date_str}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Processing configuration
        self.batch_size = cfg.activation_collection.batch_size
        self.max_length = cfg.activation_collection.max_length
        self.num_tokens_target = cfg.activation_collection.num_tokens
        self.save_every_n_batches = cfg.activation_collection.save_every_n_batches

        # DataLoader optimization settings
        self.num_workers = cfg.activation_collection.num_workers
        self.pin_memory = cfg.activation_collection.pin_memory
        self.prefetch_factor = cfg.activation_collection.prefetch_factor

        # Cache configuration
        self.cache_dir = Path(cfg.activation_collection.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Data type for storage (to save memory)
        self.storage_dtype = getattr(torch, cfg.activation_collection.storage_dtype)

    def load_model_and_tokenizer(self) -> None:
        """Load the transformer model and tokenizer."""
        model_name_or_path = self.cfg.model.name

        # Check if a local path is specified
        model_path = self.cfg.model.path
        if model_path is not None:
            model_name_or_path = model_path
            logging.info(f"Loading model from local path: {model_path}")
        else:
            logging.info(f"Loading model from HuggingFace Hub: {model_name_or_path}")

        # Prepare common loading arguments
        tokenizer_kwargs = {}
        model_kwargs = {
            "torch_dtype": getattr(torch, self.cfg.model.dtype),
            "trust_remote_code": True,
        }

        # Add cache_dir only if loading from HuggingFace Hub (not local path)
        if model_path is None:
            tokenizer_kwargs["cache_dir"] = str(self.cache_dir)
            model_kwargs["cache_dir"] = str(self.cache_dir)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )
        self.model.eval()

        # Move model to device
        self.model.to(self.device)

        logging.info(f"Model loaded on device: {self.model.device}")
        logging.info(f"Model has {self.model.config.num_hidden_layers} layers")

    def prepare_dataset(self):
        """Prepare the dataset for processing with batched tokenization."""
        if self.tokenizer is None:
            msg = "Tokenizer must be loaded before preparing dataset"
            raise ValueError(msg)

        dataset_name = self.cfg.datasets.dataset_name
        dataset_config = self.cfg.datasets.dataset_config
        split = self.cfg.datasets.dataset_split

        logging.info(
            f"Loading dataset: {dataset_name}/{dataset_config}, split: {split}",
        )

        # Load dataset with caching (not streaming)
        dataset = load_dataset(
            dataset_name, dataset_config, split=split, cache_dir=str(self.cache_dir)
        )
        dataset = dataset.shuffle(seed=self.cfg.seed)  # Shuffle for randomness
        dataset = dataset.select(
            range(1_000_000)  # Limit to 1 million samples for large datasets
        )

        def tokenize_function(examples):
            # Tokenize the text in batches
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # We'll pad in the DataLoader
                max_length=self.max_length,
                return_overflowing_tokens=False,
                return_length=True,
            )

        # Apply tokenization in batches
        logging.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,  # Large batch size for efficient tokenization
            remove_columns=dataset.column_names,  # Remove all original columns # type: ignore
        )

        # Filter out empty sequences
        tokenized_dataset = tokenized_dataset.filter(
            lambda example: len(example["input_ids"]) > 0
        )

        logging.info(f"Dataset tokenized. Total samples: {len(tokenized_dataset)}")

        # Create DataLoader with proper collation
        def collate_fn(batch):
            # Pad sequences to the same length within the batch
            input_ids = [torch.tensor(item["input_ids"]) for item in batch]

            # Pad sequences
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

            # Create attention mask
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            # Truncate to max_length if necessary
            if input_ids.size(1) > self.max_length:
                input_ids = input_ids[:, : self.max_length]
                attention_mask = attention_mask[:, : self.max_length]

            return {"input_ids": input_ids, "attention_mask": attention_mask}

        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Keep deterministic order
            collate_fn=collate_fn,
            num_workers=self.num_workers,  # Configurable number of workers
            pin_memory=self.pin_memory,  # Configurable memory pinning
            persistent_workers=True
            if self.num_workers > 0
            else False,  # Keep workers alive
            prefetch_factor=self.prefetch_factor
            if self.num_workers > 0
            else None,  # Configurable prefetch
        )

        return dataloader

    def collect_batch_activations(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Collect activations for a single batch using output_hidden_states."""
        if self.model is None:
            raise ValueError("Model must be loaded before collecting activations")

        # Use non_blocking=True for faster GPU transfer when pin_memory=True
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

        # Forward pass with output_hidden_states=True to get all layer outputs
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract hidden states (residual stream at end of each layer)
        hidden_states = (
            outputs.hidden_states
        )  # Tuple of (batch_size, seq_len, hidden_size)

        # Process activations
        batch_activations = {}
        for layer_idx, layer_hidden_states in enumerate(hidden_states):
            # Convert to storage dtype and move to CPU to save GPU memory
            layer_activations = layer_hidden_states.to(self.storage_dtype).cpu()

            # Only keep activations for non-padded tokens
            masked_activation = layer_activations * attention_mask.cpu().unsqueeze(
                -1
            ).to(layer_activations.dtype)
            batch_activations[f"layer_{layer_idx}"] = masked_activation

        # Add input metadata
        batch_activations["input_ids"] = input_ids.cpu()
        batch_activations["attention_mask"] = attention_mask.cpu()

        return batch_activations

    def wait_for_pending_saves(self):
        """Wait for all pending background saves to complete."""
        for future in self.pending_saves:
            try:
                future.result()  # This will block until the save is complete
            except Exception as e:
                logging.error(f"Error in background save: {e}")
        self.pending_saves.clear()

    def save_activations_to_zarr(
        self, activations_list: list[dict[str, torch.Tensor]], filename: str
    ):
        """Save collected activations to Zarr store."""
        filepath = self.output_dir / filename

        # Get layer names from first batch
        if not activations_list:
            return

        layer_names = [k for k in activations_list[0] if k.startswith("layer_")]

        # Find the maximum sequence length across all batches
        max_seq_len = 0
        for batch_activations in activations_list:
            seq_len = batch_activations["input_ids"].size(1)
            max_seq_len = max(max_seq_len, seq_len)

        # Concatenate all activations with proper padding
        all_input_ids = []
        all_attention_masks = []
        all_layer_activations = {layer: [] for layer in layer_names}

        for batch_activations in activations_list:
            # Get current batch info
            input_ids = batch_activations["input_ids"]
            attention_mask = batch_activations["attention_mask"]
            current_seq_len = input_ids.size(1)

            # Pad input_ids and attention_mask to max_seq_len if necessary
            if current_seq_len < max_seq_len:
                pad_length = max_seq_len - current_seq_len
                # Pad input_ids with tokenizer pad_token_id
                input_ids = torch.nn.functional.pad(
                    input_ids, (0, pad_length), value=self.tokenizer.pad_token_id
                )
                # Pad attention_mask with 0 (masked tokens)
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (0, pad_length), value=0
                )

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

            # Pad layer activations
            for layer in layer_names:
                layer_activations = batch_activations[layer]
                if layer_activations.size(1) < max_seq_len:
                    pad_length = max_seq_len - layer_activations.size(1)
                    # Pad with zeros for activations
                    layer_activations = torch.nn.functional.pad(
                        layer_activations, (0, 0, 0, pad_length), value=0.0
                    )
                all_layer_activations[layer].append(layer_activations)

        # Concatenate tensors (now they all have the same sequence length)
        input_ids_concat = torch.cat(all_input_ids, dim=0)
        attention_mask_concat = torch.cat(all_attention_masks, dim=0)

        # Create Zarr store
        store = zarr.open(str(filepath), mode="w")

        # Convert tensors to numpy arrays
        input_ids_data = input_ids_concat.numpy()
        attention_mask_data = attention_mask_concat.numpy()

        # Save metadata with automatic chunking
        compressors = zarr.codecs.BloscCodec(
            cname="zstd", clevel=9, shuffle=zarr.codecs.BloscShuffle.bitshuffle
        )
        store.create_array(
            "input_ids", data=input_ids_data, chunks="auto", compressors=compressors
        )
        store.create_array(
            "attention_mask",
            data=attention_mask_data,
            chunks="auto",
            compressors=compressors,
        )

        # Save layer activations
        hidden_size = None
        activations_group = store.create_group("activations")
        for layer_name in layer_names:
            layer_activations = torch.cat(all_layer_activations[layer_name], dim=0)
            layer_data = layer_activations.numpy()

            if hidden_size is None:
                hidden_size = layer_data.shape[-1]

            activations_group.create_array(
                layer_name,
                data=layer_data,
                chunks="auto",
                compressors=compressors,
            )

        # Save configuration as attributes
        store.attrs["model_name"] = self.cfg.model.name
        store.attrs["batch_size"] = self.batch_size
        store.attrs["max_length"] = self.max_length
        store.attrs["num_layers"] = len(layer_names)
        store.attrs["hidden_size"] = hidden_size or 0
        store.attrs["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        store.attrs["max_seq_len"] = max_seq_len

        logging.info(f"Saved activations to {filepath} (max_seq_len: {max_seq_len})")

    def collect_activations(self):
        """Main method to collect activations from the dataset."""
        logging.info("Starting activation collection...")

        # Load model
        self.load_model_and_tokenizer()

        # Prepare dataset
        dataloader = self.prepare_dataset()

        # Collection loop
        activations_buffer = []
        total_tokens_processed = 0
        batch_count = 0
        file_count = 0

        try:
            with tqdm(desc="Processing batches", unit="batch") as pbar:
                for batch in dataloader:
                    # Check if we've reached our target
                    if total_tokens_processed >= self.num_tokens_target:
                        logging.info(
                            f"Reached target of {self.num_tokens_target} tokens",
                        )
                        break

                    # Collect activations for this batch
                    batch_activations = self.collect_batch_activations(batch)
                    activations_buffer.append(batch_activations)

                    # Count tokens (excluding padding)
                    batch_tokens = batch["attention_mask"].sum().item()
                    total_tokens_processed += batch_tokens
                    batch_count += 1

                    # Update progress
                    pbar.set_postfix(
                        {
                            "tokens": f"{total_tokens_processed:,}",
                            "target": f"{self.num_tokens_target:,}",
                            "progress": f"{100 * total_tokens_processed / self.num_tokens_target:.1f}%",
                        }
                    )
                    pbar.update(1)

                    # Save periodically to avoid memory issues
                    if batch_count % self.save_every_n_batches == 0:
                        filename = f"activations_part_{file_count:04d}.zarr"
                        self.save_activations_to_zarr(activations_buffer, filename)
                        activations_buffer = []
                        file_count += 1

                        # Log memory usage
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated() / 1024**3
                            logging.info(f"GPU memory used: {memory_used:.2f} GB")

                        # Force garbage collection
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # Save remaining activations
            if activations_buffer:
                filename = f"activations_part_{file_count:04d}.zarr"
                self.save_activations_to_zarr(activations_buffer, filename)

            # Wait for all background saves to complete
            logging.info("Waiting for all background saves to complete...")
            self.wait_for_pending_saves()

            logging.info("Activation collection complete!")
            logging.info(f"Total tokens processed: {total_tokens_processed:,}")
            logging.info(f"Files saved: {file_count + 1}")

        except Exception as e:
            logging.error(f"Error during activation collection: {e}")

        return total_tokens_processed


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="activation_collection",
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    print(socket.gethostname())
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_seed(cfg.seed)

    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.experiment_name,
        config=cast("dict[str, Any] | None", OmegaConf.to_container(cfg, resolve=True)),
        mode=cfg.logger.wandb_mode,  # NOTE: disabled by default
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Loaded configuration:")
    logging.info(cfg)

    # Create activation collector and run
    collector = ActivationCollector(cfg)
    total_tokens = collector.collect_activations()

    # Log results to wandb
    wandb.log(
        {
            "total_tokens_processed": total_tokens,
            "target_tokens": cfg.activation_collection.num_tokens,
            "completion_rate": total_tokens / cfg.activation_collection.num_tokens,
        }
    )


if __name__ == "__main__":
    main()
