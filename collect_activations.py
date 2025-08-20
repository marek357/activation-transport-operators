import gc
import logging
import random
import socket
import time
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


def filter_empty_sequences(example):
    """Filter function to remove empty sequences - needed for streaming datasets."""
    return len(example["input_ids"]) > 0


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

    def collate_fn(self, batch):
        """Collate function for DataLoader - simple version for streaming datasets."""
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

    def tokenize_function(self, examples):
        """Tokenize function for dataset mapping - defined as class method to avoid pickling issues with streaming."""
        # Tokenize the text in batches
        return self.tokenizer(
            examples[self.cfg.datasets.text_column_name],
            truncation=True,
            padding=False,  # We'll pad in the DataLoader
            max_length=self.max_length,
            return_overflowing_tokens=False,
            return_length=True,
        )

    def prepare_dataset(self):
        """Prepare the dataset for processing with batched tokenization."""
        if self.tokenizer is None:
            msg = "Tokenizer must be loaded before preparing dataset"
            raise ValueError(msg)

        dataset_name = self.cfg.datasets.dataset_name
        dataset_config = self.cfg.datasets.dataset_config
        dataset_data_dir = self.cfg.datasets.dataset_data_dir
        split = self.cfg.datasets.dataset_split

        logging.info(
            f"Loading dataset: {dataset_name}/{dataset_config} from {dataset_data_dir}, split: {split}",
        )

        # Load dataset with caching (not streaming)
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            data_dir=dataset_data_dir,
            split=split,
            cache_dir=str(self.cache_dir),
            streaming=True,
        )

        # Print a couple of examples from the dataset
        logging.info("Printing a few examples from the dataset...")
        example_count = 0
        for example in dataset:
            if example_count >= 3:  # Show 3 examples
                break
            text = example[self.cfg.datasets.text_column_name]
            logging.info(
                f"Example {example_count + 1}: {text[:200]}..."
            )  # Show first 200 chars
            example_count += 1

        # Apply tokenization in batches
        logging.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            remove_columns=dataset.column_names,  # Remove all original columns # type: ignore
        )

        # Filter out empty sequences
        tokenized_dataset = tokenized_dataset.filter(filter_empty_sequences)

        # logging.info(f"Dataset tokenized.")

        # Create DataLoader with proper collation
        # For streaming datasets, we need to disable multiprocessing to avoid tensor sharing issues
        # This is the most reliable solution for streaming + GPU tensors
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Keep deterministic order
            collate_fn=self.collate_fn,
            num_workers=0,  # Must be 0 for streaming datasets to avoid _share_filename_ error
            pin_memory=False,  # Set to False when num_workers=0
            persistent_workers=False,  # Not applicable when num_workers=0
            prefetch_factor=None,  # Not applicable when num_workers=0
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
            if layer_idx == 0:
                # Input layer, no need to store embeddings
                continue
            # Convert to storage dtype and move to CPU to save GPU memory
            layer_activations = layer_hidden_states.to(self.storage_dtype).cpu()

            # Only keep activations for non-padded tokens
            masked_activation = layer_activations * attention_mask.cpu().unsqueeze(
                -1
            ).to(layer_activations.dtype)
            batch_activations[f"layer_{layer_idx - 1}"] = masked_activation

        # Add input metadata
        batch_activations["input_ids"] = input_ids.cpu()
        batch_activations["attention_mask"] = attention_mask.cpu()

        return batch_activations

    def save_activations_to_zarr(
        self, activations_list: list[dict[str, torch.Tensor]], filename: str
    ):
        """Save collected activations to Zarr store with performance optimizations."""
        start_time = time.time()
        filepath = self.output_dir / filename

        # Get layer names from first batch
        if not activations_list:
            return

        layer_names = [k for k in activations_list[0] if k.startswith("layer_")]

        # Calculate dimensions efficiently in one pass
        max_seq_len = 0
        total_samples = 0
        for batch_activations in activations_list:
            seq_len = batch_activations["input_ids"].size(1)
            max_seq_len = max(max_seq_len, seq_len)
            total_samples += batch_activations["input_ids"].size(0)

        # Get hidden size from first batch
        hidden_size = activations_list[0][layer_names[0]].size(-1)

        # Create Zarr store
        store = zarr.storage.ZipStore(str(filepath), mode="w")

        # Keep existing compression settings
        compressors = zarr.codecs.BloscCodec(
            cname="zstd", clevel=4, shuffle=zarr.codecs.BloscShuffle.bitshuffle
        )

        # Use optimal chunking for better I/O performance
        # Chunk size optimized for compression and access patterns
        chunk_samples = min(500, total_samples)  # Reasonable chunk size for samples
        chunk_seq = min(max_seq_len, 128)  # Reasonable chunk size for sequence length

        # Pre-allocate numpy arrays to avoid multiple tensor concatenations
        input_ids_data = np.full(
            (total_samples, max_seq_len), self.tokenizer.pad_token_id, dtype=np.int32
        )
        attention_mask_data = np.zeros((total_samples, max_seq_len), dtype=np.int32)

        # Pre-allocate layer activation arrays
        layer_data_arrays = {}
        for layer_name in layer_names:
            layer_data_arrays[layer_name] = np.zeros(
                (total_samples, max_seq_len, hidden_size),
                dtype=np.float16 if self.storage_dtype == torch.float16 else np.float32,
            )

        # Fill arrays efficiently without intermediate padding
        sample_offset = 0
        for batch_activations in activations_list:
            batch_size = batch_activations["input_ids"].size(0)
            current_seq_len = batch_activations["input_ids"].size(1)

            # Copy data directly to pre-allocated arrays (numpy handles the rest)
            input_ids_data[
                sample_offset : sample_offset + batch_size, :current_seq_len
            ] = batch_activations["input_ids"].numpy()
            attention_mask_data[
                sample_offset : sample_offset + batch_size, :current_seq_len
            ] = batch_activations["attention_mask"].numpy()

            # Copy layer activations
            for layer_name in layer_names:
                layer_tensor = batch_activations[layer_name]
                layer_data_arrays[layer_name][
                    sample_offset : sample_offset + batch_size, :current_seq_len
                ] = layer_tensor.numpy()

            sample_offset += batch_size

        # Save arrays to Zarr with optimal chunking
        zarr.create_array(
            store=store,
            name="input_ids",
            data=input_ids_data,
            chunks=(chunk_samples, chunk_seq),
            compressors=compressors,
        )
        zarr.create_array(
            store=store,
            name="attention_mask",
            data=attention_mask_data,
            chunks=(chunk_samples, chunk_seq),
            compressors=compressors,
        )

        # Save layer activations with optimized chunking
        activations_group = zarr.create_group(store=store, path="activations")
        for layer_name in layer_names:
            activations_group.create_array(
                name=layer_name,
                data=layer_data_arrays[layer_name],
                chunks=(chunk_samples, chunk_seq, hidden_size),
                compressors=compressors,
            )

        # Save configuration as attributes to the root group
        root_group = zarr.open_group(store=store, mode="r+")
        root_group.attrs["model_name"] = self.cfg.model.name
        root_group.attrs["batch_size"] = self.batch_size
        root_group.attrs["max_length"] = self.max_length
        root_group.attrs["num_layers"] = len(layer_names)
        root_group.attrs["hidden_size"] = hidden_size
        root_group.attrs["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        root_group.attrs["max_seq_len"] = max_seq_len
        root_group.attrs["total_samples"] = total_samples

        save_time = time.time() - start_time
        logging.info(
            f"Saved activations to {filepath} (samples: {total_samples}, max_seq_len: {max_seq_len}, save_time: {save_time:.2f}s)"
        )

    def _calculate_eta(
        self,
        total_tokens_processed: int,
        batch_count: int,
        file_count: int,
        processing_time: float,
        total_save_time: float,
    ) -> str:
        """Calculate estimated time to completion."""
        remaining_tokens = self.num_tokens_target - total_tokens_processed

        if processing_time <= 0 or total_tokens_processed <= 0:
            return "N/A"

        # Calculate pure processing rate (tokens/second excluding saves)
        processing_rate = total_tokens_processed / processing_time

        if processing_rate <= 0:
            return "N/A"

        # Estimate remaining processing time
        remaining_processing_time = remaining_tokens / processing_rate

        # Estimate remaining save operations and their time
        remaining_batches = remaining_tokens / (total_tokens_processed / batch_count)
        remaining_save_operations = remaining_batches / self.save_every_n_batches

        # Calculate average save time per operation
        if file_count > 0:
            avg_save_time = total_save_time / file_count
            remaining_save_time = remaining_save_operations * avg_save_time
        else:
            remaining_save_time = 0

        # Total estimated remaining time
        total_remaining_time = remaining_processing_time + remaining_save_time

        # Format ETA
        if total_remaining_time < 60:
            return f"{total_remaining_time:.0f}s"
        elif total_remaining_time < 3600:
            return f"{total_remaining_time / 60:.1f}m"
        else:
            return f"{total_remaining_time / 3600:.1f}h"

    def _create_progress_postfix(
        self,
        total_tokens_processed: int,
        processing_time: float,
        total_save_time: float,
        elapsed_time: float,
        eta_str: str,
    ) -> dict:
        """Create progress bar postfix dictionary."""
        postfix_dict = {
            "tokens": f"{total_tokens_processed:,}",
            "target": f"{self.num_tokens_target:,}",
            "progress": f"{100 * total_tokens_processed / self.num_tokens_target:.1f}%",
            "proc_speed": f"{total_tokens_processed / processing_time:.0f}tok/s"
            if processing_time > 0
            else "N/A",
            "eta": eta_str,
        }

        if total_save_time > 0:
            postfix_dict["save_time"] = f"{total_save_time:.1f}s"
            postfix_dict["save_pct"] = f"{100 * total_save_time / elapsed_time:.1f}%"

        return postfix_dict

    def _perform_save_and_cleanup(
        self, activations_buffer: list, file_count: int
    ) -> float:
        """Perform save operation and cleanup, return save duration."""
        filename = f"activations_part_{file_count:04d}.zarr.zip"

        # Time the save operation
        save_start_time = time.time()
        self.save_activations_to_zarr(activations_buffer, filename)
        save_duration = time.time() - save_start_time

        # Log memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"GPU memory used: {memory_used:.2f} GB")

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return save_duration

    def _log_final_statistics(
        self,
        total_tokens_processed: int,
        file_count: int,
        total_save_time: float,
        total_elapsed: float,
    ):
        """Log final collection statistics."""
        logging.info("Activation collection complete!")
        logging.info(f"Total tokens processed: {total_tokens_processed:,}")
        logging.info(f"Files saved: {file_count + 1}")
        logging.info(f"Total save time: {total_save_time:.2f}s")

        if total_elapsed > 0:
            processing_time = total_elapsed - total_save_time
            logging.info(
                f"Processing time: {processing_time:.2f}s ({100 * processing_time / total_elapsed:.1f}%)"
            )
            logging.info(f"Save overhead: {100 * total_save_time / total_elapsed:.1f}%")

    def collect_activations(self):
        """Main method to collect activations from the dataset."""
        logging.info("Starting activation collection...")

        # Load model and prepare dataset
        self.load_model_and_tokenizer()
        dataloader = self.prepare_dataset()

        # Initialize collection state
        activations_buffer = []
        total_tokens_processed = 0
        batch_count = 0
        file_count = 0
        total_save_time = 0.0

        try:
            with tqdm(desc="Processing batches", unit="batch") as pbar:
                for batch in dataloader:
                    # Check if we've reached our target
                    if total_tokens_processed >= self.num_tokens_target:
                        logging.info(
                            f"Reached target of {self.num_tokens_target} tokens"
                        )
                        break

                    # Process batch
                    batch_activations = self.collect_batch_activations(batch)
                    activations_buffer.append(batch_activations)

                    # Update counters
                    batch_tokens = batch["attention_mask"].sum().item()
                    total_tokens_processed += batch_tokens
                    batch_count += 1

                    # Update progress bar
                    elapsed_time = pbar.format_dict.get("elapsed", 1)
                    processing_time = elapsed_time - total_save_time
                    eta_str = self._calculate_eta(
                        total_tokens_processed,
                        batch_count,
                        file_count,
                        processing_time,
                        total_save_time,
                    )
                    postfix_dict = self._create_progress_postfix(
                        total_tokens_processed,
                        processing_time,
                        total_save_time,
                        elapsed_time,
                        eta_str,
                    )
                    pbar.set_postfix(postfix_dict)
                    pbar.update(1)

                    # Save periodically
                    if batch_count % self.save_every_n_batches == 0:
                        save_duration = self._perform_save_and_cleanup(
                            activations_buffer, file_count
                        )
                        total_save_time += save_duration
                        activations_buffer = []
                        file_count += 1

            # Save remaining activations
            if activations_buffer:
                save_duration = self._perform_save_and_cleanup(
                    activations_buffer, file_count
                )
                total_save_time += save_duration

            # Log final statistics
            total_elapsed = pbar.format_dict.get("elapsed", 0)
            self._log_final_statistics(
                total_tokens_processed, file_count, total_save_time, total_elapsed
            )

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
