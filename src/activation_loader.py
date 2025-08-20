import logging
import os
from pathlib import Path
from typing import Generator, Optional, List
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from tqdm import tqdm
import zarr
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from zarr.storage import StoreLike

logger = logging.getLogger(__name__)


class ActivationLoader:
    def __init__(
        self,
        activation_dir_path: str = None,
        files_to_download: Optional[List[str]] = None,
    ):
        self.activation_dir_path = activation_dir_path
        if activation_dir_path is None:
            if files_to_download is None:
                raise ValueError(
                    "Either activation_dir_path must be provided or files_to_download must be specified."
                )
            # download the path from huggingface
            for file in files_to_download:
                path = hf_hub_download(
                    repo_id="TheRootOf3/ato-activations",
                    filename=file,
                    repo_type="dataset",
                )
                self.activation_dir_path = Path(path).parent
        elif not os.path.exists(self.activation_dir_path):
            raise ValueError(
                f"Activation directory {self.activation_dir_path} does not exist."
            )
        self.store_objects: dict[int, StoreLike] = {}
        self.num_samples = 0
        self.samples_per_file = 0
        self.create_store_objects()

    def sample_map(self, idx: int) -> tuple[int, int]:
        if self.samples_per_file == 0:
            msg = "Sample map is not created."
            raise ValueError(msg)
        return (idx // self.samples_per_file, idx % self.samples_per_file)

    def _get_file_list(self) -> list[str]:
        return sorted(os.listdir(self.activation_dir_path))

    def __len__(self) -> int:
        return self.num_samples

    def create_store_objects(self) -> None:
        for i, file_name in enumerate(self._get_file_list()):
            file_path = Path(self.activation_dir_path) / file_name

            store = zarr.storage.ZipStore(
                file_path,
                read_only=True,
            )
            self.store_objects[i] = store
            self.num_samples += zarr.open(store, mode="r")["activations"][
                "layer_0"
            ].shape[0]

        assert len(self.store_objects) == len(self._get_file_list()), (
            "Not all files are loaded."
        )
        z = zarr.open(self.store_objects[0], mode="r")
        self.samples_per_file = z["activations"]["layer_0"].shape[0]

    def get_sample_sequence_length(self, sample_idx: int) -> int:
        part_id, local_sample_id = self.sample_map(sample_idx)
        store = self.store_objects[part_id]
        z = zarr.open(store, mode="r")
        return z["attention_mask"][local_sample_id].sum()

    def get_activation(
        self,
        sample_idx: int,
        position_idx: int,
        layer_idx: int,
        use_pre_loaded: bool = False,
        pre_loaded_activations: Optional[dict[int, torch.Tensor]] = None,
        pre_loaded_attention_masks: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get the activations for a specific sample, position and layer.

        Args:
            sample_idx (int): The index of the sample.
            position_idx (int): The index of the position.
            layer_idx (int): The index of the layer.
            use_pre_loaded (bool): Whether to use pre-loaded activations.
            pre_loaded_activations (Optional[dict[int, torch.Tensor]]): Pre-loaded activations for a specific layer.
            pre_loaded_attention_masks (Optional[dict[int, torch.Tensor]]): Pre-loaded attention masks for a specific layer.

        Returns:
            torch.Tensor: The activations for the specified sample, position and layer.
            Note that the return format is (num_positions, num_layers, hidden_size).

        """
        part_id, local_sample_id = self.sample_map(sample_idx)
        if use_pre_loaded:
            if pre_loaded_activations is None or pre_loaded_attention_masks is None:
                msg = "Pre-loaded activations or attention masks are not available."
                raise ValueError(msg)
            attention_mask = pre_loaded_attention_masks[part_id][local_sample_id]
            activations = pre_loaded_activations[part_id][local_sample_id]
        else:
            store = self.store_objects[part_id]
            z = zarr.open(store, mode="r")
            attention_mask = z["attention_mask"][local_sample_id]
            activations = torch.from_numpy(
                z["activations"][f"layer_{layer_idx}"][local_sample_id, :, :]
            )

        # compute attention mask
        if position_idx >= attention_mask.sum():
            msg = (
                f"Position index {position_idx} out of bounds for sample "
                f"{sample_idx}. The sample has length of "
                f"{attention_mask.sum()}"
            )
            raise ValueError(msg)

        return activations[position_idx, :].unsqueeze(0).unsqueeze(0)

    def get_all_activations_per_layer(
        self, layer: int
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """
        Get all activations and attention masks for a specific layer.

        Args:
            layer (int): The layer index.

        Returns:
            tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]: A tuple containing
            two dictionaries: the first maps store IDs to activation tensors, and the
            second maps store IDs to attention mask tensors.
        """
        logger.info(f"Pre-loading activations and attention masks for layer {layer}...")
        all_activations, all_attention_masks = {}, {}
        for store_id, store_object in self.store_objects.items():
            z = zarr.open(store_object, mode="r")

            np_activations_arr = np.ascontiguousarray(
                z["activations"][f"layer_{layer}"]
            )
            np_attention_mask_arr = np.ascontiguousarray(z["attention_mask"])

            all_activations[store_id] = torch.from_numpy(np_activations_arr)
            all_attention_masks[store_id] = torch.from_numpy(np_attention_mask_arr)
        logger.info(f"Pre-loaded activations and attention masks for layer {layer}")
        return all_activations, all_attention_masks


class ActivationDataset(IterableDataset):
    def __init__(
        self,
        activation_loader: ActivationLoader,
        idx_list: list[int],
        j_policy: str,
        L: int,
        k: int,
        dataset_id: str = "default",
    ):
        self.activation_loader = activation_loader
        self.idx_list = idx_list
        self.j_policy = j_policy
        self.L = L
        self.k = k
        self.dataset_id = dataset_id

    def _get_worker_indices(self) -> list[int]:
        """Get the subset of indices that this worker should process ."""
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading, return all indices
            return self.idx_list

        # Multi-process data loading, partition indices among workers
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        # Partition indices for this worker
        worker_indices = []
        for i, idx in enumerate(self.idx_list):
            if i % num_workers == worker_id:
                worker_indices.append(idx)

        return worker_indices

    def get_next_j_equals_i(
        self,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        worker_indices = self._get_worker_indices()
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else -1
        num_workers = worker_info.num_workers if worker_info else 0

        for idx in tqdm(
            worker_indices,
            desc="Processing sequence activations for "
            + str(f"worker id: {worker_id} (out of {num_workers})")
            if worker_id != -1
            else "the main process",
        ):
            try:
                sample_sequence_length = (
                    self.activation_loader.get_sample_sequence_length(idx)
                )
                for pos in range(sample_sequence_length):
                    x_up = self.activation_loader.get_activation(
                        idx,
                        pos,
                        self.L,
                    )
                    y_down = self.activation_loader.get_activation(
                        idx,
                        pos,
                        self.L + self.k,
                    )

                    # Ensure consistent tensor shapes (squeeze position dim since we're getting single positions)
                    # Remove position and layer dimension
                    x_up = x_up.squeeze(0).squeeze(0)
                    # Remove position and layer dimension
                    y_down = y_down.squeeze(0).squeeze(0)

                    yield (x_up, y_down)
            except (ValueError, IndexError) as e:
                logger.warning(
                    "Skipping sample %d due to error: %s", idx, str(e), exc_info=True
                )
                continue

    def __iter__(self):
        if self.j_policy == "j==i":
            return self.get_next_j_equals_i()
        else:
            msg = "Other j-policies are not yet implemented."
            raise NotImplementedError(msg)


class EfficientActivationDataset(IterableDataset):
    def __init__(
        self,
        activation_loader: ActivationLoader,
        idx_list: list[int],
        j_policy: str,
        L: int,
        k: int,
        dataset_id: str = "default",
    ):
        self.activation_loader = activation_loader
        self.idx_list = idx_list
        self.j_policy = j_policy
        self.L = L
        self.k = k
        self.dataset_id = dataset_id
        self.pre_loaded_activations = {}
        self.pre_loaded_attention_masks = {}

        if self.activation_loader is not None:
            self._pre_load_layer_activations()
        else:
            logger.warning("Activation loader is not available. Entering dummy mode.")

    def _pre_load_layer_activations(self) -> None:
        self.pre_loaded_activations[self.L], self.pre_loaded_attention_masks[self.L] = (
            self.activation_loader.get_all_activations_per_layer(self.L)
        )
        (
            self.pre_loaded_activations[self.L + self.k],
            self.pre_loaded_attention_masks[self.L + self.k],
        ) = self.activation_loader.get_all_activations_per_layer(self.L + self.k)

    def _get_worker_indices(self) -> list[int]:
        """Get the subset of indices that this worker should process ."""
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading, return all indices
            return self.idx_list

        # Multi-process data loading, partition indices among workers
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        # Partition indices for this worker
        worker_indices = []
        for i, idx in enumerate(self.idx_list):
            if i % num_workers == worker_id:
                worker_indices.append(idx)

        return worker_indices

    def get_next_j_equals_i(
        self,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        worker_indices = self._get_worker_indices()
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else -1
        num_workers = worker_info.num_workers if worker_info else 0

        for idx in tqdm(
            worker_indices,
            desc="Processing sequence activations for "
            + str(f"worker id: {worker_id} (out of {num_workers})")
            if worker_id != -1
            else "the main process",
        ):
            try:
                sample_sequence_length = (
                    self.activation_loader.get_sample_sequence_length(idx)
                )
                for pos in range(sample_sequence_length):
                    x_up = self.activation_loader.get_activation(
                        idx,
                        pos,
                        self.L,
                        use_pre_loaded=True,
                        pre_loaded_activations=self.pre_loaded_activations[self.L],
                        pre_loaded_attention_masks=self.pre_loaded_attention_masks[
                            self.L
                        ],
                    )
                    y_down = self.activation_loader.get_activation(
                        idx,
                        pos,
                        self.L + self.k,
                        use_pre_loaded=True,
                        pre_loaded_activations=self.pre_loaded_activations[
                            self.L + self.k
                        ],
                        pre_loaded_attention_masks=self.pre_loaded_attention_masks[
                            self.L + self.k
                        ],
                    )

                    # Ensure consistent tensor shapes (squeeze position dim since we're getting single positions)
                    # Remove position and layer dimension
                    x_up = x_up.squeeze(0).squeeze(0)
                    # Remove position and layer dimension
                    y_down = y_down.squeeze(0).squeeze(0)

                    yield (x_up, y_down)
            except (ValueError, IndexError) as e:
                logger.warning(
                    "Skipping sample %d due to error: %s", idx, str(e), exc_info=True
                )
                continue

    def __iter__(self):
        if self.j_policy == "j==i":
            return self.get_next_j_equals_i()
        else:
            msg = "Other j-policies are not yet implemented."
            raise NotImplementedError(msg)


def partition_loader(
    num_samples: int,
    train_prop: float,
    val_prop: float,
    test_prop: float,
):
    assert train_prop + val_prop + test_prop == 1, "Proportions must sum to 1"

    train_size = int(num_samples * train_prop)
    val_size = int(num_samples * val_prop)

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, num_samples))

    return train_indices, val_indices, test_indices


def get_train_val_test_datasets(
    L: int,
    k: int,
    loader: ActivationLoader,
    j_policy: str = "j==i",
    load_efficient: bool = True,
) -> tuple[
    ActivationDataset | EfficientActivationDataset,
    ActivationDataset | EfficientActivationDataset,
    ActivationDataset | EfficientActivationDataset,
]:
    train_indices, val_indices, test_indices = partition_loader(
        num_samples=len(loader), train_prop=0.6, val_prop=0.2, test_prop=0.2
    )

    ds_class = ActivationDataset
    if load_efficient:
        ds_class = EfficientActivationDataset

    train_dataset = ds_class(loader, train_indices, j_policy, L, k, f"train_L{L}_k{k}")
    val_dataset = ds_class(loader, val_indices, j_policy, L, k, f"val_L{L}_k{k}")
    test_dataset = ds_class(loader, test_indices, j_policy, L, k, f"test_L{L}_k{k}")

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    loader = ActivationLoader("./activations-gemma2-2b-slimpajama-250k")

    # Load activations for the first layer and the first position
    item = loader.get_activation(4, 0, 0)
    print("Sample 4, Layer 0, Position 0. Shape:", item.shape)

    print("Testing the IterableDataset")
    print("Total samples (sequences) in activation loader:", len(loader))
    train_indices, val_indices, test_indices = partition_loader(
        len(loader),
        train_prop=0.6,
        val_prop=0.2,
        test_prop=0.2,
    )
    dataset = EfficientActivationDataset(
        loader, train_indices, "j==i", 0, 2, "test_dataset"
    )
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

    for x, y in dataloader:
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        print("X dtype:", x.dtype)
        print("Y dtype:", y.dtype)
        break
