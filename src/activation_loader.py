import os
from pathlib import Path
from typing import Generator, Optional, List
from huggingface_hub import hf_hub_download
from typing import Generator, Optional, List
from huggingface_hub import hf_hub_download
import torch
import zarr
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from zarr.storage import StoreLike


class ActivationLoader:
    def __init__(self, activation_dir_path: str = None, files_to_download: Optional[List[str]] = None):

    def __init__(self, activation_dir_path: str = None, files_to_download: Optional[List[str]] = None):
        self.activation_dir_path = activation_dir_path
        if activation_dir_path is None or not os.path.exists(self.activation_dir_path):
            if files_to_download is None:
                raise ValueError(
                    "Either activation_dir_path must be provided or files_to_download must be specified.")
            # download the path from huggingface
            for file in files_to_download:
                path = hf_hub_download(
                    repo_id="TheRootOf3/ato-activations", filename=file, repo_type="dataset"
                )
                self.activation_dir_path = Path(path).parent
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
        return os.listdir(self.activation_dir_path)

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
            self.num_samples += zarr.open(store,
                                          mode="r")["activations"]["layer_0"].shape[0]

        assert len(self.store_objects) == len(
            self._get_file_list()), "Not all files are loaded."
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
        position_idx: int = -1,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """Get the activations for a specific sample, position and layer.

        Args:
            sample_idx (int): The index of the sample.
            position_idx (int): The index of the position. If -1, all positions are selected.
            layer_idx (int): The index of the layer. If -1, all layers are selected.

        Returns:
            torch.Tensor: The activations for the specified sample, position and layer.
            Note that the return format is (num_positions, num_layers, hidden_size).

        """
        part_id, local_sample_id = self.sample_map(sample_idx)
        store = self.store_objects[part_id]
        z = zarr.open(store, mode="r")

        # compute attention mask
        if position_idx == -1:
            pos_slice = slice(0, z["attention_mask"][local_sample_id].sum())
        else:
            if position_idx >= z["attention_mask"][local_sample_id].sum():
                msg = (
                    f"Position index {position_idx} out of bounds for sample "
                    f"{sample_idx}. The sample has length of "
                    f"{z['attention_mask'][local_sample_id].sum()}"
                )
                raise ValueError(msg)
            pos_slice = slice(position_idx, position_idx + 1)

        if layer_idx != -1:
            return torch.tensor(
                z["activations"][f"layer_{layer_idx}"][local_sample_id,
                                                       pos_slice, :],
            ).unsqueeze(1)

        return torch.stack(
            [
                torch.tensor(
                    z["activations"][f"layer_{layer}"][local_sample_id,
                                                       pos_slice, :],
                )
                for layer in range(len(z["activations"]))
            ],
            dim=1,
        )


class ActivationDataset(IterableDataset):
    def __init__(
        self,
        activation_loader: ActivationLoader,
        idx_list: list[int],
        j_policy: str,
        L: int,
        k: int,
    ):
        self.activation_loader = activation_loader
        self.idx_list = idx_list
        self.j_policy = j_policy
        self.L = L
        self.k = k

    def _get_worker_indices(self) -> list[int]:
        """Get the subset of indices that this worker should process."""
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

        for idx in worker_indices:
            try:
                sample_sequence_length = self.activation_loader.get_sample_sequence_length(
                    idx)
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
                # TODO: Replace with logging
                print(f"Warning: Skipping sample {idx} due to error: {e}")
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


def get_train_val_test_datasets(L, k, loader: ActivationLoader):
    train_indices, val_indices, test_indices = partition_loader(
        num_samples=len(loader),
        train_prop=0.8,
        val_prop=0.1,
        test_prop=0.1
    )

    train_dataset = ActivationDataset(loader, train_indices, "i==j", L, k)
    val_dataset = ActivationDataset(loader, val_indices, "i==j", L, k)
    test_dataset = ActivationDataset(loader, test_indices, "i==j", L, k)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    loader = ActivationLoader("./activations-gemma2-2b-slimpajama-500k")

    # Load activations for the first layer and the first position
    item = loader.get_activation(4, 0, 0)
    print("Sample 4, Layer 0, Position 0. Shape:", item.shape)

    # Load activations for the first layer and all positions
    item = loader.get_activation(4, -1, 0)
    print("Sample 4, Layer 0, All positions. Shape:", item.shape)

    # Load activations for all layers and positions
    item = loader.get_activation(4, -1, -1)
    print("Sample 4, All layers, All positions. Shape:", item.shape)

    print("Testing the IterableDataset")
    print("Total samples (sequences) in activation loader:", len(loader))
    train_indices, val_indices, test_indices = partition_loader(
        len(loader),
        train_prop=0.8,
        val_prop=0.1,
        test_prop=0.1,
    )
    dataset = ActivationDataset(loader, train_indices, "j==i", 0, 2)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

    for x, y in dataloader:
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        break
