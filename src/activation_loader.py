import os
import torch
import zarr
from pathlib import Path
from zarr.storage import StoreLike


class ActivationLoader:
    def __init__(self, activation_dir_path: str):
        self.activation_dir_path = activation_dir_path
        self.store_objects: dict[int, StoreLike] = {}
        self.sample_map: callable = None

        self.create_store_objects_and_sample_map()

    def _get_file_list(self) -> list[str]:
        return os.listdir(self.activation_dir_path)

    def create_store_objects_and_sample_map(self) -> None:
        for i, file_name in enumerate(self._get_file_list()):
            file_path = Path(self.activation_dir_path) / file_name

            store = zarr.storage.ZipStore(
                file_path,
                read_only=True,
            )
            self.store_objects[i] = store

        assert len(self.store_objects) == len(self._get_file_list()), (
            "Not all files are loaded."
        )
        z = zarr.open(self.store_objects[0], mode="r")
        batch_size = z["activations"]["layer_0"].shape[0]

        # Note, this assumes each zarr activation part is of equal size
        self.sample_map = lambda x: (x // batch_size, x % batch_size)

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
        if self.sample_map is None:
            msg = "Sample map is not created."
            raise ValueError(msg)
        part, sample = self.sample_map(sample_idx)
        store = self.store_objects[part]
        z = zarr.open(store, mode="r")

        # compute attention mask
        if position_idx == -1:
            pos_slice = slice(0, z["attention_mask"][sample_idx].sum())
        else:
            if position_idx >= z["attention_mask"][sample_idx].sum():
                msg = (
                    f"Position index {position_idx} out of bounds for sample "
                    f"{sample_idx}. The sample has length of "
                    f"{z['attention_mask'][sample_idx].sum()}"
                )
                raise ValueError(msg)
            pos_slice = slice(position_idx, position_idx + 1)

        if layer_idx != -1:
            return torch.tensor(
                z["activations"][f"layer_{layer_idx}"][sample, pos_slice, :],
            ).unsqueeze(1)

        return torch.stack(
            [
                torch.tensor(
                    z["activations"][f"layer_{layer}"][sample, pos_slice, :],
                )
                for layer in range(len(z["activations"]))
            ],
            dim=1,
        )


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
