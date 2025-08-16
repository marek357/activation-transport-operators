"""Script to create a sample from a zarr archive file with specified number of samples."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import zarr
from tqdm import tqdm


def main() -> None:
    """Create a sample from a zarr archive file with specified number of samples."""
    parser = argparse.ArgumentParser(
        description="Create a sample from a zarr archive file")
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="./activations-gemma2-2b-slimpajama-500k/activations_part_0000.zarr.zip",
        help=(
            "Path to the source zarr file "
            "(default: ./activations-gemma2-2b-slimpajama-500k/activations_part_0000.zarr.zip)"
        ),
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=10,
        help="Number of samples to extract (default: 10)",
    )

    args = parser.parse_args()
    n_samples = args.samples
    source_file_path = Path(args.source)

    # Generate target file path based on source file and number of samples
    source_dir = source_file_path.parent
    source_filename = source_file_path.name

    # Extract the base directory name for the target
    source_dir_name = source_dir.name if source_dir.name != "." else "activations"
    target_dir_name = f"{source_dir_name}_sample{n_samples}"
    target_file_path = source_dir.parent / target_dir_name / source_filename
    target_file_path.parent.mkdir(parents=True, exist_ok=True)

    source_store = zarr.storage.ZipStore(
        str(source_file_path),
        read_only=True,
    )
    target_store = zarr.storage.ZipStore(
        str(target_file_path),
        mode="w",
    )

    compressors = zarr.codecs.BloscCodec(
        cname="zstd", clevel=9, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

    source_z = zarr.open(source_store, mode="r")

    zarr.create_array(
        store=target_store,
        name="input_ids",
        data=source_z["input_ids"][:n_samples],
        chunks="auto",
        compressors=compressors,
    )
    zarr.create_array(
        store=target_store,
        name="attention_mask",
        data=source_z["attention_mask"][:n_samples],
        chunks="auto",
        compressors=compressors,
    )

    activations_group = zarr.create_group(
        store=target_store, path="activations")
    for layer_name in tqdm(source_z["activations"]):
        layer_data = source_z["activations"][layer_name][:n_samples]
        activations_group.create_array(
            name=layer_name,
            data=layer_data,
            chunks="auto",
            compressors=compressors,
        )

    # Update attributes
    root_group = zarr.open_group(store=target_store, mode="r+")
    for key, value in source_z.attrs.items():
        root_group.attrs[key] = value

    root_group.attrs["batch_size"] = n_samples
    root_group.attrs["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    root_group.attrs["max_seq_len"] = int(
        max(source_z["attention_mask"][:n_samples].sum(axis=1)))


if __name__ == "__main__":
    main()
