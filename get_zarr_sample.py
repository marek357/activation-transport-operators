"""Script to create a sample from a zarr archive file with specified number of samples."""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path

import zarr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_zarr_file(
    source_file_path: Path,
    target_dir_name: Path,
    n_samples: int,
    compression_level: int,
) -> None:
    """Process a single zarr file and create a sample."""
    # Generate target file path based on source file and number of samples
    source_filename = source_file_path.name

    logger.info(f"Processing zarr file: {source_file_path}")
    target_file_path = target_dir_name / source_filename
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
        cname="zstd",
        clevel=compression_level,
        shuffle=zarr.codecs.BloscShuffle.bitshuffle,
    )

    source_z = zarr.open(source_store, mode="r")
    actual_n_samples = n_samples
    if n_samples < 0:
        actual_n_samples = source_z["input_ids"].shape[0]
        logger.info(f"Extracting all {actual_n_samples} samples from {source_filename}")
    else:
        logger.info(f"Extracting {actual_n_samples} samples from {source_filename}")

    zarr.create_array(
        store=target_store,
        name="input_ids",
        data=source_z["input_ids"][:actual_n_samples],
        chunks=source_z["input_ids"].chunks,
        compressors=compressors,
    )
    zarr.create_array(
        store=target_store,
        name="attention_mask",
        data=source_z["attention_mask"][:actual_n_samples],
        chunks=source_z["attention_mask"].chunks,
        compressors=compressors,
    )

    activations_group = zarr.create_group(store=target_store, path="activations")
    for layer_name in tqdm(
        source_z["activations"], desc=f"Processing {source_filename}"
    ):
        layer_data = source_z["activations"][layer_name][:actual_n_samples]
        activations_group.create_array(
            name=layer_name,
            data=layer_data,
            chunks=source_z["activations"][layer_name].chunks,
            compressors=compressors,
        )

    # Update attributes
    root_group = zarr.open_group(store=target_store, mode="r+")
    for key, value in source_z.attrs.items():
        root_group.attrs[key] = value

    root_group.attrs["batch_size"] = actual_n_samples
    root_group.attrs["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    root_group.attrs["max_seq_len"] = int(
        max(source_z["attention_mask"][:actual_n_samples].sum(axis=1))
    )


def main() -> None:
    """Create a sample from a zarr archive file with specified number of samples."""
    parser = argparse.ArgumentParser(
        description="Create a sample from a zarr archive file or all files in a directory"
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="./activations-gemma2-2b-slimpajama-500k/activations_part_0000.zarr.zip",
        help=(
            "Path to the source zarr file or directory containing zarr files "
            "(default: ./activations-gemma2-2b-slimpajama-500k/activations_part_0000.zarr.zip)"
        ),
    )
    parser.add_argument(
        "--target_dir",
        "-t",
        type=str,
        help=("Path to the target zarr directory"),
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=10,
        help="Number of samples to extract (default: 10). Negative values will extract all samples.",
    )
    parser.add_argument(
        "--compression_level",
        "-cl",
        type=int,
        default=4,
        help="Compression level for the output zarr file (default: 4)",
    )

    args = parser.parse_args()
    n_samples = args.samples
    source_path = Path(args.source)
    target_dir_name = Path(args.target_dir)

    logger.info(f"Source path: {source_path}")
    logger.info(f"Target directory: {target_dir_name}")

    # Check if source is a directory or a file
    if source_path.is_dir():
        logger.info(f"Processing directory: {source_path}")
        # Find all zarr files in the directory
        zarr_files = list(source_path.glob("*.zarr.zip"))
        if not zarr_files:
            logger.error(f"No zarr.zip files found in directory: {source_path}")
            return

        logger.info(f"Found {len(zarr_files)} zarr files to process")
        for zarr_file in zarr_files:
            process_zarr_file(
                zarr_file, target_dir_name, n_samples, args.compression_level
            )

    elif source_path.is_file():
        logger.info(f"Processing single file: {source_path}")
        if not source_path.name.endswith(".zarr.zip"):
            logger.warning(f"File {source_path} does not have .zarr.zip extension")
        process_zarr_file(
            source_path, target_dir_name, n_samples, args.compression_level
        )

    else:
        logger.error(f"Source path {source_path} is neither a file nor a directory")
        return

    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
