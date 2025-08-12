#! /usr/bin/env python3
"""Script to fetch feature explanations from Neuronpedia S3 bucket.

This script downloads and loads explanation files for specified feature IDs
from the neuronpedia-datasets S3 bucket. Files are in JSONL.gz format
(gzipped JSON Lines, one JSON object per line).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_s3_client() -> boto3.client:
    """Create an S3 client with unsigned config for public bucket access."""
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
    )


def list_explanation_files(s3_client: boto3.client, sae_id: str) -> list[str]:
    """List all explanation files for a given SAE ID.

    Args:
        s3_client: Boto3 S3 client
        sae_id: SAE identifier

    Returns:
        List of S3 object keys for explanation files

    """
    bucket_name = "neuronpedia-datasets"
    prefix = f"v1/gemma-2-2b/{sae_id}/explanations/"

    logging.info(f"Listing files with prefix: {prefix}")

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        files = []
        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith(".jsonl.gz"):
                        files.append(key)

        logging.info(f"Found {len(files)} explanation files")
        return files

    except Exception as e:
        logging.exception(f"Error listing files: {e}")
        raise


def download_and_parse_file(
    s3_client: boto3.client,
    bucket_name: str,
    key: str,
) -> list[dict]:
    """Download and parse a single explanation file.

    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        key: S3 object key

    Returns:
        List of explanation dictionaries

    """
    try:
        # Download the file
        response = s3_client.get_object(Bucket=bucket_name, Key=key)

        # Decompress and parse JSONL
        with gzip.GzipFile(fileobj=response["Body"]) as gz_file:
            content = gz_file.read().decode("utf-8")

        # Parse JSONL format (one JSON object per line)
        explanations = []
        for line in content.strip().split("\n"):
            if line.strip():
                explanations.append(json.loads(line))

        return explanations

    except Exception as e:
        logging.exception(f"Error processing file {key}: {e}")
        return []


def get_cached_file_path(cache_dir: str, key: str) -> Path:
    """Get the local cache file path for an S3 key.

    Args:
        cache_dir: Cache directory path
        key: S3 object key

    Returns:
        Path object for the cached file
    """
    # Replace path separators to create a valid filename
    filename = key.replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / filename


def download_and_cache_file(
    s3_client: boto3.client,
    bucket_name: str,
    key: str,
    cache_path: Path,
) -> None:
    """Download a file from S3 and save it to cache.

    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        key: S3 object key
        cache_path: Local path to save the file
    """
    try:
        # Create cache directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        response = s3_client.get_object(Bucket=bucket_name, Key=key)

        # Save to cache
        with cache_path.open("wb") as f:
            f.write(response["Body"].read())

    except Exception as e:
        logging.exception(f"Error downloading file {key}: {e}")
        raise


def load_cached_file(cache_path: Path) -> list[dict]:
    """Load and parse a cached JSONL.gz file.

    Args:
        cache_path: Path to the cached file

    Returns:
        List of explanation dictionaries
    """
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            content = f.read()

        # Parse JSONL format (one JSON object per line)
        explanations = []
        for line in content.strip().split("\n"):
            if line.strip():
                explanations.append(json.loads(line))

        return explanations

    except Exception as e:
        logging.exception(f"Error loading cached file {cache_path}: {e}")
        return []


def download_and_parse_file_with_cache(
    s3_client: boto3.client,
    bucket_name: str,
    key: str,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """Download and parse a single explanation file, with optional caching.

    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        key: S3 object key
        cache_dir: Optional directory to cache files

    Returns:
        List of explanation dictionaries

    """
    # If no cache directory provided, download directly
    if cache_dir is None:
        return download_and_parse_file(s3_client, bucket_name, key)

    # Check if file exists in cache
    cache_path = get_cached_file_path(cache_dir, key)

    if cache_path.exists():
        logging.info(f"Loading from cache: {cache_path}")
        return load_cached_file(cache_path)

    # Download and cache the file
    logging.info(f"Downloading and caching: {key}")
    download_and_cache_file(s3_client, bucket_name, key, cache_path)

    # Load the cached file
    return load_cached_file(cache_path)


def load_explanations_dict(
    sae_id: str,
    feature_ids: Optional[list[int]] = None,
    cache_dir: Optional[str] = None,
) -> dict[int, dict]:
    """Load explanations for specified feature IDs into a dictionary.

    Args:
        sae_id: SAE identifier
        feature_ids: List of feature IDs to fetch. If None, fetch all.
        cache_dir: Optional directory to cache downloaded files

    Returns:
            Dictionary mapping feature ID (int) to explanation data

    """
    s3_client = create_s3_client()
    bucket_name = "neuronpedia-datasets"

    # Convert feature_ids to set for faster lookup
    target_features: Optional[set[int]] = None
    if feature_ids is not None:
        target_features = set(feature_ids)
        logging.info(
            f"Fetching explanations for {len(target_features)} specific features",
        )
    else:
        logging.info("Fetching all available explanations")

    # List all explanation files
    explanation_files = list_explanation_files(s3_client, sae_id)

    if not explanation_files:
        logging.warning(f"No explanation files found for SAE ID: {sae_id}")
        return {}

    # Dictionary to store explanations
    explanations_dict: dict[int, dict] = {}

    # Process each file
    for file_key in tqdm(explanation_files, desc="Processing explanation files"):
        explanations = download_and_parse_file_with_cache(
            s3_client,
            bucket_name,
            file_key,
            cache_dir,
        )

        for explanation in explanations:
            try:
                feature_idx = int(explanation["index"])

                # If we're filtering by feature IDs, skip if not in target set
                if target_features is not None and feature_idx not in target_features:
                    continue

                explanations_dict[feature_idx] = explanation

                # If we found all target features, we can stop early
                if target_features is not None and len(explanations_dict) >= len(
                    target_features,
                ):
                    remaining_features = target_features - set(explanations_dict.keys())
                    if not remaining_features:
                        logging.info("Found all requested features, stopping early")
                        break

            except (KeyError, ValueError) as e:
                logging.warning(f"Skipping malformed explanation entry: {e}")
                continue

        # Break out of outer loop if we found all features
        if target_features is not None and len(explanations_dict) >= len(
            target_features,
        ):
            remaining_features = target_features - set(explanations_dict.keys())
            if not remaining_features:
                break

    logging.info(f"Loaded {len(explanations_dict)} explanations")

    # Report missing features if we were looking for specific ones
    if target_features is not None:
        missing_features = target_features - set(explanations_dict.keys())
        if missing_features:
            logging.warning(
                f"Could not find explanations for {len(missing_features)} features: {sorted(missing_features)}",
            )

    return explanations_dict


def get_feature_explanation(
    explanations_dict: dict[int, dict],
    feature_id: int,
) -> Optional[dict]:
    """Get explanation for a specific feature ID.

    Args:
        explanations_dict: Dictionary of explanations
        feature_id: Feature ID to look up

    Returns:
        Explanation dictionary or None if not found

    """
    return explanations_dict.get(feature_id)


def print_explanation_summary(explanation: dict) -> None:
    """Print a summary of an explanation."""
    print(
        f"Feature {explanation['index']}: {explanation.get('description', 'No description')}",
    )
    print(f"Model: {explanation.get('modelId', 'Unknown')}")
    print(f"Layer: {explanation.get('layer', 'Unknown')}")
    print(f"Author: {explanation.get('authorId', 'Unknown')}")
    if "umap_cluster" in explanation:
        print(f"UMAP Cluster: {explanation['umap_cluster']}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Fetch feature explanations from Neuronpedia",
    )
    parser.add_argument("sae_id", help="SAE identifier")
    parser.add_argument(
        "--feature-ids",
        type=int,
        nargs="+",
        help="Specific feature IDs to fetch (if not provided, fetches all)",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file to save explanations",
    )
    parser.add_argument(
        "--cache-dir",
        default="./explanations_cache",
        help="Directory to cache downloaded files (default: ./explanations_cache)",
    )

    args = parser.parse_args()
    setup_logging()

    try:
        # Load explanations with caching
        explanations_dict = load_explanations_dict(
            args.sae_id,
            args.feature_ids,
            args.cache_dir,
        )

        # Print summary
        print(f"\nLoaded {len(explanations_dict)} explanations")

        # Show first few explanations as examples
        for i, (_, explanation) in enumerate(
            list(explanations_dict.items())[:3],
        ):
            print(f"\n--- Example {i + 1} ---")
            print_explanation_summary(explanation)

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(explanations_dict, f, indent=2)
            print(f"\nSaved explanations to {args.output}")

    except Exception as e:
        logging.exception(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
