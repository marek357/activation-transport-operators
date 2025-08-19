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
import hydra
from botocore import UNSIGNED
from botocore.config import Config
from omegaconf import DictConfig
from tqdm import tqdm


# Create module-level logger
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_feature_ids_from_json(json_file_path: str) -> list[int]:
    """Load feature IDs from a JSON file.

    Args:
        json_file_path: Path to JSON file containing a list of feature IDs

    Returns:
        List of feature IDs

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
        ValueError: If the JSON doesn't contain a list of integers
    """
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)

        if "high_quality_feature_ids" in data:
            data = data["high_quality_feature_ids"]
        else:
            raise ValueError("JSON file must contain 'high_quality_feature_ids' key")

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of feature IDs")

        # Validate that all items are integers
        feature_ids = []
        for item in data:
            if not isinstance(item, int):
                raise ValueError(
                    f"All feature IDs must be integers, found: {type(item)}"
                )
            feature_ids.append(item)

        return feature_ids

    except FileNotFoundError:
        raise FileNotFoundError(f"Feature IDs JSON file not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in feature IDs file: {e}")


def create_s3_client() -> boto3.client:
    """Create an S3 client with unsigned config for public bucket access."""
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
    )


def list_explanation_files(s3_client: boto3.client, sae_path: str) -> list[str]:
    """List all explanation files for a given SAE path.

    Args:
        s3_client: Boto3 S3 client
        sae_path: SAE path

    Returns:
        List of S3 object keys for explanation files

    """
    bucket_name = "neuronpedia-datasets"
    prefix = f"v1/gemma-2-2b/{sae_path}/explanations/"

    logger.info("Listing files with prefix: %s", prefix)

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

        logger.info("Found %s explanation files", len(files))
        return files

    except Exception:
        logger.exception("Error listing files")
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

    except Exception:
        logger.exception("Error processing file %s", key)
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

    except Exception:
        logger.exception("Error downloading file %s", key)
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

    except Exception:
        logger.exception("Error loading cached file %s", cache_path)
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
        logger.info("Loading from cache: %s", cache_path)
        return load_cached_file(cache_path)

    # Download and cache the file
    logger.info("Downloading and caching: %s", key)
    download_and_cache_file(s3_client, bucket_name, key, cache_path)

    # Load the cached file
    return load_cached_file(cache_path)


def load_explanations_dict(
    sae_path: str,
    feature_ids: Optional[list[int]] = None,
    cache_dir: Optional[str] = None,
) -> dict[int, dict]:
    """Load explanations for specified feature IDs into a dictionary.

    Args:
        sae_path: SAE path
        feature_ids: List of feature IDs to fetch. If None, fetch all.
        cache_dir: Optional directory to cache downloaded files

    Returns:
            Dictionary mapping feature ID (int) to explanation data
            Order is preserved if feature_ids list is provided.

    """
    s3_client = create_s3_client()
    bucket_name = "neuronpedia-datasets"

    # Convert feature_ids to set for faster lookup
    target_features: Optional[set[int]] = None
    if feature_ids is not None:
        target_features = set(feature_ids)
        logger.info(
            "Fetching explanations for %s specific features",
            len(target_features),
        )
    else:
        logger.info("Fetching all available explanations")

    # List all explanation files
    explanation_files = list_explanation_files(s3_client, sae_path)

    if not explanation_files:
        logger.warning("No explanation files found for SAE ID: %s", sae_path)
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
                        logger.info("Found all requested features, stopping early")
                        break

            except (KeyError, ValueError):
                logger.warning("Skipping malformed explanation entry")
                continue

        # Break out of outer loop if we found all features
        if target_features is not None and len(explanations_dict) >= len(
            target_features,
        ):
            remaining_features = target_features - set(explanations_dict.keys())
            if not remaining_features:
                break

    logger.info("Loaded %s explanations", len(explanations_dict))

    # Report missing features if we were looking for specific ones
    if target_features is not None:
        missing_features = target_features - set(explanations_dict.keys())
        if missing_features:
            logger.warning(
                "Could not find explanations for %s features: %s",
                len(missing_features),
                sorted(missing_features),
            )

    return explanations_dict


def print_explanation_summary(explanation: dict, logger: logging.Logger) -> None:
    """Print a summary of an explanation."""
    logger.info(
        "Feature %s: %s",
        explanation["index"],
        explanation.get("description", "No description"),
    )
    logger.info("Model: %s", explanation.get("modelId", "Unknown"))
    logger.info("Layer: %s", explanation.get("layer", "Unknown"))
    logger.info("Author: %s", explanation.get("authorId", "Unknown"))
    if "umap_cluster" in explanation:
        logger.info("UMAP Cluster: %s", explanation["umap_cluster"])


@hydra.main(version_base=None, config_path="configs", config_name="explain_features")
def main(cfg: DictConfig) -> None:
    """Run feature explanation fetching using Hydra configuration."""
    setup_logging()

    sae_path = cfg.sae_path

    logger.info("Using SAE ID: %s", sae_path)

    # Get feature IDs from configuration
    feature_ids = load_feature_ids_from_json(cfg.feature_ids_file)
    logger.info("Loaded %s feature IDs from %s", len(feature_ids), cfg.feature_ids_file)

    # Get cache directory
    cache_dir = cfg.cache_dir

    try:
        # Load explanations with caching
        explanations_dict = load_explanations_dict(
            sae_path,
            feature_ids,
            cache_dir,
        )

        # Print summary
        logger.info("Loaded %s explanations", len(explanations_dict))

        # Show first few explanations as examples
        for i, (_, explanation) in enumerate(
            list(explanations_dict.items())[:3],
        ):
            logger.info("--- Example %s ---", i + 1)
            print_explanation_summary(explanation, logger)

        # Save to file if requested
        output_file = cfg.get("output_file")
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # If specific feature IDs were provided, preserve their order
            if feature_ids:
                ordered_explanations = [
                    explanations_dict[feature_id]
                    for feature_id in feature_ids
                    if feature_id in explanations_dict
                ]

                explanations = {}
                explanations["data"] = ordered_explanations
                explanations["meta"] = {
                    "sae_id": sae_path,
                    "sae_layer": cfg.sae_layer,
                    "feature_ids_file": cfg.feature_ids_file,
                    "num_features": len(ordered_explanations),
                }
                with output_path.open("w") as f:
                    json.dump(explanations, f, indent=2)

            logger.info("Saved explanations to %s", output_file)

    except Exception:
        logger.exception("Error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
