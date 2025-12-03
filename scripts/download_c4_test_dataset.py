#!/usr/bin/env python3
"""
Download the NeelNanda/c4_test dataset from Hugging Face and save it locally.
This allows using the local dataset entry `c4_test_local` in configs.
"""

import argparse
import os
from datasets import load_dataset, load_from_disk


def download_c4_test_dataset(save_dir: str = "./assets/hf/c4_test", streaming: bool = False):
    repo_id = "NeelNanda/c4_test"
    print(f"Downloading dataset: {repo_id}")
    print(f"Save directory: {save_dir}")
    print(f"Streaming mode: {streaming}")

    ds = load_dataset(repo_id, split="train", streaming=streaming)

    if streaming:
        print("Streaming mode enabled; dataset will not be saved. Remove --streaming to save to disk.")
        return

    os.makedirs(save_dir, exist_ok=True)
    ds.save_to_disk(save_dir)
    print(f"Dataset saved to: {save_dir}")

    # Quick verification
    loaded = load_from_disk(save_dir)
    print(f"Verification: loaded {len(loaded)} examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the NeelNanda/c4_test dataset locally")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./assets/hf/c4_test",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (won't save to disk)",
    )
    args = parser.parse_args()
    download_c4_test_dataset(save_dir=args.save_dir, streaming=args.streaming)
