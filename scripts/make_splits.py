import argparse
import os
from typing import List

import numpy as np
from tqdm import tqdm

from utils.lmdb import read_lmdb


def read_integer_keys(lmdb_path: str) -> List[int]:
    """
    Read all LMDB keys and return them as a list of integers.

    Assumes keys are stored as ASCII strings representing integer indices,
    consistent with MOFDiff/MOFFlow dataset conventions.
    """
    env = read_lmdb(lmdb_path)
    num_entries = env.stat().get('entries', None)
    keys: List[int] = []
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        iterator = cursor
        if num_entries is not None:
            iterator = tqdm(cursor, desc="Scanning LMDB keys", total=num_entries)
        for key_bytes, _ in iterator:
            try:
                key_int = int(key_bytes.decode("ascii"))
            except Exception:
                # Fallback: skip non-integer keys
                continue
            keys.append(key_int)
    env.close()
    return keys


def write_split_file(path: str, indices: List[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for idx in indices:
            f.write(f"{idx}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/val split index files from a MetalOxo LMDB")
    parser.add_argument(
        "--lmdb-path",
        default="/home/suny0a/chem_root/MOFFlow-2/MOFFLOW2_data/lmdb/MetalOxo.lmdb",
        help="Path to MetalOxo.lmdb",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/suny0a/chem_root/MOFFlow-2/MOFFLOW2_data/splits",
        help="Directory where split files will be written",
    )
    parser.add_argument(
        "--task",
        choices=["csp", "gen"],
        default="csp",
        help="Task subdirectory name under output-dir (default: csp)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of data to use for training (default: 0.9)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing split files",
    )
    args = parser.parse_args()

    # Read keys
    print(f"Reading keys from: {args.lmdb_path}")
    keys = read_integer_keys(args.lmdb_path)
    if not keys:
        raise RuntimeError("No integer keys found in the LMDB. Aborting.")
    print(f"Found {len(keys)} total keys")

    # Shuffle deterministically
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(keys))
    keys = [keys[i] for i in perm]

    # Compute split sizes
    train_count = int(len(keys) * args.train_ratio)
    # Ensure at least one sample in val if possible
    if train_count >= len(keys) and len(keys) > 1:
        train_count = len(keys) - 1

    train_indices = sorted(keys[:train_count])
    val_indices = sorted(keys[train_count:])

    # Prepare output paths
    task_dir = os.path.join(args.output_dir, args.task)
    train_path = os.path.join(task_dir, "train_split.txt")
    val_path = os.path.join(task_dir, "val_split.txt")

    # Existence checks
    for p in (train_path, val_path):
        if os.path.exists(p) and not args.overwrite:
            raise FileExistsError(f"Split file already exists: {p} (use --overwrite to replace)")

    # Write files
    write_split_file(train_path, train_indices)
    write_split_file(val_path, val_indices)

    print(f"Wrote train split: {train_path} (n={len(train_indices)})")
    print(f"Wrote  val  split: {val_path} (n={len(val_indices)})")


if __name__ == "__main__":
    main()


