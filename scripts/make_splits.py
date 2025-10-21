import argparse
import os
from typing import List

import numpy as np
from tqdm import tqdm

from utils.lmdb import read_lmdb


def read_integer_keys(lmdb_path: str, print_count: int = 0, print_raw: bool = False, print_hex: bool = False) -> List[int]:
    """
    Read all LMDB keys and return them as a list of integers.

    Assumes keys are stored as ASCII strings representing integer indices,
    consistent with MOFDiff/MOFFlow dataset conventions.
    Optionally prints the first N scanned keys for debugging.
    """
    env = read_lmdb(lmdb_path)
    num_entries = env.stat().get('entries', None)
    keys: List[int] = []
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        iterator = cursor
        if num_entries is not None:
            iterator = tqdm(cursor, desc="Scanning LMDB keys", total=num_entries)
        printed = 0
        for key_bytes, _ in iterator:
            # Convert potential memoryview to bytes first
            raw = bytes(key_bytes)
            key_str = None
            parsed_int = None
            # Try to decode as ASCII string
            try:
                key_str = raw.decode("ascii")
            except Exception:
                key_str = None
            # Try to parse integer
            if key_str is not None:
                try:
                    parsed_int = int(key_str)
                except Exception:
                    parsed_int = None
            if parsed_int is not None:
                keys.append(parsed_int)
            # Debug printing of keys
            if print_count and printed < print_count:
                if print_raw and key_str is not None:
                    print(f"DEBUG:: key_raw='{key_str}' parsed_int={parsed_int}")
                elif print_hex:
                    print(f"DEBUG:: key_hex='{raw.hex()}' len={len(raw)} parsed_int={parsed_int}")
                else:
                    print(f"DEBUG:: parsed_int={parsed_int}")
                printed += 1
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
    parser.add_argument(
        "--print-keys",
        type=int,
        default=0,
        help="Print first N keys during scan for debugging (default: 0)",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print raw decoded key strings (default: print parsed integers)",
    )
    parser.add_argument(
        "--print-hex",
        action="store_true",
        help="Print raw key bytes in hex (and length) for debugging",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: assume keys are contiguous numeric 0..N-1 and skip scanning",
    )

    args = parser.parse_args()

    # Read keys
    print(f"Reading keys from: {args.lmdb_path}")
    if args.fast:
        print("Fast mode enabled: generating contiguous indices from LMDB entry count")
        env = read_lmdb(args.lmdb_path)
        total = env.stat().get('entries', 0)
        env.close()
        if total <= 0:
            raise RuntimeError("LMDB appears empty (0 entries). Aborting.")
        keys = list(range(total))
    else:
        keys = read_integer_keys(
            args.lmdb_path, print_count=args.print_keys, print_raw=args.print_raw, print_hex=args.print_hex
        )
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


