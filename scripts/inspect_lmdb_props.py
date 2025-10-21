import argparse
import os
from collections import Counter, defaultdict

import pickle
from tqdm import tqdm

from utils.lmdb import read_lmdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect prop_dict keys in MetalOxo LMDB and show frequencies")
    parser.add_argument(
        "--lmdb-path",
        required=True,
        help="Path to MetalOxo.lmdb (e.g., /path/to/MetalOxo.lmdb)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of entries to scan",
    )
    args = parser.parse_args()

    if not os.path.exists(args.lmdb_path):
        raise FileNotFoundError(args.lmdb_path)

    env = read_lmdb(args.lmdb_path)
    key_count = 0
    freq = Counter()
    examples = defaultdict(list)

    with env.begin() as txn:
        cursor = txn.cursor()
        for key_bytes, value in tqdm(cursor, desc="Scanning LMDB", total=env.stat().get("entries", None)):
            try:
                data = pickle.loads(value)
            except Exception:
                continue

            # Expect data.prop_dict to be a dict-like mapping of property name -> value
            prop_dict = getattr(data, 'prop_dict', None)
            if prop_dict is None and isinstance(data, dict):
                prop_dict = data.get('prop_dict', None)

            if isinstance(prop_dict, dict):
                for k, v in prop_dict.items():
                    freq[k] += 1
                    if len(examples[k]) < 3:
                        examples[k].append(v)

            key_count += 1
            if args.limit is not None and key_count >= args.limit:
                break

    env.close()

    print("\nFound property keys and their frequencies (count of entries containing the key):")
    for k, c in freq.most_common():
        ex = examples.get(k, [])
        print(f"- {k}: count={c}, examples={ex}")

    print(f"\nScanned entries: {key_count}")


if __name__ == "__main__":
    main()
