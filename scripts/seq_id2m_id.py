#!/usr/bin/env python
"""
Convert sequence ID to m_id from LMDB database
Usage: python seq_id2m_id.py <seq_id> [--lmdb-path LMDB_PATH]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.lmdb import read_lmdb, get_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert sequence ID to m_id from LMDB")
    parser.add_argument(
        "seq_id",
        type=str,
        help="Sequence ID to look up (e.g., 52387, 52999)",
    )
    parser.add_argument(
        "--lmdb-path",
        type=str,
        default="/home/suny0a/chem_root/MOFFLOW2_data/lmdb/MetalOxo.lmdb",
        help="Path to MetalOxo.lmdb",
    )
    args = parser.parse_args()

    if not os.path.exists(args.lmdb_path):
        raise FileNotFoundError(f"LMDB file not found: {args.lmdb_path}")

    env = read_lmdb(args.lmdb_path)
    try:
        with env.begin(buffers=True) as txn:
            data = get_data(txn, args.seq_id)
            if hasattr(data, 'm_id'):
                print(data.m_id)
            else:
                print(f"Error: Data for ID {args.seq_id} does not have m_id attribute", file=sys.stderr)
                sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to retrieve data for ID {args.seq_id}: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()

