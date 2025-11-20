#!/usr/bin/env python
"""
Convert sequence ID(s) to m_id from LMDB database
Usage: 
    python 2_seqID2mID.py <seq_id> [--lmdb-path LMDB_PATH]
    python 2_seqID2mID.py --input-file <file> [--output OUTPUT_FILE] [--lmdb-path LMDB_PATH]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.lmdb import read_lmdb, get_data


def convert_seq_id_to_m_id(seq_id: str, txn, verbose: bool = False) -> str:
    """Convert a single seq_id to m_id."""
    try:
        data = get_data(txn, seq_id)
        if hasattr(data, 'm_id'):
            return data.m_id
        else:
            if verbose:
                print(f"Warning: Data for ID {seq_id} does not have m_id attribute", file=sys.stderr)
            return None
    except Exception as e:
        if verbose:
            print(f"Error: Failed to retrieve data for ID {seq_id}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert sequence ID(s) to m_id from LMDB",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "seq_id",
        type=str,
        nargs='?',
        default=None,
        help="Single sequence ID to look up (e.g., 52387, 52999). Ignored if --input-file is provided.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to file containing sequence IDs (one per line). If provided, processes all IDs in the file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output file. If not provided, outputs to stdout.",
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

    # Determine input mode
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
        # Read seq_ids from file
        seq_ids = []
        with open(args.input_file, 'r') as f:
            for line in f:
                seq_id = line.strip()
                if seq_id:  # Skip empty lines
                    seq_ids.append(seq_id)
        
        if not seq_ids:
            print("Warning: No sequence IDs found in input file", file=sys.stderr)
            return
        
        # Batch processing
        env = read_lmdb(args.lmdb_path)
        results = []
        try:
            with env.begin(buffers=True) as txn:
                for seq_id in seq_ids:
                    m_id = convert_seq_id_to_m_id(seq_id, txn, verbose=True)
                    if m_id:
                        results.append((seq_id, m_id))
        finally:
            env.close()
        
        # Output results
        output_file = open(args.output, 'w') if args.output else sys.stdout
        try:
            for seq_id, m_id in results:
                output_file.write(f"{m_id}\n")
            if args.output:
                print(f"Successfully converted {len(results)}/{len(seq_ids)} sequence IDs to m_id")
                print(f"Results written to {args.output}")
        finally:
            if args.output:
                output_file.close()
    
    elif args.seq_id:
        # Single seq_id mode (backward compatibility)
        env = read_lmdb(args.lmdb_path)
        try:
            with env.begin(buffers=True) as txn:
                m_id = convert_seq_id_to_m_id(args.seq_id, txn, verbose=True)
                if m_id:
                    output_file = open(args.output, 'w') if args.output else sys.stdout
                    try:
                        output_file.write(f"{m_id}\n")
                    finally:
                        if args.output:
                            output_file.close()
                else:
                    print(f"Error: Failed to convert seq_id {args.seq_id} to m_id", file=sys.stderr)
                    sys.exit(1)
        finally:
            env.close()
    else:
        parser.error("Either provide a seq_id or use --input-file")


if __name__ == "__main__":
    main()

