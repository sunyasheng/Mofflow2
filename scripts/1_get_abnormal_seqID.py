#!/usr/bin/env python
"""
Read a JSON file and output all the IDs (keys) from it.
Usage: python 1_get_abnormal_seqID.py [--json-path JSON_PATH]
"""
import json
import argparse
import sys
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Read JSON file and output IDs")
    parser.add_argument(
        "--json-path",
        type=str,
        default="seqs/mof_sequence_train.json",
        help="Path to the JSON file to read",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mof_abnormal_id_train.txt",
        help="Path to the output file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.json_path, 'r') as f:
            data = json.load(f)
        
        # Extract all IDs (keys) from the JSON
        ids = list(data.keys())
        
        # Write each ID to the output file
        with open(args.output, 'w') as out_f:
            for seq_id in ids:
                out_f.write(f"{seq_id}\n")
        
        print(f"Successfully wrote {len(ids)} IDs to {args.output}")
            
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# python 1_get_abnormal_seqID.py --json-path /ibex/project/c2318/material_discovery/MOFFLOW2_data/seqs/mof_sequence_train.json --output mof_abnormal_id_train.txt
