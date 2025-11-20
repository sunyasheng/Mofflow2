#!/usr/bin/env python
"""
Find sequence IDs that contain abnormal oxidation states in SMILES sequences.
Usage: python 1_get_abnormal_seqID.py [--json-path JSON_PATH] [--output OUTPUT_FILE]
"""
import json
import argparse
import sys
import os
import re


# Define abnormal oxidation states: (element, charge)
ABNORMAL_CHARGES = [
    ('Cu', '+3'), ('Cu', '+4'), ('Cu', '+5'), ('Cu', '+6'), ('Cu', '+7'), ('Cu', '+8'),
    ('Zn', '+3'), ('Zn', '+4'), ('Zn', '+5'), ('Zn', '+6'), ('Zn', '+7'), ('Zn', '+8'),
    ('Pd', '+3'),
    ('V', '+6'),
    ('Ce', '+6'),
    ('Er', '+6'),
    ('Al', '+9'), ('Al', '+2'), ('Al', '+15'),
]


def contains_abnormal_charge(seq: str) -> bool:
    """
    Check if a sequence contains any abnormal oxidation states.
    """
    for element, charge in ABNORMAL_CHARGES:
        # Match patterns like [Cu+3], [Zn+4], etc.
        pattern = rf'\[{re.escape(element)}{re.escape(charge)}\]'
        if re.search(pattern, seq):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find sequence IDs containing abnormal oxidation states"
    )
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
        
        # Find IDs with abnormal charges
        abnormal_ids = []
        for seq_id, entry in data.items():
            seq = entry.get('seq', '')
            if contains_abnormal_charge(seq):
                abnormal_ids.append(seq_id)
        
        # Write abnormal IDs to the output file
        with open(args.output, 'w') as out_f:
            for seq_id in abnormal_ids:
                out_f.write(f"{seq_id}\n")
        
        print(f"Found {len(abnormal_ids)} sequences with abnormal oxidation states")
        print(f"Successfully wrote {len(abnormal_ids)} IDs to {args.output}")
            
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# python 1_get_abnormal_seqID.py --json-path /ibex/project/c2318/material_discovery/MOFFLOW2_data/seqs/mof_sequence_train.json --output mof_abnormal_id_train.txt
# Found 968 sequences with abnormal oxidation states
# Successfully wrote 968 IDs to mof_abnormal_id_train.txt

# python 1_get_abnormal_seqID.py --json-path /ibex/project/c2318/material_discovery/MOFFLOW2_data/seqs/mof_sequence_val.json --output mof_abnormal_id_val.txt
# Found 112 sequences with abnormal oxidation states
# Successfully wrote 112 IDs to mof_abnormal_id_val.txt
