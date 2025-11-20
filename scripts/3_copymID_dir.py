#!/usr/bin/env python
"""
Copy CIF files based on m_id list from source directory to local directory.
Usage: python 3_copymID_dir.py --input-file <m_id_list> [--source-dir SOURCE_DIR] [--output-dir OUTPUT_DIR]
"""
import argparse
import sys
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy CIF files based on m_id list",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to file containing m_id list (one per line)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/ibex/project/c2318/material_discovery/raw_DB2/raw/MOF_DB_2_0_hMOF_CIFs_18042025",
        help="Source directory containing CIF files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./abnormal_train",
        help="Output directory to copy CIF files to",
    )
    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # Check source directory
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory not found: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read m_id list
    m_ids = []
    with open(args.input_file, 'r') as f:
        for line in f:
            m_id = line.strip()
            if m_id:  # Skip empty lines
                m_ids.append(m_id)

    if not m_ids:
        print("Warning: No m_ids found in input file", file=sys.stderr)
        return

    print(f"Found {len(m_ids)} m_ids in input file")
    print(f"Source directory: {args.source_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Copy files
    copied_count = 0
    not_found_count = 0
    source_path = Path(args.source_dir)
    output_path = Path(args.output_dir)

    for m_id in m_ids:
        # Construct CIF filename (add .cif extension if not present)
        if m_id.endswith('.cif'):
            cif_filename = m_id
        else:
            cif_filename = f"{m_id}.cif"
        
        source_file = source_path / cif_filename
        dest_file = output_path / cif_filename

        if source_file.exists():
            try:
                shutil.copy2(source_file, dest_file)
                copied_count += 1
                if copied_count % 100 == 0:
                    print(f"Copied {copied_count} files...", end='\r')
            except Exception as e:
                print(f"Error copying {cif_filename}: {e}", file=sys.stderr)
        else:
            not_found_count += 1
            print(f"Warning: File not found: {source_file}", file=sys.stderr)

    print()
    print(f"Successfully copied {copied_count}/{len(m_ids)} files")
    if not_found_count > 0:
        print(f"Warning: {not_found_count} files were not found in source directory", file=sys.stderr)


if __name__ == "__main__":
    main()

# python 3_copymID_dir.py --input-file mof_abnormal_m_id_train.txt --source-dir /ibex/project/c2318/material_discovery/raw_DB2/raw/MOF_DB_2_0_hMOF_CIFs_18042025 --output-dir ./abnormal_train
# python 3_copymID_dir.py --input-file mof_abnormal_m_id_val.txt --source-dir /ibex/project/c2318/material_discovery/raw_DB2/raw/MOF_DB_2_0_hMOF_CIFs_18042025 --output-dir ./abnormal_val

