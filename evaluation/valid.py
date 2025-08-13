import os
import re
import glob
import warnings
import pandas as pd
from pathlib import Path
from collections import defaultdict
from p_tqdm import p_map
from utils.environment import timeout
import argparse

import torch

from mofchecker import MOFChecker

EXPECTED_CHECK_VALUES = {
    "has_carbon": True,
    "has_hydrogen": True,
    "has_atomic_overlaps": False,
    "has_overcoordinated_c": False,
    "has_overcoordinated_n": False,
    "has_overcoordinated_h": False,
    "has_undercoordinated_c": False,
    "has_undercoordinated_n": False,
    "has_undercoordinated_rare_earth": False,
    "has_metal": True,
    "has_lone_molecule": False,
    "has_high_charges": False,
    "is_porous": True,
    "has_suspicicious_terminal_oxo": False,
    "has_undercoordinated_alkali_alkaline": False,
    "has_geometrically_exposed_metal": False,
    # "has_3d_connected_graph": True    # MOFDiff does not have this check
}

def check_criteria(descriptors: dict, expected_values: dict, verbose=False):
    """
    Returns:
    - True if all expected values match the descriptors
    - False if any expected value does not match the descriptors
    """
    for key, expected_value in expected_values.items():
        if descriptors[key] != expected_value:
            if verbose: print(f"Mismatch found for {key}: expected {expected_value}, found {descriptors[key]}")
            return False, key
    return True, None

def process_cif_file(mof_path):
    mofchecker = MOFChecker.from_cif(mof_path)
    descriptors = mofchecker.get_mof_descriptors()
    valid, info = check_criteria(descriptors, EXPECTED_CHECK_VALUES)
    return valid, info

@timeout(120)
def process_one(mof_path):
    # Silent warnings
    warnings.filterwarnings("ignore")
    try:
        result = process_cif_file(mof_path)
        return result
    except TimeoutError:
        return (False, "Timeout")
    except Exception as e:
        return (False, str(e))
        
def main(cif_path, num_cpus, csv_path):
    cif_files = sorted(glob.glob(cif_path))
    print(f"INFO:: Found {len(cif_files)} cif files")
    assert cif_files, f"No files matched pattern {cif_path}"
    
    results = defaultdict(list)
        
    # Process files in parallel
    p_results = p_map(process_one, cif_files, num_cpus=num_cpus)
    
    # Separate results into valid and info
    for valid, info in p_results:
        results["valid"].append(valid)
        results["info"].append(info)
    
    # Compute statistics
    num_valid = sum(results["valid"])
    print(f"Percentage of valid MOFs: {num_valid / len(results['valid']) * 100}%")
    
    # Save results
    cif_dir = Path(cif_path).parent
    print("INFO:: Saving results to", os.path.join(cif_dir, "valid_check.pt"))
    torch.save(results, os.path.join(cif_dir, "valid_check.pt"))

    # Save results to CSV
    indices = [int(Path(f).stem.split('_')[1]) for f in cif_files]

    df = pd.read_csv(csv_path, index_col=None)
    df_tmp = pd.DataFrame({'idx': indices, 'mofchecker': results["valid"]})
    df = df.join(df_tmp.set_index('idx'), how='left')

    # Final evaluation
    df['final_valid'] = df['valid_smi'] & df['mofchecker']
    df['final_vnu'] = df['valid_smi'] & df['novelty'] & df['uniqueness'] & df['mofchecker']
    df['final_novel_bb'] = df['valid_smi'] & df['mofchecker'] & df['novel_bb']

    # Print final statistics
    print(f"Final valid: {df['final_valid'].sum()}/{len(df)} ({df['final_valid'].mean() * 100:.2f}%)")
    print(f"Final VNU:   {df['final_vnu'].sum()}/{len(df)} ({df['final_vnu'].mean() * 100:.2f}%)")
    print(f"Final novel bb: {df['final_novel_bb'].sum()}/{len(df)} ({df['final_novel_bb'].mean() * 100:.2f}%)")

    csv_dir = Path(csv_path).parent
    df.to_csv(csv_dir / "final_vnu.csv", index=False)
    print("INFO:: Saving results to", os.path.join(csv_dir, "final_vnu.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif_path', type=str, default="PATH/cif/*.cif")
    parser.add_argument('--num_cpus', type=int, default=100)
    parser.add_argument('--csv_path', type=str, default="PATH/inference/vnu_seq_only.csv") # vnu results for seq only (from evaluate_seq.py)
    
    args = parser.parse_args()
    main(cif_path=args.cif_path, num_cpus=args.num_cpus, csv_path=args.csv_path)
