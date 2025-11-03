#!/usr/bin/env python3
"""Script to read and inspect metal_lib_train.pkl file"""

import pickle
import sys
from pathlib import Path

# Get file path from command line argument
if len(sys.argv) < 2:
    print("Usage: python read_metal_lib.py <path_to_pkl_file>")
    print("Example: python read_metal_lib.py /home/suny0a/chem_root/MOFFlow-2/MOFFLOW2_data/metals/gen/metal_lib_train.pkl")
    sys.exit(1)

file_path = sys.argv[1]

if not Path(file_path).exists():
    print(f"Error: File not found: {file_path}")
    sys.exit(1)

print(f"Reading from: {file_path}")
with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("\n" + "="*60)
print("DATA STRUCTURE ANALYSIS")
print("="*60)
print(f"Top-level type: {type(data)}")
print(f"Top-level type name: {type(data).__name__}")

if isinstance(data, dict):
    print(f"Number of top-level keys: {len(data)}")
    print(f"Top-level keys: {list(data.keys())}")
    
    # Based on code analysis, metal_lib_train.pkl should have 'metal_bb_library' key
    if 'metal_bb_library' in data:
        metal_bb_library = data['metal_bb_library']
        print("\n" + "-"*60)
        print("METAL BB LIBRARY ANALYSIS")
        print("-"*60)
        print(f"metal_bb_library type: {type(metal_bb_library)}")
        
        if isinstance(metal_bb_library, dict):
            print(f"Number of metal building blocks: {len(metal_bb_library)}")
            
            # Get all metal SMILES (keys)
            metal_smiles = list(metal_bb_library.keys())
            print(f"\nTotal unique metal building blocks: {len(metal_smiles)}")
            
            # Show first 20 metal SMILES
            print(f"\nFirst 20 metal building blocks (SMILES):")
            for i, smiles in enumerate(metal_smiles[:20], 1):
                print(f"  {i}. {smiles}")
            
            # Show sample value structure
            if metal_smiles:
                sample_smiles = metal_smiles[0]
                sample_value = metal_bb_library[sample_smiles]
                print(f"\nSample metal BB structure:")
                print(f"  Key (SMILES): {sample_smiles}")
                print(f"  Value type: {type(sample_value)}")
                if hasattr(sample_value, '__dict__'):
                    print(f"  Value attributes: {list(vars(sample_value).keys())}")
                elif isinstance(sample_value, dict):
                    print(f"  Value keys: {list(sample_value.keys())[:10]}")
            
            # Show all metals (sorted)
            print(f"\n" + "-"*60)
            print(f"ALL METAL BUILDING BLOCKS ({len(metal_smiles)} total):")
            print("-"*60)
            for i, smiles in enumerate(sorted(metal_smiles), 1):
                print(f"{i}. {smiles}")
            
        elif isinstance(metal_bb_library, (list, tuple, set)):
            metals = list(metal_bb_library) if not isinstance(metal_bb_library, set) else sorted(list(metal_bb_library))
            print(f"Number of metal building blocks: {len(metals)}")
            print(f"\nFirst 20 metal building blocks:")
            for i, metal in enumerate(metals[:20], 1):
                print(f"  {i}. {metal}")
            
            print(f"\n" + "-"*60)
            print(f"ALL METAL BUILDING BLOCKS ({len(metals)} total):")
            print("-"*60)
            for i, metal in enumerate(sorted(metals) if not isinstance(metal_bb_library, set) else metals, 1):
                print(f"{i}. {metal}")
        else:
            print(f"Unexpected type for metal_bb_library: {type(metal_bb_library)}")
            print(f"Content preview: {str(metal_bb_library)[:500]}")
    
    # Show other keys if any
    other_keys = [k for k in data.keys() if k != 'metal_bb_library']
    if other_keys:
        print(f"\nOther keys in pickle file: {other_keys}")
        for key in other_keys:
            print(f"  {key}: {type(data[key])}")

else:
    print(f"Unexpected top-level type. Content preview:")
    print(str(data)[:1000])

print("\n" + "="*60)
