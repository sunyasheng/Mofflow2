"""
Find MOF IDs from LMDB based on IDs from JSON file
Usage: python debug_visualize_scripts/find_mof_ids.py [--json-path JSON_PATH] [--lmdb-path LMDB_PATH] [--num-ids NUM_IDS]
"""
import sys
import json
import argparse
sys.path.insert(0, '.')
from utils.lmdb import read_lmdb, get_data
import torch

parser = argparse.ArgumentParser(description='Find MOF IDs from LMDB based on JSON IDs')
parser.add_argument('--json-path', type=str, 
                    default='/home/suny0a/chem_root/MOFFlow-2/MOFFLOW2_data/seqs/mof_sequence_val.json',
                    help='Path to JSON file containing IDs')
parser.add_argument('--lmdb-path', type=str, 
                    default='/home/suny0a/chem_root/MOFFlow-2/MOFFLOW2_data/lmdb/MetalOxo.lmdb',
                    help='Path to LMDB file')
parser.add_argument('--num-ids', type=int, default=10,
                    help='Number of IDs to process from JSON')
args = parser.parse_args()

# Read JSON file
print(f"Reading JSON file: {args.json_path}")
with open(args.json_path, 'r') as f:
    json_data = json.load(f)

# Get IDs from JSON (first num_ids)
json_ids = list(json_data.keys())[:args.num_ids]
print(f"\nFound {len(json_ids)} IDs in JSON (processing first {args.num_ids}):")
print(json_ids)

# Open LMDB
print(f"\nOpening LMDB: {args.lmdb_path}")
env = read_lmdb(args.lmdb_path)

print(f"\n{'='*80}")
print(f"Looking up MOF IDs in LMDB:")
print(f"{'='*80}")

results = []

with env.begin() as txn:
    for json_id in json_ids:
        print(f"\n{'='*60}")
        print(f"Processing JSON ID: {json_id}")
        print(f"{'='*60}")
        
        try:
            # Try to get data using the JSON ID as key
            data = get_data(txn, json_id)
            
            print(f"Successfully retrieved data for ID: {json_id}")
            print(f"Data type: {type(data)}")
            
            # Try to find MOF ID - it should be in m_id attribute
            mof_id = None
            mof_id_source = None
            
            # Method 1: Check if data has m_id as attribute (primary method)
            if hasattr(data, 'm_id'):
                mof_id = data.m_id
                mof_id_source = 'data.m_id'
            
            # Method 2: If it's a dict, check dict keys
            elif isinstance(data, dict):
                if 'm_id' in data:
                    mof_id = data['m_id']
                    mof_id_source = 'dict[m_id]'
                elif 'mof_id' in data:
                    mof_id = data['mof_id']
                    mof_id_source = 'dict[mof_id]'
                elif 'id' in data:
                    mof_id = data['id']
                    mof_id_source = 'dict[id]'
            
            # Method 3: Check if prop_dict has m_id
            if mof_id is None and hasattr(data, 'prop_dict') and isinstance(data.prop_dict, dict):
                print(f"\nprop_dict keys: {list(data.prop_dict.keys())[:10]}")
                if 'm_id' in data.prop_dict:
                    mof_id = data.prop_dict['m_id']
                    mof_id_source = 'prop_dict[m_id]'
                elif 'mof_id' in data.prop_dict:
                    mof_id = data.prop_dict['mof_id']
                    mof_id_source = 'prop_dict[mof_id]'
            
            # Method 4: Use JSON ID as MOF ID if nothing else found
            if mof_id is None:
                mof_id = json_id
                mof_id_source = 'json_id (fallback)'
            
            print(f"\nMOF ID: {mof_id} (from: {mof_id_source})")
            
            # Store result
            results.append({
                'json_id': json_id,
                'mof_id': mof_id,
                'mof_id_source': mof_id_source,
                'found': True
            })
            
            # Show some additional info about the data
            if hasattr(data, 'num_atoms'):
                print(f"  num_atoms: {data.num_atoms}")
            if hasattr(data, 'prop_dict') and isinstance(data.prop_dict, dict):
                print(f"  prop_dict has {len(data.prop_dict)} keys")
            
        except Exception as e:
            print(f"Error processing ID {json_id}: {e}")
            results.append({
                'json_id': json_id,
                'mof_id': None,
                'mof_id_source': None,
                'found': False,
                'error': str(e)
            })

env.close()

# Print summary
print(f"\n{'='*80}")
print(f"Summary:")
print(f"{'='*80}")
print(f"\n{'JSON ID':<15} {'MOF ID':<20} {'Source':<25} {'Status'}")
print(f"{'-'*80}")
for r in results:
    status = '✓ Found' if r['found'] else '✗ Error'
    mof_id_str = str(r['mof_id']) if r['mof_id'] is not None else 'N/A'
    source_str = r['mof_id_source'] if r['mof_id_source'] else 'N/A'
    print(f"{r['json_id']:<15} {mof_id_str:<20} {source_str:<25} {status}")

