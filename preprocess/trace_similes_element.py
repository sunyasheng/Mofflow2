"""
Trace and view element by index in detail
Usage: python preprocess/trace_similes_element.py <index> [--lmdb-path LMDB_PATH]
"""
import sys
import argparse
sys.path.insert(0, '.')
from utils.lmdb import read_lmdb, get_data
import torch

parser = argparse.ArgumentParser(description='View element by index from LMDB')
parser.add_argument('index', type=int, help='Index of the element to view (e.g., 153135)')
parser.add_argument('--lmdb-path', type=str, 
                    default='/home/suny0a/chem_root/MOFFlow-2/MOFFLOW2_data/lmdb/MetalOxo.lmdb',
                    help='Path to LMDB file')
args = parser.parse_args()

lmdb_path = args.lmdb_path
index = args.index

env = read_lmdb(lmdb_path)
with env.begin() as txn:
    data = get_data(txn, str(index))
    
    print(f"="*60)
    print(f"Element {index} Details")
    print(f"="*60)
    print(f"\nType: {type(data)}")
    
    # Try to access data through torch_geometric's store mechanism
    if hasattr(data, 'keys'):
        print(f"\nData keys (via data.keys()): {list(data.keys())}")
    if hasattr(data, '_store'):
        print(f"Data store type: {type(data._store)}")
        if hasattr(data._store, '_mapping'):
            print(f"Store mapping keys: {list(data._store._mapping.keys()) if hasattr(data._store, '_mapping') else 'N/A'}")
    
    # Get ALL attributes (including private ones)
    all_attrs = [attr for attr in dir(data)]
    public_attrs = [attr for attr in all_attrs if not attr.startswith('_')]
    private_attrs = [attr for attr in all_attrs if attr.startswith('_')]
    
    print(f"\n{'='*60}")
    print(f"Public Attributes ({len(public_attrs)}):")
    print(f"{'='*60}")
    for attr in sorted(public_attrs):
        try:
            val = getattr(data, attr)
            if not callable(val):
                if isinstance(val, torch.Tensor):
                    print(f"  {attr}: Tensor shape={val.shape}, dtype={val.dtype}")
                    if val.numel() <= 5:
                        print(f"    Values: {val}")
                    else:
                        print(f"    Min: {val.min().item():.4f}, Max: {val.max().item():.4f}, Mean: {val.float().mean().item():.4f}")
                elif isinstance(val, (list, tuple)):
                    print(f"  {attr}: {type(val).__name__} length={len(val)}")
                    if len(val) <= 3:
                        print(f"    Values: {val}")
                    elif len(val) > 0:
                        print(f"    First element type: {type(val[0])}")
                        if isinstance(val[0], torch.Tensor):
                            print(f"    First element shape: {val[0].shape}")
                elif isinstance(val, dict):
                    print(f"  {attr}: dict with {len(val)} keys: {list(val.keys())[:10]}")
                    if len(val) <= 5:
                        for k, v in val.items():
                            print(f"    {k}: {v}")
                else:
                    print(f"  {attr}: {type(val).__name__} = {val}")
        except Exception as e:
            print(f"  {attr}: Error accessing - {e}")
    
    print(f"\n{'='*60}")
    print(f"Private Attributes ({len(private_attrs)}):")
    print(f"{'='*60}")
    for attr in sorted(private_attrs):
        try:
            val = getattr(data, attr)
            if not callable(val):
                if isinstance(val, torch.Tensor):
                    print(f"  {attr}: Tensor shape={val.shape}, dtype={val.dtype}")
                    if val.numel() > 5:
                        print(f"    Min: {val.min().item():.4f}, Max: {val.max().item():.4f}")
                elif isinstance(val, (list, tuple)):
                    print(f"  {attr}: {type(val).__name__} length={len(val)}")
                    if len(val) <= 3:
                        print(f"    Values: {val}")
                elif isinstance(val, dict):
                    print(f"  {attr}: dict with {len(val)} keys: {list(val.keys())[:5]}")
                else:
                    print(f"  {attr}: {type(val).__name__} = {val if not isinstance(val, (list, tuple)) or len(val) <= 3 else f'{type(val).__name__}({len(val)})'}")
        except Exception as e:
            print(f"  {attr}: Error accessing - {e}")
    
    # Show important properties
    print(f"\n{'='*60}")
    print("Key Properties:")
    print(f"{'='*60}")
    
    if hasattr(data, 'num_atoms'):
        print(f"num_atoms: {data.num_atoms}")
    if hasattr(data, 'num_components'):
        print(f"num_components: {data.num_components}")
    if hasattr(data, 'prop_dict'):
        print(f"prop_dict: {data.prop_dict}")
    if hasattr(data, 'y'):
        y = data.y
        print(f"y (properties tensor): shape={y.shape}, dtype={y.dtype}")
        if y.numel() <= 5:
            print(f"  Values: {y}")
        else:
            print(f"  Min: {y.min().item():.4f}, Max: {y.max().item():.4f}, Mean: {y.float().mean().item():.4f}")
    if hasattr(data, 'lengths'):
        lengths = data.lengths
        if isinstance(lengths, torch.Tensor):
            print(f"lengths: shape={lengths.shape}, values={lengths.tolist() if lengths.numel() <= 10 else f'Min={lengths.min():.4f}, Max={lengths.max():.4f}'}")
        else:
            print(f"lengths: {lengths}")
    if hasattr(data, 'angles'):
        angles = data.angles
        if isinstance(angles, torch.Tensor):
            print(f"angles: shape={angles.shape}, values={angles.tolist() if angles.numel() <= 10 else f'Min={angles.min():.4f}, Max={angles.max():.4f}'}")
        else:
            print(f"angles: {angles}")
    if hasattr(data, 'bbs'):
        print(f"bbs: {len(data.bbs)} building blocks")
        for i, bb in enumerate(data.bbs):
            print(f"  BB {i}: num_atoms={bb.num_atoms if hasattr(bb, 'num_atoms') else 'N/A'}")
    
    # Display all stored data via keys() if available
    print(f"\n{'='*60}")
    print("All Stored Data (via data access):")
    print(f"{'='*60}")
    try:
        if hasattr(data, 'keys'):
            keys = list(data.keys())
            print(f"Available keys: {keys}")
            for key in keys:
                try:
                    val = data[key]
                    if isinstance(val, torch.Tensor):
                        print(f"  {key}: Tensor shape={val.shape}, dtype={val.dtype}")
                        if val.numel() <= 5:
                            print(f"    Values: {val}")
                        else:
                            print(f"    Min: {val.min().item():.4f}, Max: {val.max().item():.4f}, Mean: {val.float().mean().item():.4f}")
                    elif isinstance(val, (list, tuple)):
                        print(f"  {key}: {type(val).__name__} length={len(val)}")
                        if len(val) <= 3:
                            print(f"    Values: {val}")
                        elif len(val) > 0:
                            print(f"    First element type: {type(val[0])}")
                    elif isinstance(val, dict):
                        print(f"  {key}: dict with keys: {list(val.keys())}")
                    else:
                        print(f"  {key}: {type(val).__name__} = {val}")
                except Exception as e:
                    print(f"  {key}: Error accessing - {e}")
    except Exception as e:
        print(f"Error accessing data keys: {e}")
    
env.close()

