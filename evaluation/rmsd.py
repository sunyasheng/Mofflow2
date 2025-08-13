"""
CSP: Compute RMSD between predicted and ground truth MOF structures. 
"""
import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
from torch_geometric.data import Batch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher


def merge_prediction_files(pt_files):
    """
    Merges multiple rank-specific .pt prediction files into a single dictionary.
    """
    all_cart_coords = []
    all_num_atoms = []
    all_atom_types = []
    all_lattices = []
    all_gt_data = []

    for f in tqdm(pt_files, desc="Loading prediction files"):
        data = torch.load(f, map_location='cpu')

        all_cart_coords.append(data["cart_coords"])  # [k, num_atoms, 3]
        all_num_atoms.append(data["num_atoms"])      # [k, num_graphs]
        all_atom_types.append(data["atom_types"])    # [k, num_atoms]
        all_lattices.append(data["lattices"])        # [k, num_graphs, 6]

        gt_data = data["gt_data_batch"].to_data_list()
        all_gt_data.extend(gt_data)

    merged = {
        "cart_coords": torch.cat(all_cart_coords, dim=1),   # [k, total_atoms, 3]
        "num_atoms": torch.cat(all_num_atoms, dim=1),       # [k, total_graphs]
        "atom_types": torch.cat(all_atom_types, dim=1),     # [k, total_atoms]
        "lattices": torch.cat(all_lattices, dim=1),         # [k, total_graphs, 6]
        "gt_data_batch": Batch.from_data_list(all_gt_data, exclude_keys=["rotable_atom_mask"])
    }

    return merged

def split_data(cart_coords, atom_types, lattices, num_atoms):
    """
    Splits concatenated data into a list of samples.
    Args:
        cart_coords (torch.Tensor): [num_atoms, 3]
        atom_types (torch.Tensor): [num_atoms]
        lattices (torch.Tensor): [num_samples, 3, 3]
        num_atoms (list): [num_samples]
            each element is the number of atoms in a sample
    Returns:
        split_list (list): [num_samples]
            each element is a dict with keys 'cart_coords', 'atom_types', 'lattice'
    """
    split_list = []
    
    cart_coords = cart_coords.split(num_atoms)
    atom_types = atom_types.split(num_atoms)
    
    for coords, atoms, lattice in zip(cart_coords, atom_types, lattices):
        split_list.append({
            'cart_coords': coords,
            'atom_types': atoms,
            'lattice': lattice
        })
    
    return split_list


def process_one(gt, pred, matcher):
    """
    Args:
        gt (dict): keys 'cart_coords', 'atom_types', 'lattice'
        pred (dict): keys 'cart_coords', 'atom_types', 'lattice'
    Returns:
        rmsd (float): RMSD between the two structures
    """
    try:
        gt_structure = Structure(
            lattice=Lattice.from_parameters(*gt['lattice']),
            species=gt['atom_types'],
            coords=gt['cart_coords'],
            coords_are_cartesian=True
        )
        pred['lattice'][3:] = torch.clip(pred['lattice'][3:], 0.0, 179.0) # Clip lattice angles to valid range
        pred_structure = Structure(
            lattice=Lattice.from_parameters(*pred['lattice']),
            species=pred['atom_types'],
            coords=pred['cart_coords'],
            coords_are_cartesian=True
        )
        
        rmsd = matcher.get_rms_dist(gt_structure, pred_structure)
        rmsd = rmsd if rmsd is None else rmsd[0]
        
        return rmsd
    except Exception as e:
        print(f"Error processing structures: {e}")
        return None

def main(args):
    # Find all .pt files
    pt_files = sorted(glob.glob(args.save_pt))
    assert pt_files, f"No files matched pattern {args.save_pt}"

    # Merge
    results = merge_prediction_files(pt_files)
    
    # Prepare ground truth data
    gt_batch = results['gt_data_batch'].to('cpu')
    gt_list = split_data(
        gt_batch['gt_coords'], # Use gt_coords
        gt_batch['atom_types'],
        gt_batch['lattice_1'],
        gt_batch['num_atoms'].tolist()
    )
    
    # Prepare predicted data
    pred_lists = []
    if args.num_samples is None:
        args.num_samples = results['cart_coords'].shape[0]
    for k in range(args.num_samples):
        pred_list = split_data(
            results['cart_coords'][k],
            results['atom_types'][k],
            results['lattices'][k],
            results['num_atoms'][k].tolist()
        )
        pred_lists.append(pred_list)
    
    # Compute metrics
    matcher = StructureMatcher(stol=args.stol, ltol=args.ltol, angle_tol=args.angle_tol)
    rmsd_df = pd.DataFrame(columns=[k for k in range(args.num_samples)])
    for k in range(args.num_samples):
        rmsd_list = Parallel(n_jobs=args.num_cpus)(
            delayed(process_one)(gt, pred, matcher=matcher)
            for gt, pred in tqdm(zip(gt_list, pred_lists[k]), total=len(gt_list), desc="Computing RMSD")
        )
        rmsd_df[k] = rmsd_list
    
    rmsd_df['min'] = rmsd_df.min(axis=1)
    
    # Compute summary
    summary = {
        'Match rate (%)': len(rmsd_df['min'].dropna()) / len(rmsd_df['min']) * 100,
        'Mean RMSE': rmsd_df['min'].mean(),
        'Std RMSE': rmsd_df['min'].std()
    }
    print(summary)

    # Save the results
    save_dir = Path(pt_files[0]).parent
    rmsd_df.to_csv(save_dir / f'rmsd_stol-{args.stol}.csv', index=False)
    with open(save_dir / f'summary_stol-{args.stol}.json', 'w') as f:
        json.dump(summary, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_pt', type=str, required=True) # "PATH/TO/*.pt"
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--stol', type=float, default=0.5)
    parser.add_argument('--ltol', type=float, default=0.3)
    parser.add_argument('--angle_tol', type=float, default=10)
    args = parser.parse_args()
    main(args)