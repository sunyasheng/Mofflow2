"""
Filter MOFs and BBs by size (code adapted from MOFDiff)
"""
import time
import hydra
import pickle
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed
from omegaconf import DictConfig
from utils.lmdb import read_lmdb, write_lmdb
from utils.environment import PROJECT_ROOT
from mofdiff.common.data_utils import (
    lattice_params_to_matrix_torch,
    frac_to_cart_coords
)
from mofdiff.common.atomic_utils import (
    frac2cart,
    compute_distance_matrix,
)


def mof_criterion(mof, max_bbs=20):
    try:
        if (
            mof.num_components > max_bbs
            or mof.y.isnan().sum() > 0
            or mof.y.isinf().sum() > 0
        ):
            return False
        cell = lattice_params_to_matrix_torch(mof.lengths, mof.angles).squeeze()
        distances = compute_distance_matrix(
            cell, frac2cart(mof.cg_frac_coords, cell)
        ).fill_diagonal_(5.0)
        return (
            (not (distances < 1.0).any())
            and mof.num_components <= max_bbs
            and mof.y.isnan().sum() == 0
            and mof.y.isinf().sum() == 0
        )
    except Exception:
        return False

def bb_criterion(bb, max_atoms=200, max_cps=20):
    try:
        bb.num_cps = bb.is_anchor.long().sum()
        if (bb.num_atoms > max_atoms) or (bb.num_cps > max_cps):
            return None, False

        cart_coords = frac_to_cart_coords(
            bb.frac_coords, bb.lengths, bb.angles, bb.num_atoms
        )
        pdist = torch.cdist(cart_coords, cart_coords).fill_diagonal_(5.0)

        # detect BBs with problematic bond info.
        edge_index = bb.edge_index
        j, i = edge_index
        bond_dist = (cart_coords[i] - cart_coords[j]).pow(2).sum(dim=-1).sqrt()

        success = (
            pdist.min() > 0.25
            and bond_dist.max() < 5.0
            and (bb.num_atoms <= max_atoms)
            and (bb.num_cps <= max_cps)
        )
        return success
    except Exception:
        return False

def process_one(idx, value, max_bbs=20, max_atoms=200, max_cps=20, prop_list=None):
    # Load data
    data = pickle.loads(value)

    # Add properties
    data['y'] = torch.tensor([data.prop_dict[prop] for prop in prop_list], dtype=torch.float32).view(1, -1)

    # Check criteria
    if mof_criterion(data, max_bbs) and all(bb_criterion(bb, max_atoms, max_cps) for bb in data.bbs):
        return idx, value
    else:
        return idx, None


class FilterMOF:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess
        filter_cfg = cfg.preprocess.filter

        # Task
        self.task = process_cfg.task # 'gen' or 'csp'

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir
        self.split_dir = process_cfg.split_dir

        # Filter parameters
        self.max_bbs = filter_cfg.max_bbs
        self.max_atoms = filter_cfg.max_atoms
        self.max_cps = filter_cfg.max_cps

        # Properties
        self.prop_list = filter_cfg.prop_list

        # Number of CPUs
        self.num_cpus = filter_cfg.num_cpus

    def process(self, split="train"):
        print(f"Filtering {split} split for task {self.task}...")
        
        # Start timer
        start_time = time.time()

        # Load split indices
        split_file = f"{self.split_dir}/{self.task}/{split}_split.txt"
        split_idx = np.loadtxt(split_file, dtype=int)

        # Read data
        data_dict = {}
        src_env = read_lmdb(f"{self.lmdb_dir}/MetalOxo.lmdb")
        with src_env.begin() as src_txn:
            for idx in tqdm(split_idx, desc="Reading data"):
                key_bytes = f"{idx}".encode('ascii')
                value = src_txn.get(key_bytes)
                if value is None:
                    print(f"WARNING:: Index {idx} not found")
                    continue
                data_dict[idx] = value
        src_env.close()

        # Filter data
        process_one_partial = partial(
            process_one, 
            max_bbs=self.max_bbs, 
            max_atoms=self.max_atoms, 
            max_cps=self.max_cps, 
            prop_list=self.prop_list
        )
        filtered_list = Parallel(n_jobs=self.num_cpus)(delayed(process_one_partial)(idx, value) for idx, value in tqdm(data_dict.items()))

        # Write filtered data
        dest_env = write_lmdb(f"{self.lmdb_dir}/{self.task}/MetalOxo_filtered_{split}.lmdb")
        with dest_env.begin(write=True) as dest_txn:
            for idx, value in tqdm(filtered_list, desc="Writing data"):
                if value is not None:
                    key_bytes = f"{idx}".encode('ascii')
                    dest_txn.put(key_bytes, value)
        
        num_dest_entries = dest_env.stat()['entries']
        print(f"INFO:: Remaining samples: {num_dest_entries}/{len(split_idx)}")
        dest_env.close()

        # End timer
        print(f"INFO:: Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):
    filter = FilterMOF(cfg=cfg)

    # Set splits
    if cfg.preprocess.task == "gen":
        splits = ["train", "val"]
    elif cfg.preprocess.task == "csp":
        splits = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")
    
    # Process
    for split in splits[::-1]:
        filter.process(split=split)

if __name__ == "__main__":
    main()