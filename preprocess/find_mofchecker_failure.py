"""
Check MOF validity. Filter out MOFs that do not meet the criteria.
"""
import os
import sys
# 保证从项目根可 import utils（用 python preprocess/find_mofchecker_failure.py 时）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import hydra
import pickle
import warnings
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pymatgen.core import Structure
from utils.lmdb import read_lmdb, write_lmdb
from utils.check_mof_validity import check_mof
import utils.check_mof_validity as _check_mof_module
from utils.environment import PROJECT_ROOT

# Export (TODO: before import)
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '64'


def process_one(idx, value):
    # Disable warnings within each process
    warnings.filterwarnings("ignore")

    try:
        feats = pickle.loads(value)
        structure = Structure(
            lattice=feats['cell_1'],
            species=feats['atom_types'],
            coords=feats['gt_coords'],
            coords_are_cartesian=True
        )
        desc, valid = check_mof(structure)
        return (desc, feats) if not valid else None
    except Exception:
        return None

def process_matched_one(idx, value): # TODO: remove
    # Disable warnings within each process
    warnings.filterwarnings("ignore")

    try:
        feats = pickle.loads(value)
        if feats['rmsd'][-1] is None:
            return None
        structure = Structure(
            lattice=feats['cell_1'],
            species=feats['atom_types'],
            coords=feats['matched_coords'][-1],
            coords_are_cartesian=True
        )
        desc, valid = check_mof(structure)
        return (desc, feats) if not valid else None
    except Exception:
        return None

class CheckMOF:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Task
        self.task = process_cfg.task  # 'gen' or 'csp'

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir
        self.split_dir = process_cfg.split_dir
        self.data_dir = cfg.paths.data_dir

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus

    def process(self, split="train"):
        print(f"utils.check_mof_validity __file__: {_check_mof_module.__file__}")
        print(f"Checking {split} split...")

        # Start timer
        start_time = time.time()

        # Set up base directory (same as check_mof_validity.py)
        base_dir = f"{self.lmdb_dir}/{self.task}"

        # Load split indices
        split_file = f"{self.split_dir}/{self.task}/{split}_split.txt"
        if not os.path.exists(split_file):
            # Fallback to old path structure
            split_file = f"{self.split_dir}/{split}_split.txt"
        split_idx = np.loadtxt(split_file, dtype=int)

        # Read data
        data_dict = {}
        # src_env = read_lmdb(f"{base_dir}/MetalOxo_feats_{split}.lmdb")
        src_env = read_lmdb(f"{base_dir}/MetalOxo_matched_{split}_3.lmdb")  # TODO: Remove
        with src_env.begin() as src_txn:
            for idx in tqdm(split_idx, desc="Reading data"):
                key_bytes = f"{idx}".encode('ascii')
                value = src_txn.get(key_bytes)
                if value is None:
                    continue
                data_dict[idx] = value
        src_env.close()

        # Process data
        filtered_list = Parallel(n_jobs=self.num_cpus)(delayed(process_matched_one)(idx, value) for idx, value in tqdm(data_dict.items()))
        filtered_list = [item for item in filtered_list if item is not None]

        # Write extracted failures with pickle
        output_file = f"{base_dir}/invalid_mofs_matched_{split}.pkl"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "wb") as f:  # TODO: remove matched
            pickle.dump(filtered_list, f)
        print(f"INFO:: Saved {len(filtered_list)} invalid MOFs to {output_file}")
        
        # End timer
        print(f"INFO:: Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):
    checker = CheckMOF(cfg=cfg)
    checker.process(split="train")
    checker.process(split="val")


if __name__ == "__main__":
    main()