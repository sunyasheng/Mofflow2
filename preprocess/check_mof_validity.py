"""
Check MOF validity. Filter out MOFs that do not meet the criteria.
"""
import os
import time
import json
import hydra
import pickle
import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pymatgen.core import Structure
from utils.lmdb import read_lmdb, write_lmdb
from utils.check_mof_validity import check_mof
from utils.environment import PROJECT_ROOT

# Export (TODO: before import)
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


class CheckMOF:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Task
        self.task = process_cfg.task # 'gen' or 'csp
        self.use_matched_coords = process_cfg.mof_checker.use_matched_coords

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus
    
    def process_one(self, idx, value):
        warnings.filterwarnings("ignore")
        try:
            feats = pickle.loads(value)

            if self.use_matched_coords:
                if feats['rmsd'][-1] is None:
                    return idx, None, 'rmsd_none'
                coords = feats['matched_coords'][-1]
            else:
                coords = feats['gt_coords']

            structure = Structure(
                lattice=feats['cell_1'],
                species=feats['atom_types'],
                coords=coords,
                coords_are_cartesian=True
            )
            _, valid = check_mof(structure)
            return (idx, value, 'success') if valid else (idx, None, 'invalid')
        except Exception:
            return idx, None, 'exception'
    
    def process(self, split="train"):
        print(f"Checking {split} split...")

        # Start timer
        start_time = time.time()

        # Set up base directory
        base_dir = f"{self.lmdb_dir}/{self.task}"

        # Determine src file
        if self.use_matched_coords:
            print("INFO:: Using matched coordinates")
            pattern = f"MetalOxo_matched_{split}_*.lmdb"
            self.num_prev_trial = len(list(Path(base_dir).glob(pattern)))
            src_env = read_lmdb(f"{base_dir}/MetalOxo_matched_{split}_{self.num_prev_trial}.lmdb")
        else:
            print("INFO:: Using ground truth coordinates")
            src_env = read_lmdb(f"{base_dir}/MetalOxo_feats_{split}.lmdb")

        # Read data
        data_dict = {}
        with src_env.begin() as src_txn:
            num_src_entries = src_env.stat()['entries']
            cursor = src_txn.cursor()
            for key_bytes, value in tqdm(cursor, desc="Reading data", total=num_src_entries):
                idx = int(key_bytes.decode('ascii'))
                data_dict[idx] = value
        src_env.close()

        # Process data
        results = Parallel(n_jobs=self.num_cpus)(delayed(self.process_one)(idx, value) for idx, value in tqdm(data_dict.items()))

        # Track failure counts
        filtered_list = []
        status_list = []

        for idx, value, status in tqdm(results, desc="Tracking failure counts"):
            filtered_list.append((idx, value))
            status_list.append(status)

        counters = Counter(status_list)
        print(f"INFO:: Failure counts: {dict(counters)}")

        # Save failure counts to JSON
        with open(f"{base_dir}/mofchecker_failure_counts_{split}.json", "w") as f:
            json.dump(counters, f, indent=2)

        # Write valid MOFs to LMDB
        dest_env = write_lmdb(f"{base_dir}/MetalOxo_mofchecker_{split}.lmdb")
        with dest_env.begin(write=True) as dest_txn:
            for idx, value in tqdm(filtered_list):
                if value is not None:
                    key_bytes = f"{idx}".encode('ascii')
                    dest_txn.put(key_bytes, value)
        num_dest_entries = dest_env.stat()['entries']
        print(f"INFO:: Remaining samples: {num_dest_entries}/{num_src_entries}")
        dest_env.close()
        
        # End timer
        print(f"INFO:: Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):
    checker = CheckMOF(cfg=cfg)

    # Set splits
    if cfg.preprocess.task == "gen":
        splits = ["train", "val"]
    elif cfg.preprocess.task == "csp":
        splits = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")
    
    # Process
    for split in splits[::-1]:
        checker.process(split=split)

if __name__ == "__main__":
    main()