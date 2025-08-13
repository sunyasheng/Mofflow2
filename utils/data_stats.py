"""
Script for extracting dataset statistics (as csv) from the lmdb dataset.
"""
import os
import time
import gzip
import hydra
import pickle
import torch
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from omegaconf import DictConfig
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from openbabel import pybel
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from torch_geometric.utils import scatter
from utils.lmdb import read_lmdb, write_lmdb
from utils import data as du
from utils import molecule as mu
from utils.environment import PROJECT_ROOT


class GetStatistics:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Task
        self.task = process_cfg.task  # 'gen' or 'csp'

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir
        self.stats_dir = Path(f'{cfg.paths.data_dir}/stats/{self.task}')
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus  
    
    def get_structure(self, feats):
        return Structure(
            lattice=Lattice.from_parameters(*feats['lattice_1']),
            species=feats['atom_types'],
            coords=feats['matched_coords'][-1],
            coords_are_cartesian=True
        )
    
    def process_one(self, idx, value):
        try:
            feats = pickle.loads(value)
            structure = self.get_structure(feats)
            return {
                "n_elements": len(structure.elements),
                "n_sites": structure.num_sites,
                "volume": structure.volume,
                "density": structure.density,
                "lattice_a": structure.lattice.a,
                "lattice_b": structure.lattice.b,
                "lattice_c": structure.lattice.c,
                "lattice_alpha": structure.lattice.alpha,
                "lattice_beta": structure.lattice.beta,
                "lattice_gamma": structure.lattice.gamma,
            }
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None
    
    def process(self, split="train"):
        print(f"Checking {split} split...")

        # Start timer
        start_time = time.time()

        # Set directories
        src_dir = f"{self.lmdb_dir}/{self.task}"
        src_path = f"{src_dir}/MetalOxo_final_{split}.lmdb"
        dest_path = self.stats_dir / f"{split}.csv"

        # Read data
        data_dict = {}
        src_env = read_lmdb(src_path)
        with src_env.begin() as src_txn:
            num_src_entries = src_env.stat()['entries']
            cursor = src_txn.cursor()
            for key_bytes, value in tqdm(cursor, desc="Reading data", total=num_src_entries):
                idx = int(key_bytes.decode('ascii'))
                data_dict[idx] = value
        src_env.close()

        # Process data
        dict_list = Parallel(n_jobs=self.num_cpus)(delayed(self.process_one)(idx, value) for idx, value in tqdm(data_dict.items()))
        dict_list = [d for d in dict_list if d is not None] # Filter out None values
        print(f"INFO:: Number of computed samples: {len(dict_list)}/{num_src_entries}")

        # Write as csv
        df = pd.DataFrame(dict_list)
        df.to_csv(dest_path, index=False)
        print(f"INFO:: Wrote {len(dict_list)} samples to {dest_path}")
        
        # End timer
        print(f"INFO:: Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):
    stats = GetStatistics(cfg=cfg)

    # Set splits
    if cfg.preprocess.task == "gen":
        splits = ["train", "val"]
    elif cfg.preprocess.task == "csp":
        splits = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")
    
    # Process
    for split in splits[::-1]:
        stats.process(split=split)

if __name__ == "__main__":
    main()