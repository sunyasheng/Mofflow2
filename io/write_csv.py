"""
Code for creating CSV files from processed LMDB files (for training general CSP models).
"""
import os
import time
import hydra
import pickle
import torch
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
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


class CSVConverter:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Task
        self.task = process_cfg.task  # 'gen' or 'csp'

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir
        self.csv_dir = f'{cfg.paths.data_dir}/csv/{self.task}'
        os.makedirs(self.csv_dir, exist_ok=True)

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus  

    def process_one(self, idx, value):
        """
        Converts a single LMDB entry to a CSV row with the following columns:
        - material_id: The index of the entry in the LMDB file.
        - cif: The CIF string of the material.
        """
        try:
            feats = pickle.loads(value)

            # Extract material ID
            material_id = idx

            # Convert structure to CIF
            structure = Structure(
                lattice=feats['cell_1'],
                species=feats['atom_types'],
                coords=feats['gt_coords'], # Use gt_coords
                coords_are_cartesian=True
            )
            cif = str(CifWriter(structure))

            # Create a dictionary for the CSV row
            row = {
                "material_id": material_id,
                "cif": cif,
            }
            return row
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None
    
    def process(self, split="train"):
        print(f"Checking {split} split...")

        # Start timer
        start_time = time.time()

        # Set directories
        base_dir = f"{self.lmdb_dir}/{self.task}"
        src_path = f"{base_dir}/MetalOxo_mofchecker_{split}.lmdb"
        dest_path = f"{self.csv_dir}/{split}.csv"

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
        feats_list = Parallel(n_jobs=self.num_cpus)(delayed(self.process_one)(idx, value) for idx, value in tqdm(data_dict.items()))
        feats_list = [row for row in feats_list if row is not None]  # Filter out None values
        print(f"INFO:: Number of valid samples: {len(feats_list)}/{num_src_entries}")

        # Write to csv
        df = pd.DataFrame(feats_list)
        df.to_csv(dest_path, index=False)
        print(f"INFO:: CSV file saved to {dest_path}")
        
        # End timer
        print(f"INFO:: Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):
    converter = CSVConverter(cfg=cfg)

    # Set splits
    if cfg.preprocess.task == "gen":
        splits = ["train", "val"]
    elif cfg.preprocess.task == "csp":
        splits = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")
    
    # Process
    for split in splits[::-1]:
        converter.process(split=split)

if __name__ == "__main__":
    main()