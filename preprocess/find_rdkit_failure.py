"""
Find building blocks that fail to generate RDKit conformers.
"""
import os
import time
import hydra
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rdkit import Chem, RDLogger
from joblib import Parallel, delayed
from omegaconf import DictConfig
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from utils.environment import PROJECT_ROOT
from utils.lmdb import read_lmdb, write_lmdb
from utils import conformer_matching as cu
from utils import molecule as mu


class MOFMatcher:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Directories
        self.split_dir = process_cfg.split_dir
        self.lmdb_dir = process_cfg.lmdb_dir
        self.data_dir = cfg.paths.data_dir

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus

    def process_one(self, idx, value):
        # Disable warnings within each process
        RDLogger.DisableLog('rdApp.*')
        feats = pickle.loads(value)
        bb_mols = [Chem.MolFromMolBlock(mol_str, removeHs=False) for mol_str in feats['bb_mols']]
        for bb_idx, bb_mol in enumerate(bb_mols):
            if not mu.is_metal_bb(bb_mol):
                try:
                    cu.get_rd_conformer(bb_mol)
                    return None
                except Exception as e:
                    print(f"Failed to process {idx} building block {bb_idx}: {e}")
                    return bb_mol


    def process(self, split="train"):
        print(f"Matching {split} split...")

        # Start timer
        start_time = time.time()

        # Load split indices
        split_file = f"{self.split_dir}/{split}_split.txt"
        split_idx = np.loadtxt(split_file, dtype=int)

        # Read data
        data_dict = {}
        src_env = read_lmdb(f"{self.lmdb_dir}/MetalOxo_feats_{split}.lmdb")
        with src_env.begin() as src_txn:
            for idx in tqdm(split_idx, desc="Reading data"):
                key_bytes = f"{idx}".encode('ascii')
                value = src_txn.get(key_bytes)
                if value is None:
                    continue
                data_dict[idx] = value
        src_env.close()

        # Process data
        bb_mol_list = Parallel(n_jobs=self.num_cpus)(delayed(self.process_one)(idx, value) for idx, value in tqdm(data_dict.items()))
        bb_mol_list = [bb_mol for bb_mol in bb_mol_list if bb_mol is not None]
        
        # Write extracted features with pickle
        with open(f"{self.data_dir}/failed_bb_mols_{split}.pkl", "wb") as f:
            pickle.dump(bb_mol_list, f)
        
        # End timer
        print(f"Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):

    mofmatcher = MOFMatcher(cfg=cfg)
    # mofmatcher.process(split='train')
    mofmatcher.process(split='val')

if __name__ == "__main__":
    main()
