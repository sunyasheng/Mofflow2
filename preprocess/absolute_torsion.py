"""
Add torsion symmetry information to data
"""
import time
import hydra
import pickle
import torch
import warnings
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from omegaconf import DictConfig
from rdkit import Chem, RDLogger
from openbabel import pybel
from pymatgen.core import Lattice
from utils.lmdb import read_lmdb, write_lmdb
from utils import torsion as tu
from utils import molecule as mu
from utils import data_features as fu
from utils.environment import PROJECT_ROOT


def process_one(idx, value):
    # Disable warnings within each process
    RDLogger.DisableLog('rdApp.*') 
    pybel.ob.obErrorLog.SetOutputLevel(0)
    warnings.filterwarnings("ignore")

    try:
        # Load data
        feats = pickle.loads(value)
        bb_mols = [Chem.MolFromMolBlock(mol_str, removeHs=False) for mol_str in feats['bb_mols']]
        
        # Extract canonical torsion tuples
        offset = 0
        total_torsion_tuples = []
        for i, bb_mol in enumerate(bb_mols):
            bb_feats = mu.featurize_mol(bb_mol)
            bond_mask, _ = mu.get_transformation_mask(bb_feats, bb_mol)

            canonical_tuples = tu.get_canonical_torsion_tuples_from_bonds(bb_mol, bb_feats.edge_index[:, bond_mask])
            canonical_tuples = canonical_tuples + offset

            total_torsion_tuples.append(canonical_tuples)

            bb_num_atoms = bb_mol.GetNumAtoms()
            offset += bb_num_atoms

        total_torsion_tuples = torch.cat(total_torsion_tuples, dim=0)  # [total_num_rotatable_bonds, 4]
        feats['canonical_torsion_tuples'] = total_torsion_tuples

        new_value = pickle.dumps(feats)
        return idx, new_value
    except Exception as e:
        print(f"Failed to process {idx}: {e}")
        return idx, None


class ExtractFeatures:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir
        self.split_dir = process_cfg.split_dir

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus  

    def process(self, split="train"):
        print(f"Checking {split} split...")

        # Start timer
        start_time = time.time()

        # Load split indices
        split_file = f"{self.split_dir}/{split}_split.txt"
        split_idx = np.loadtxt(split_file, dtype=int)

        # Read data
        data_dict = {}
        src_env = read_lmdb(f"{self.lmdb_dir}/MetalOxo_final_{split}.lmdb")
        with src_env.begin() as src_txn:
            for idx in tqdm(split_idx, desc="Reading data"):
                key_bytes = f"{idx}".encode('ascii')
                value = src_txn.get(key_bytes)
                if value is None:
                    continue
                data_dict[idx] = value
        num_src_entries = src_env.stat()['entries']
        src_env.close()

        # Process data
        feats_list = Parallel(n_jobs=self.num_cpus)(delayed(process_one)(idx, value) for idx, value in tqdm(data_dict.items()))

        # Write extracted features to LMDB
        dest_env = write_lmdb(f"{self.lmdb_dir}/MetalOxo_absolute_torsion_{split}.lmdb")
        with dest_env.begin(write=True) as dest_txn:
            for idx, value in tqdm(feats_list):
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
    extractor = ExtractFeatures(cfg=cfg)
    extractor.process(split="val")
    extractor.process(split="train")

if __name__ == "__main__":
    main()