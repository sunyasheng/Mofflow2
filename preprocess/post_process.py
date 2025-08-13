"""
Tentative codes for post-processing the data. Currently includes:
- Assigning positional index for building blocks 
- Filtering building blocks != 4
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
from rdkit.Chem import Descriptors
from openbabel import pybel
from pymatgen.core import Lattice
from torch_geometric.utils import scatter
from utils.lmdb import read_lmdb, write_lmdb
from utils import data as du
from utils import molecule as mu
from utils.environment import PROJECT_ROOT


def process_one(idx, value):
    # Disable warnings within each process
    RDLogger.DisableLog('rdApp.*') 
    pybel.ob.obErrorLog.SetOutputLevel(0)
    warnings.filterwarnings("ignore")

    try:
        # Load data
        feats = pickle.loads(value)

        # Check if the building block is 4
        num_bbs = len(feats['bb_num_vec'])
        if num_bbs != 4:
            return idx, None

        # Assign building block positional index
        bb_mols = [Chem.MolFromMolBlock(mol_str) for mol_str in feats['bb_mols']]
        mol_wt = torch.tensor([Descriptors.ExactMolWt(mol) for mol in bb_mols])
        is_metal = torch.tensor([mu.is_metal_bb(mol) for mol in bb_mols]).bool()

        # Compute centroids
        matched_coords = feats['matched_coords'][-1] # [num_atoms, 3]
        bb_vec = du.repeat_interleave(feats['bb_num_vec'])
        bb_centroids = scatter(matched_coords, bb_vec, dim=0, reduce='mean') # [num_bbs, 3]

        # Stack for lexsort (sort by last key first)
        organic_flag = (~is_metal).long() # metals are 0, organics are 1 
        sort_keys = torch.stack([
            organic_flag,
            mol_wt,
            bb_centroids[:, 0], # x
            bb_centroids[:, 1], # y
            bb_centroids[:, 2], # z
        ], dim=0).numpy() # [num_sort_keys, num_bbs]

        # Lexicographic sort
        sort_keys = sort_keys[::-1] # Reverse order for lexsort
        sorted_indices = torch.from_numpy(np.lexsort(sort_keys)).long() # [num_bbs]

        # Map building block index to sorted index
        inverse_map = torch.empty_like(sorted_indices)
        inverse_map[sorted_indices] = torch.arange(len(sorted_indices))

        # Assign position to each atom
        bb_pos_idx = inverse_map[bb_vec] # [num_atoms]
        feats['bb_pos_idx'] = bb_pos_idx

        new_value = pickle.dumps(feats)
        return idx, new_value
    except Exception as e:
        print(f"Failed to process {idx}: {e}")
        return idx, None


class PostProcess:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus  

    def process(self, split="train"):
        print(f"Checking {split} split...")

        # Start timer
        start_time = time.time()

        # Read data
        data_dict = {}
        src_env = read_lmdb(f"{self.lmdb_dir}/MetalOxo_absolute_torsion_{split}.lmdb")
        with src_env.begin() as src_txn:
            num_src_entries = src_env.stat()['entries']
            cursor = src_txn.cursor()
            for key_bytes, value in tqdm(cursor, desc="Reading data", total=num_src_entries):
                idx = int(key_bytes.decode('ascii'))
                data_dict[idx] = value
        src_env.close()

        # Process data
        feats_list = Parallel(n_jobs=self.num_cpus)(delayed(process_one)(idx, value) for idx, value in tqdm(data_dict.items()))

        # Write extracted features to LMDB
        dest_env = write_lmdb(f"{self.lmdb_dir}/MetalOxo_bb-4_{split}.lmdb")
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
    extractor = PostProcess(cfg=cfg)
    # extractor.process(split="val")
    extractor.process(split="train")

if __name__ == "__main__":
    main()