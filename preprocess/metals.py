"""
Temporary python file for conformer matching
"""
import os
import copy
import time
import torch
import hydra
import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from utils.lmdb import read_lmdb, write_lmdb
from utils.environment import PROJECT_ROOT
from utils import molecule as mu


def process_one(value):
    """
    Extract metal building blocks from a single MOF. 
    """
    metal_mol_list = []
    feats = pickle.loads(value)
    
    for mol_str in feats['bb_mols']:
        bb_mol = Chem.MolFromMolBlock(mol_str, removeHs=False)

        # Check if metal atom is present
        if mu.is_metal_bb(bb_mol):
            metal_mol_list.append(bb_mol)
        
    return metal_mol_list

def align_mol_list(mol_list: list):
    """
    Args:
    - mol_list: list of RDKit molecule objects
    
    Returns:
    - aligned_mol_list: list of aligned RDKit molecule objects
    - rmsd (float): RMSD of the aligned molecules
    """
    ref_mol = mol_list[0]
    
    aligned_mol_list = []
    rmsd_list = []
    for mol in mol_list:
        rmsd, trans_mat, atom_map = AllChem.GetBestAlignmentTransform(prbMol=mol, refMol=ref_mol)
                
        new_mol = copy.deepcopy(mol)
    
        # Apply transformation
        coords = new_mol.GetConformer().GetPositions()
        coords = np.append(coords, np.ones((coords.shape[0], 1)), axis=1)
        new_coords = coords.dot(trans_mat.T)[:, :3]
        new_mol.GetConformer().SetPositions(new_coords)
        
        # Assign new atom orders
        atom_map = sorted(atom_map, key=lambda x: x[1])
        order = [prb_idx for prb_idx, ref_idx in atom_map]
        new_mol = Chem.RenumberAtoms(new_mol, order)
        
        # Append
        rmsd_list.append(rmsd)
        aligned_mol_list.append(new_mol)
    
    return aligned_mol_list, rmsd_list

def get_avg_mol(aligned_mol_list):
    """
    Args:
    - aligned_mol_list: list of aligned RDKit molecule objects
    
    Returns:
    - avg_mol: average RDKit molecule object
    """
    avg_mol = copy.deepcopy(aligned_mol_list[0])
    
    # Get average coordinates
    avg_coords = np.zeros((avg_mol.GetNumAtoms(), 3))
    for mol in aligned_mol_list:
        coords = mol.GetConformer().GetPositions()
        avg_coords += coords
    avg_coords /= len(aligned_mol_list)
    avg_mol.GetConformer().SetPositions(avg_coords)
    
    return avg_mol

def process_metal_type(smi, mol_list):
    """
    pass   
    """
    aligned_mol_list, rmsd_list = align_mol_list(mol_list)
    avg_mol = get_avg_mol(aligned_mol_list)
    return smi, avg_mol, rmsd_list


class CreateMetalLibrary:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Task
        self.task = process_cfg.task # 'gen' or 'csp'
        
        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir
        self.metal_dir = process_cfg.metal_dir
        if not os.path.exists(self.metal_dir):
            os.makedirs(self.metal_dir)

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus

    def process(self, split='train'):
        print(f"Processing {split} split for task {self.task}...")

        # Start timer
        start_time = time.time()
        
        # Read data
        data_dict = {}
        src_env = read_lmdb(f"{self.lmdb_dir}/{self.task}/MetalOxo_feats_{split}.lmdb")
        with src_env.begin() as src_txn:
            num_src_entries = src_env.stat()['entries']
            cursor = src_txn.cursor()
            for key_bytes, value in tqdm(cursor, desc="Reading data", total=num_src_entries):
                idx = int(key_bytes.decode('ascii'))
                data_dict[idx] = value
        src_env.close()
        
        # Extract metal building blocks
        metal_mol_list = Parallel(n_jobs=self.num_cpus)(delayed(process_one)(value) for value in tqdm(data_dict.values()))
        metal_mol_list = [mol for sublist in metal_mol_list for mol in sublist]
        
        # Group by building block type
        metal_mol_dict = defaultdict(list)
        for mol in metal_mol_list:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            metal_mol_dict[smiles].append(mol)
        print(f"INFO:: {len(metal_mol_dict)} metal building block types extracted: {list(metal_mol_dict.keys())}")
        
        # Create metallic building block library
        self.metal_bb_library = {}
        rmsd_dict = {}
        for smi, mol_list in tqdm(metal_mol_dict.items(), desc="Creating metal library"):
            aligned_mol_list, rmsd_list = align_mol_list(mol_list)
            avg_mol = get_avg_mol(aligned_mol_list)
            
            self.metal_bb_library[smi] = avg_mol
            rmsd_dict[smi] = rmsd_list
        
        # Collect results
        results = {
            'metal_mol_dict': metal_mol_dict,
            'metal_bb_library': self.metal_bb_library,
            'rmsd_dict': rmsd_dict
        }

        # Save processed data
        save_path = f"{self.metal_dir}/{self.task}/metal_lib_{split}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"Saving processed data to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        # End timer
        print(f"INFO:: Time taken: {time.time() - start_time:.4f} s")
    
    def filter_one(self, idx, value):
        """
        Return True only if all metal building blocks in the MOF exist in the library.
        """
        feats = pickle.loads(value)
        bb_mols = [Chem.MolFromMolBlock(mol_str, removeHs=False) for mol_str in feats['bb_mols']]

        for bb_mol in bb_mols:
            if mu.is_metal_bb(bb_mol):
                metal_smi = Chem.MolToSmiles(bb_mol, canonical=True)
                if metal_smi not in self.metal_bb_library:
                    return idx, None

        return idx, value

    def filter(self, split='val'):
        """
        Filter out MOFs that do not contain metal building blocks from the extracted library.
        """
        print(f"Processing {split} split for task {self.task}...")

        # Start timer
        start_time = time.time()

        # Paths
        src_path = f"{self.lmdb_dir}/{self.task}/MetalOxo_feats_{split}.lmdb"
        dest_path = f"{self.lmdb_dir}/{self.task}/MetalOxo_feats_{split}_tmp.lmdb"

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

        # Filter data
        filtered_data = Parallel(n_jobs=self.num_cpus)(
            delayed(self.filter_one)(idx, value) for idx, value in tqdm(data_dict.items())
        )

        # Write filtered data to LMDB
        dest_env = write_lmdb(dest_path)
        with dest_env.begin(write=True) as dest_txn:
            for idx, value in tqdm(filtered_data, desc="Writing data"):
                if value is not None:
                    key_bytes = f"{idx}".encode('ascii')
                    dest_txn.put(key_bytes, value)
        num_dest_entries = dest_env.stat()['entries']
        print(f"Remaining samples: {num_dest_entries}/{num_src_entries}")
        dest_env.close()

        # Replace original LMDB with filtered one
        os.remove(src_path)
        os.rename(dest_path, src_path)

        # End timer
        print(f"INFO:: Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):

    metal_lib = CreateMetalLibrary(cfg=cfg)

    # Set splits
    if cfg.preprocess.task == "gen":
        splits = ["val"]
    elif cfg.preprocess.task == "csp":
        splits = ["val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")

    # Extract from training data only
    metal_lib.process(split="train")

    # Filter out MOFs
    for split in splits:
        metal_lib.filter(split=split)

if __name__ == "__main__":
    main()
