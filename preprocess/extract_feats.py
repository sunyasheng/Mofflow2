"""
Extract key features from the raw data.
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
from utils import data_features as fu
from utils.environment import PROJECT_ROOT


def process_one(idx, value):
    # Disable warnings within each process
    RDLogger.DisableLog('rdApp.*') 
    pybel.ob.obErrorLog.SetOutputLevel(0)
    warnings.filterwarnings("ignore")

    try:
        # Load data
        data = pickle.loads(value)

        # Get Niggli cell
        lattice = Lattice(data.cell)
        lattice = lattice.get_niggli_reduced_lattice()

        # Get canonical rotation matrix
        c_rotmat, lattice = fu.get_canonical_rotmat(lattice)

        # Get Cartesian coordinates (recentered)
        gt_coords = fu.get_gt_coords(data, c_rotmat)
        bb_num_vec = torch.tensor([bb.num_atoms for bb in data.pyg_mols])

        # Convert numpy arrays to torch tensors
        feats = {
            'gt_coords': gt_coords.float(),
            'bb_num_vec': bb_num_vec.int(),
            'atom_types': torch.cat([bb.atom_types for bb in data.pyg_mols], dim=0).int(),
            'lattice_1': torch.tensor(lattice.parameters).float(),
            'cell_1': torch.tensor(lattice.matrix).float(),
            'props': data.y # [1, prop_dim]
        }
        
        # Analyze point group symmetry
        pg_types, symops, bb_centered_coords = fu.get_pga_info(
            gt_coords=feats['gt_coords'], 
            atom_types=feats['atom_types'], 
            bb_num_vec=feats['bb_num_vec']
        )
        # Get symmetrically (rotational) equivalent coordinates
        equiv_coords = fu.get_equiv_coords(pg_types, symops, bb_centered_coords)
        feats['equiv_coords'] = equiv_coords

        # Chemical validity check (RDKit)
        bb_mols = fu.get_bb_mols_from_feats(feats)
        import pdb; pdb.set_trace()

        # Extract rotatable bonds
        rotable_bond_data = tu.get_rotable_bond_data(bb_mols, feats['bb_num_vec'])
        feats['bb_mols'] = [Chem.MolToMolBlock(bb_mol) for bb_mol in bb_mols]
        for key, value in rotable_bond_data.items():
            feats[key] = value
        return idx, pickle.dumps(feats)
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        return idx, None


class ExtractFeatures:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Task
        self.task = process_cfg.task # 'gen' or 'csp'

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus

    def process(self, split="train"):
        print(f"Processing {split} split for task {self.task}...")

        # Start timer
        start_time = time.time()

        # Read data
        data_dict = {}
        src_env = read_lmdb(f"{self.lmdb_dir}/{self.task}/MetalOxo_filtered_{split}.lmdb")
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
        dest_env = write_lmdb(f"{self.lmdb_dir}/{self.task}/MetalOxo_feats_{split}.lmdb")
        with dest_env.begin(write=True) as dest_txn:
            for idx, feats in feats_list:
                if feats is not None:
                    key_bytes = f"{idx}".encode('ascii')
                    dest_txn.put(key_bytes, feats)

        num_dest_entries = dest_env.stat()['entries']
        print(f"INFO:: Remaining samples: {num_dest_entries}/{num_src_entries}")        
        dest_env.close()
        
        # End timer
        print(f"INFO:: Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):
    extractor = ExtractFeatures(cfg=cfg)

    # Set splits
    if cfg.preprocess.task == "gen":
        splits = ["train", "val"]
    elif cfg.preprocess.task == "csp":
        splits = ["train", "val", "test"]
    else:
        raise ValueError(f"Unknown task: {cfg.preprocess.task}")
    
    # Process
    for split in splits[::-1]:
        extractor.process(split=split)

if __name__ == "__main__":
    main()