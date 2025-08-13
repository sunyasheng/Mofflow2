"""
Code for creating _processed.pkl.gz files from processed LMDB files (for training MOFFlow-1).
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


class FileConverter:
    def __init__(self, cfg: DictConfig):
        process_cfg = cfg.preprocess

        # Task
        self.task = process_cfg.task  # 'gen' or 'csp'

        # Directories
        self.lmdb_dir = process_cfg.lmdb_dir
        self.pkl_dir = f'{cfg.paths.data_dir}/pkl/{self.task}'
        os.makedirs(self.pkl_dir, exist_ok=True)

        # Number of CPUs
        self.num_cpus = process_cfg.num_cpus  

    @staticmethod
    def _get_equiv_vec(cart_coords, atom_types):

        centroid = np.mean(cart_coords, axis=0)

        # Center of mass weighted by atomic number
        weight = atom_types / atom_types.sum()
        weighted_centroid = np.sum(cart_coords * weight[:, None], axis=0)

        # Equivariant vector
        equiv_vec = weighted_centroid - centroid

        # If v = 0 (symmetric), take the closest non-zero atom
        if np.allclose(equiv_vec, 0):
            dist = np.linalg.norm(cart_coords, axis=1)
            sorted_indices = np.argsort(dist)

            i = 0
            while i < len(sorted_indices) and np.allclose(equiv_vec, 0):
                equiv_vec = cart_coords[sorted_indices[i]]
                i += 1
        
        assert not np.allclose(equiv_vec, 0), "Equivariant vector is zero"
        return equiv_vec
    
    @staticmethod
    def _get_pca_axes(data):
        # Center the data
        data_mean = np.mean(data, axis=0)
        centered_data = data - data_mean

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)
        if covariance_matrix.ndim == 0:
            return np.zeros(3), np.eye(3)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues, eigenvectors  
    
    def _get_equivariant_axes(self, cart_coords, atom_types):
        """
        Return:
            R: equivariant rotation matrix
        """

        if cart_coords.shape[0] == 1:
            return np.eye(3)

        equiv_vec = self._get_equiv_vec(cart_coords, atom_types)    # v(X)

        _, axes = self._get_pca_axes(cart_coords)                   # PCA(X)
        ve = equiv_vec @ axes
        flips = ve < 0 
        axes = np.where(flips[None], -axes, axes)

        right_hand = np.stack(
            [axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])], axis=1
        )
        
        return right_hand
    
    def _rotate_bb(self, bb_coord, bb_atom_type):
        """
        Returns:
            rotmats: numpy array of shape (3, 3)
            local_coord: numpy array of shape (n_bb_atoms, 3) 
        """
        rotmats = self._get_equivariant_axes(bb_coord, bb_atom_type) # f(X)
        local_coord = bb_coord @ rotmats                             # g(X) = X f(X)

        return rotmats, local_coord 
    
    def process_one(self, idx, value):
        try:
            feats = pickle.loads(value)

            bb_num_vec = feats['bb_num_vec']
            real_gt_coords = feats['gt_coords'] # Use gt_coords
            gt_coords = feats['matched_coords'][-1] # Use matched_coords
            atom_types = feats['atom_types']

            # Create initial features
            new_feats = {
                'bb_mols': feats['bb_mols'], # For inference
                'res_mask': torch.ones_like(bb_num_vec, dtype=torch.int),
                'diffuse_mask': torch.ones_like(bb_num_vec, dtype=torch.int),
                'gt_coords': gt_coords,
                'real_gt_coords': real_gt_coords,
                'bb_num_vec': bb_num_vec,
                'atom_types': atom_types,
                'lattice_1': feats['lattice_1'],
                'cell_1': feats['cell_1'],
            }

            # Compute trans_1
            bb_vec = du.repeat_interleave(bb_num_vec)
            trans_1 = scatter(gt_coords, bb_vec, dim=0, reduce='mean') # [num_bbs, 3]

            # Center the coordinates
            bb_coords = torch.split(gt_coords, bb_num_vec.tolist(), dim=0)
            centered_bb_coords = [coords - centroid for coords, centroid in zip(bb_coords, trans_1)]
                
            # Compute rotmats_1, local_coords
            rotmats_1 = []
            local_coords = []
            bb_atom_types = torch.split(atom_types, bb_num_vec.tolist(), dim=0)

            for bb_coords, bb_types in zip(centered_bb_coords, bb_atom_types):
                # Convert to numpy
                coords_np = bb_coords.detach().cpu().numpy()
                types_np = bb_types.detach().cpu().numpy()

                # Compute rotation and local coordinates
                rotmat_np, local_coord_np = self._rotate_bb(coords_np, types_np)

                # Convert back to torch and append
                rotmats_1.append(torch.from_numpy(rotmat_np).float())
                local_coords.append(torch.from_numpy(local_coord_np).float())

            rotmats_1 = torch.stack(rotmats_1, dim=0) # [num_bbs, 3, 3]
            local_coords = torch.cat(local_coords, dim=0) # [num_atoms, 3]

            # Update new_feats
            new_feats['trans_1'] = trans_1
            new_feats['rotmats_1'] = rotmats_1
            new_feats['local_coords'] = local_coords

            return new_feats
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
        dest_path = f"{self.pkl_dir}/{split}_matched_processed.pkl.gz"

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
        feats_list = [feats for feats in feats_list if feats is not None]  # Filter out None values
        print(f"INFO:: Number of valid samples: {len(feats_list)}/{num_src_entries}")

        # Write as .pkl.gz
        with gzip.open(f"{dest_path}", 'wb') as f:
            pickle.dump(feats_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # End timer
        print(f"INFO:: Time taken: {time.time() - start_time:.4f} s")


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):
    converter = FileConverter(cfg=cfg)

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