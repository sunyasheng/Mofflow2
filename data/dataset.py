import os
import json
import copy
import pickle
import gzip
import logging
import numpy as np
import torch
from tqdm import tqdm
from rdkit import RDLogger, Chem
from rdkit.Chem import Descriptors
from torch.utils.data import Dataset
from torch.distributions import LogNormal, Uniform
from torch_geometric.utils import scatter
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from scipy.spatial.transform import Rotation
from torch.distributions import LogNormal, Uniform
from utils import data as du
from utils import so3 as su
from utils import torsion as tu
from utils import molecule as mu
from utils.pyg_data import MOFData
from utils.lmdb import read_lmdb, get_all_keys, get_data

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*') 

class MOFDataset(Dataset):
    def __init__(self, *, dataset_cfg, split='train'):

        self._log = logging.getLogger(__name__)
        self._dataset_cfg = dataset_cfg
        self._split = split

        # Load keys
        self._load_keys()

        # Number of units for dynamic batch sampler
        if dataset_cfg.loader.sampler == 'dynamic':
            self._load_num_units()

        # For corrupting data
        self._set_lattice_dist()
        self._rot_symmetry = self._dataset_cfg.rot_symmetry

    def _set_lattice_dist(self):
        lattice_cfg = self._dataset_cfg.lattice
        self._lognormal = LogNormal(
            loc=torch.Tensor(lattice_cfg.lognormal.loc),
            scale=torch.Tensor(lattice_cfg.lognormal.scale)
        )
        self._uniform = Uniform(
            low=lattice_cfg.uniform.low - lattice_cfg.uniform.eps, 
            high=lattice_cfg.uniform.high + lattice_cfg.uniform.eps
        )    
    
    def _load_keys(self):
        lmdb_dir = self._dataset_cfg.lmdb_dir
        dataset_prefix = self._dataset_cfg.dataset_prefix
        if self._split == 'train':
            sample_limit = self._dataset_cfg.train_sample_limit
        elif self._split == 'val':
            sample_limit = self._dataset_cfg.val_sample_limit
        elif self._split == 'test':
            sample_limit = self._dataset_cfg.test_sample_limit
        else:
            sample_limit = None

        env = read_lmdb(f"{lmdb_dir}/{dataset_prefix}_{self._split}.lmdb")
        with env.begin(buffers=True) as txn:
            self.keys = get_all_keys(txn)
        env.close()
        rank_zero_info(f"INFO:: {len(self.keys)} datapoints in {self._split} split.")

        # Limit sample size
        if sample_limit is not None:
            self.keys = self.keys[:sample_limit]
            rank_zero_info(f"INFO:: Limiting {self._split} split to {len(self.keys)} samples.")
        
    def _load_num_units(self):
        lmdb_dir = self._dataset_cfg.lmdb_dir
        dataset_prefix = self._dataset_cfg.dataset_prefix

        env = read_lmdb(f"{lmdb_dir}/{dataset_prefix}_{self._split}.lmdb")
        with env.begin(buffers=True) as txn:
            self.num_units = [get_data(txn, key)['gt_coords'].shape[0] for key in self.keys]
        env.close()
    
    def _open_lmdb(self, split):
        if hasattr(self, 'env') and hasattr(self, 'txn'):
            return # Already opened

        lmdb_dir = self._dataset_cfg.lmdb_dir
        dataset_prefix = self._dataset_cfg.dataset_prefix

        self.env = read_lmdb(f"{lmdb_dir}/{dataset_prefix}_{split}.lmdb")
        self.txn = self.env.begin(buffers=True, write=False)

    def _get_unique_rotmats(self, bb_rotate_coords_0: list, all_equiv_coords: list):
        """
        Get the unique rotation matrix considering all rotational symmetries.
        """
        all_rotmats_1to0 = []
        # Compute RMSD
        for bb_idx, bb_coords_0 in enumerate(bb_rotate_coords_0):
            bb_rotate_coords_1 = all_equiv_coords[bb_idx] # [num_equiv, num_atoms, 3]
            rmsds = du.pairwise_rmsd(bb_coords_0, bb_rotate_coords_1)
            bb_coords_1_min = bb_rotate_coords_1[rmsds.argmin()] # C_1*

            # Kabsch alignment
            rotmats_0to1_transpose, _, _ = du.kabsch_match(q_coord=bb_coords_1_min, p_coord=bb_coords_0)
            rotmats_1to0 = rotmats_0to1_transpose.unsqueeze(0)

            all_rotmats_1to0.append(rotmats_1to0)
                
        all_rotmats_1to0 = torch.cat(all_rotmats_1to0, dim=0) # [num_bbs, 3, 3]
        return all_rotmats_1to0
    
    @staticmethod
    def _sample_symmetric_torsion_angles(symmetry_scores: torch.Tensor) -> torch.Tensor:
        """
        Sample torsion angles based on bond symmetry.
        - Score 1: sample from [-π, π] (no symmetry)
        - Score 2: sample from [-π/2, π/2] (π symmetry)
        - Score 3: sample from [-π/3, π/3] (2π/3 symmetry)

        Args:
            symmetry_scores (torch.Tensor): [num_rotable_bonds], tensor of symmetry scores for each rotable bond
        """
        angle_limits = torch.pi / symmetry_scores
        torsion_angles = (2 * angle_limits) * torch.rand_like(symmetry_scores) - angle_limits
        return torsion_angles
        
    def _corrupt_trans(self, noisy_data, trans_0, t):
        # Sample trans_0
        if trans_0 is None:
            trans_0 = torch.randn(noisy_data.num_bbs, 3) # [num_bbs, 3]
        trans_0 = trans_0 - trans_0.mean(dim=0, keepdim=True)
        trans_0 = trans_0 * du.NM_TO_ANG_SCALE

        # Interpolate
        trans_t = (1 - t) * trans_0 + t * noisy_data.trans_1
        return trans_t

    def _corrupt_rotmats(self, noisy_data, rotmats_1to0, equiv_coords, t):
        # Sample rotmats_1to0
        if rotmats_1to0 is None:
            rotmats_1to0 = torch.tensor(
                Rotation.random(noisy_data.num_bbs).as_matrix()
            ).float() # [num_bbs, 3, 3]
        
        # Find unique rotmats_1to0
        if self._rot_symmetry:
            rotate_coords_0 = du.apply_rototranslation(
                X_atoms=noisy_data.matched_coords,
                rotmats=rotmats_1to0,
                trans=torch.zeros(noisy_data.num_bbs, 3),
                bb_num_vec=noisy_data.bb_num_vec
            )
            bb_rotate_coords_0 = torch.split(rotate_coords_0, noisy_data.bb_num_vec.tolist())
            rotmats_1to0 = self._get_unique_rotmats(bb_rotate_coords_0, equiv_coords)

        # Interpolate
        rotmats_1tot = su.rotvec_to_rotmat((1-t) * su.rotmat_to_rotvec(rotmats_1to0))
        return rotmats_1tot

    def _corrupt_torsions(self, noisy_data, torsion_0, t):
        # Sample torsions_1to0
        num_rotable_bonds = noisy_data.rotable_bond_mask.sum().item()
        if torsion_0 is None:
            torsion_0 = torch.empty(num_rotable_bonds).uniform_(-np.pi, np.pi) # [num_rotable_bonds]
        
        # Interpolate
        torsion_t = tu.get_xt(torsion_0, noisy_data.torsion_1, t) # [num_rotable_bonds]

        return torsion_t
    
    def _corrupt_lattice(self, noisy_data, lattice_0, t):
        # Sample lattice_0
        if lattice_0 is None:
            lengths_0 = self._lognormal.sample((1,)) # [1, 3]
            angles_0 = self._uniform.sample((1, 3)) # [1, 3]
            lattice_0 = torch.cat([lengths_0, angles_0], dim=-1) # [1, 6]

        # Interpolate
        lattice_t = (1 - t) * lattice_0 + t * noisy_data.lattice_1
        return lattice_t
    
    def corrupt_data(
            self, 
            data, 
            equiv_coords=None, 
            t=None, 
            rotmats_1to0=None, 
            torsion_0=None,
            trans_0=None,
            lattice_0=None
        ):
        noisy_data = copy.deepcopy(data)
        
        # Sample t
        if t is None:
            if self._dataset_cfg.t_sample_dist == 'uniform':
                t_min = self._dataset_cfg.t_uniform.t_min
                t_max = self._dataset_cfg.t_uniform.t_max
                t = torch.empty(1).uniform_(t_min, t_max).item() 
            elif self._dataset_cfg.t_sample_dist == 'logit_normal':
                mean = self._dataset_cfg.t_logit_normal.mean
                std = self._dataset_cfg.t_logit_normal.std
                normal_sample = torch.normal(mean=mean, std=std, size=(1,))
                t = torch.sigmoid(normal_sample).item()
            else:
                raise ValueError(f"Unknown t_sample_dist: {self._dataset_cfg.t_sample_dist}")
        t_bbs = t * torch.ones(noisy_data.num_bbs, 1) # [num_bbs, 1]
        t_atoms = t * torch.ones(noisy_data.num_atoms, 1) # [num_atoms, 1]
        
        # Sample rotmats_1tot
        if self._dataset_cfg.corrupt_rots:
            rotmats_1tot = self._corrupt_rotmats(noisy_data, rotmats_1to0, equiv_coords, t)
        else:
            rotmats_1tot = torch.eye(3).expand(noisy_data.num_bbs, 3, 3)
        
        # Sample trans_t
        bb_vec = du.repeat_interleave(noisy_data.bb_num_vec)
        noisy_data.trans_1 = scatter(noisy_data.matched_coords, bb_vec, dim=0, reduce='mean') # [num_bbs, 3]
        if self._dataset_cfg.corrupt_trans:
            trans_t = self._corrupt_trans(noisy_data, trans_0, t)
        else:
            trans_t = noisy_data.trans_1

        # Sample torsion_t
        noisy_data.torsion_1 = tu.get_dihedrals(noisy_data.matched_coords, noisy_data.canonical_torsion_tuples)
        if self._dataset_cfg.corrupt_torsions:
            torsion_t = self._corrupt_torsions(noisy_data, torsion_0, t)
        else:
            torsion_t = noisy_data.torsion_1
        
        # Create noisy coordinates
        coords_t = du.apply_rototranslation(
            X_atoms=noisy_data.matched_coords,
            rotmats=rotmats_1tot,
            trans=trans_t,
            bb_num_vec=noisy_data.bb_num_vec
        )
        coords_t = tu.apply_absolute_torsion(
            pos=coords_t,
            rotable_atom_mask=noisy_data.rotable_atom_mask,
            torsion_targets=torsion_t,
            torsion_tuples=noisy_data.canonical_torsion_tuples,
        )
        
        # Sample lattice_t
        if self._dataset_cfg.corrupt_lattice:
            lattice_t = self._corrupt_lattice(noisy_data, lattice_0, t)
        else:
            lattice_t = noisy_data.lattice_1
        
        # Add input
        noisy_data.t = t
        noisy_data.t_bbs = t_bbs
        noisy_data.t_atoms = t_atoms
        noisy_data.coords_t = coords_t
        noisy_data.lattice_t = lattice_t
        
        # Add ground-truth
        noisy_data.rotmats_tto1 = su.rot_transpose(rotmats_1tot)
        
        return noisy_data        

    @staticmethod
    def _assign_bb_idx(feats):
        """
        Assign positional index to each building block.
        Rules:
        - Metal building blocks have lower index than organic building blocks.
        - Building blocks of different types are arranged in increasing order of MolWt
        - Building blocks of the same type break ties with centroid coordinates.

        Returns:
            bb_pos_idx (torch.Tensor): [num_atoms], positional index for each atom (same index for atoms in the same building block)
        """
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

        return feats

    @staticmethod
    def _recenter_coords(coords, bb_num_vec):
        """
        Recenter coordinates so that the average of the building block centroids is at the origin.

        Args:
            coords (torch.Tensor): [num_atoms, 3], atomic coordinates.
            bb_num_vec (torch.Tensor): [num_bbs], number of atoms in each building block.
        Returns:
            torch.Tensor: [num_atoms, 3], recentered coordinates.
        """
        # Compute centroids
        bb_vec = du.repeat_interleave(bb_num_vec)
        bb_centroids = scatter(coords, bb_vec, dim=0, reduce='mean')
        centroid = bb_centroids.mean(dim=0, keepdim=True) # [1, 3]

        # Recenter coordinates
        recentered_coords = coords - centroid # [num_atoms, 3]
        return recentered_coords

    @staticmethod    
    def _get_data_from_feats(feats):
        best_matched_coords = feats['matched_coords'][-1] # [num_atoms, 3]

        data = MOFData(
            num_nodes=len(feats['bb_num_vec']),                             # M (necessary for computing batch.batch)
            num_atoms=best_matched_coords.shape[0],                         # N
            num_bbs=len(feats['bb_num_vec']),                               # M
            num_rotable_bonds=feats['rotable_bond_mask'].sum().item(),      # num_rotable_bonds
            gt_coords=feats['gt_coords'],                                   # [N, 3]
            matched_coords=best_matched_coords,                             # [N, 3]
            bb_num_vec=feats['bb_num_vec'],                                 # [M,]
            bb_pos_idx=feats['bb_pos_idx'],                                 # [N,]
            atom_types=feats['atom_types'],                                 # [N,]
            lattice_1=feats['lattice_1'].unsqueeze(0),                      # [1, 6]
            cell_1=feats['cell_1'].unsqueeze(0),                            # [1, 3, 3]
            atom_feats=feats['atom_feats'],                                 # [N, num_atom_features]
            bond_index=feats['bond_index'],                                 # [2, num_bonds]
            bond_feats=feats['bond_feats'],                                 # [num_bonds, num_bond_features]
            rotable_bond_mask=feats['rotable_bond_mask'],                   # [num_bonds]
            rotable_atom_mask=feats['rotable_atom_mask'],                   # [num_rotable_bonds, N]
            canonical_torsion_tuples=feats['canonical_torsion_tuples'],     # [num_rotable_bonds, 4]
        )
        return data

    def close_lmdb(self):
        if hasattr(self, 'txn'):
            self.txn.abort()
            del self.txn
        if hasattr(self, 'env'):
            self.env.close()
            del self.env

    def __del__(self):
        self.close_lmdb()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        if not hasattr(self, 'txn'):
            self._open_lmdb(split=self._split)
        feats = get_data(self.txn, self.keys[idx])
        feats = MOFDataset._assign_bb_idx(feats)
        data = MOFDataset._get_data_from_feats(feats)
        noisy_data = self.corrupt_data(data=data, equiv_coords=feats['equiv_coords'])
        noisy_data.data_idx = idx # Save the index for ordering cif files (during test time)
        return noisy_data        