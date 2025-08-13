from collections import defaultdict
import roma
import torch
import numpy as np
from utils import so3 as su
from utils import data as du
from utils import torsion as tu
from scipy.spatial.transform import Rotation
from torch_geometric.utils import scatter
from torch.distributions import LogNormal, Uniform
import copy


class Sampler:

    def __init__(self, cfg):
        self._model_cfg = cfg.model
        self._dataset_cfg = cfg.data
        self._sampler_cfg = cfg.inference.sampler

        self.use_svd = cfg.model.output_model_cfg.rotation.use_svd
        self.normalize_torsion = cfg.model.output_model_cfg.torsion.normalize

        self._set_lattice_dist()

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

    def _sample_t(self, num_batch):
        if self._dataset_cfg.t_sample_dist == 'uniform':
            t_min = self._dataset_cfg.t_uniform.t_min
            t_max = self._dataset_cfg.t_uniform.t_max
            t = torch.empty(num_batch, device=self.device).uniform_(t_min, t_max)
        elif self._dataset_cfg.t_sample_dist == 'logit_normal':
            mean = self._dataset_cfg.t_logit_normal.mean
            std = self._dataset_cfg.t_logit_normal.std
            normal_sample = torch.normal(mean=mean, std=std, size=(num_batch,), device=self.device)
            t = torch.sigmoid(normal_sample)
        else:
            raise ValueError(f"Unknown t_sample_dist: {self._dataset_cfg.t_sample_dist}")
        return t[:, None] # [num_batch, 1]
    
    def _sample_trans_0(self, noisy_batch):
        # Sample trans_0
        num_bbs = noisy_batch.num_bbs.sum().item()
        trans_0 = torch.randn(num_bbs, 3, device=self.device) # [num_bbs, 3]

        # Remove mean
        centroids = scatter(trans_0, noisy_batch.batch, dim=0, reduce='mean') # [num_batch, 3]
        centroids = centroids.repeat_interleave(noisy_batch.num_bbs, dim=0) # [num_bbs, 3]
        trans_0 = trans_0 - centroids
        trans_0 = trans_0 * du.NM_TO_ANG_SCALE

        return trans_0

    def _sample_rotmats_1to0(self, noisy_batch):
        num_bbs = noisy_batch.num_bbs.sum().item()
        # Sample rotmats_1to0
        rotmats_1to0 = torch.tensor(
            Rotation.random(num_bbs).as_matrix(), device=self.device
        ).float() # [num_bbs, 3, 3]
        
        return rotmats_1to0

    def _sample_torsion_0(self, noisy_batch):
        # Sample torsions_0
        num_rotable_bonds = noisy_batch.num_rotable_bonds.sum().item()
        torsion_0 = torch.empty(num_rotable_bonds, device=self.device).uniform_(-np.pi, np.pi)  # [num_rotable_bonds]

        return torsion_0
    
    def _sample_lattice_0(self, noisy_batch):
        lengths_0 = self._lognormal.sample((noisy_batch.num_graphs,))           # [num_batch, 3]
        angles_0 = self._uniform.sample((noisy_batch.num_graphs, 3))            # [num_batch, 3]
        lattice_0 = torch.cat([lengths_0, angles_0], dim=-1).to(self.device)    # [num_batch, 6]

        return lattice_0
    
    def _sample_prior(self, batch, init_coords, min_t):
        noisy_batch = copy.deepcopy(batch)
        num_batch = noisy_batch.num_graphs

        # Set t
        t = torch.tensor([min_t], device=self.device, dtype=torch.float32).expand(num_batch)[:, None] # [num_batch, 1]
        t_atoms = t.repeat_interleave(noisy_batch.num_atoms)[:, None] # [num_atoms, 1]

        # Sample trans_0
        trans_0 = self._sample_trans_0(noisy_batch)

        # Sample rotmats_1to0
        rotmats_1to0 = self._sample_rotmats_1to0(noisy_batch)

        # Sample torsion_0
        torsion_0 = self._sample_torsion_0(noisy_batch)
        
        # Sample lattice_t
        lattice_0 = self._sample_lattice_0(noisy_batch)
        
        # Create X_0
        init_coords = init_coords if init_coords is not None else noisy_batch.matched_coords
        coords_0 = du.apply_rototranslation(
            X_atoms=init_coords,
            rotmats=rotmats_1to0,
            trans=trans_0,
            bb_num_vec=noisy_batch.bb_num_vec
        )
        coords_0 = tu.apply_absolute_torsion(
            pos=coords_0,
            rotable_atom_mask=noisy_batch.rotable_atom_mask,
            torsion_targets=torsion_0,
            torsion_tuples=noisy_batch.canonical_torsion_tuples,
        )
        
        # Add input
        noisy_batch.t_atoms = t_atoms
        noisy_batch.coords_t = coords_0
        noisy_batch.lattice_t = lattice_0
        
        return noisy_batch        

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t > 0
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_tto1):
        assert d_t > 0
        rots_vf = su.rotmat_to_rotvec(rotmats_tto1) / (1 - t)
        return su.rotvec_to_rotmat(rots_vf * d_t)

    def _torsion_euler_step(self, d_t, t, torsion_1, torsion_t):
        assert d_t > 0
        torsion_vf = tu.get_ut(torsion_t, torsion_1, t)
        return tu.sym_mod(torsion_t + torsion_vf * d_t)
    
    def sample(
            self, 
            batch,
            init_coords=None,
            model=None,
            num_timesteps=None
        ):
        self.device = batch.atom_types.device

        # Sample prior
        batch = self._sample_prior(batch, init_coords, min_t=self._sampler_cfg.min_t)

        # Set timesteps
        if num_timesteps is None:
            num_timesteps = self._sampler_cfg.num_timesteps
        ts = torch.linspace(self._sampler_cfg.min_t, 1.0, num_timesteps).to(self.device)
        t_1 = ts[0]

        # Initialize trajectories
        coord_traj, lattice_traj = [], []

        coord_traj.append(batch.coords_t)
        lattice_traj.append(batch.lattice_t)

        with torch.no_grad():
            for t_2 in ts[1:]:
                dt = t_2 - t_1

                # Forward pass
                outputs = model(batch)

                # Compute translation update
                if 'pred_trans' in outputs:
                    pred_trans_1 = outputs['pred_trans'] * du.NM_TO_ANG_SCALE

                    # Compute trans_t
                    bb_vec = du.repeat_interleave(batch.bb_num_vec)
                    trans_t_1 = scatter(batch.coords_t, bb_vec, dim=0, reduce='mean')

                    # Take a step
                    trans_t_2 = self._trans_euler_step(
                        d_t=dt, t=t_1, trans_1=pred_trans_1, trans_t=trans_t_1,
                    )
                else:
                    trans_t_2 = batch.trans_1
                
                # Compute rotation update
                if 'pred_rotmats' in outputs:
                    pred_rotmats_t1_to_1 = outputs['pred_rotmats']
                    if not self.use_svd:
                        pred_rotmats_t1_to_1 = roma.special_procrustes(pred_rotmats_t1_to_1)

                    # Take a step
                    rotmats_t1_to_t2 = self._rots_euler_step(
                        d_t=dt, t=t_1, rotmats_tto1=pred_rotmats_t1_to_1
                    )
                else:
                    rotmats_t1_to_t2 = torch.eye(3).expand(batch.num_bbs, 3, 3).to(self.device)
                
                # Compute torsion update
                if 'pred_torsion' in outputs:
                    pred_torsion_1 = outputs['pred_torsion'] # [num_rotable_bonds, 2]
                    if not self.normalize_torsion:
                        pred_torsion_1 = pred_torsion_1 / (pred_torsion_1.norm(dim=-1, keepdim=True) + 1e-8)
                    pred_torsion_1 = tu.cos_sin_to_angle(pred_torsion_1) # [num_rotable_bonds]

                    # Compute torsion_t
                    torsion_t = tu.get_dihedrals(batch.coords_t, batch.canonical_torsion_tuples)

                    # Take a step
                    torsion_t_2 = self._torsion_euler_step(
                        d_t=dt, t=t_1, torsion_1=pred_torsion_1, torsion_t=torsion_t
                    )
                else:
                    torsion_t_2 = batch.torsion_1
                
                # Compute lattice update
                if 'pred_lattice' in outputs:
                    pred_lattice_1 = du.lattice_to_ang_degrees(outputs['pred_lattice'])

                    # Compute lattice_t
                    lattice_t_1 = batch.lattice_t

                    # Take a step
                    lattice_t_2 = self._trans_euler_step(
                        d_t=dt, t=t_1, trans_1=pred_lattice_1, trans_t=lattice_t_1
                    )
                else:
                    lattice_t_2 = batch.lattice_1
                
                # Update coordinates
                coords_t_2 = tu.apply_absolute_torsion(
                    pos=batch.coords_t,
                    rotable_atom_mask=batch.rotable_atom_mask,
                    torsion_targets=torsion_t_2,
                    torsion_tuples=batch.canonical_torsion_tuples,
                )
                coords_t_2 = du.apply_rototranslation(
                    X_atoms=coords_t_2,
                    rotmats=rotmats_t1_to_t2,
                    trans=trans_t_2,
                    bb_num_vec=batch.bb_num_vec,
                )

                # Update data
                batch.coords_t = coords_t_2
                batch.lattice_t = lattice_t_2
                batch.t_atoms = t_2 * torch.ones(batch.coords_t.shape[0], 1, device=self.device)

                # Append to trajectory
                coord_traj.append(coords_t_2.detach().cpu())
                lattice_traj.append(lattice_t_2.detach().cpu())

                # Update time
                t_1 = t_2

        return coord_traj, lattice_traj