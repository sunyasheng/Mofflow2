import numpy as np
import roma
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from models.flow_model import FlowModel
from data.sampler import Sampler
from pytorch_lightning import LightningModule
from utils import data as du
from utils import torsion as tu
from utils import model as mu
from utils import so3 as su
from utils.visualize import visualize_torsion, visualize_rotations


class FlowModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # Configs
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        
        # Model
        self.model = FlowModel(cfg.model)

    def compute_loss(self, pred_dict, noisy_batch, mode=None):
        training_cfg = self._exp_cfg.training
        visualize_freq = self._exp_cfg.visualize_freq
        losses = {}
        
        # Translation loss
        if 'pred_trans' in pred_dict:
            gt_trans = noisy_batch.trans_1 * du.ANG_TO_NM_SCALE
            pred_trans = pred_dict['pred_trans']

            trans_error = (gt_trans - pred_trans) # [num_bbs, 3]
            trans_loss = torch.sum(trans_error ** 2, dim=-1) # [num_bbs]
            trans_loss = scatter(trans_loss, noisy_batch.batch, dim=0, reduce='mean') # [batch_size]

            losses['trans_loss'] = trans_loss * training_cfg.translation_loss_weight

        # Rotation loss
        if 'pred_rotmats' in pred_dict:
            gt_rotmats = noisy_batch.rotmats_tto1
            pred_rotmats = pred_dict['pred_rotmats']

            if (
                mode == 'train' 
                and self.logger is not None 
                and visualize_freq 
                and self.global_step % visualize_freq == 0
            ):
                # Apply SVD to raw rotation matrices
                pred_rotmats_vis = (
                    roma.special_procrustes(pred_rotmats)
                    if not self._model_cfg.output_model_cfg.rotation.use_svd
                    else pred_rotmats
                )

                visualize_rotations(
                    gt_rotmats=gt_rotmats,
                    pred_rotmats=pred_rotmats_vis,
                    global_step=self.global_step,
                    logger=self.logger,
                    tag = 'img/rotation',
                    is_global_zero=self.trainer.is_global_zero
                )

            rotmats_error = (gt_rotmats - pred_rotmats) # [num_bbs, 3, 3]
            rotmats_loss = torch.sum(rotmats_error ** 2, dim=[1, 2]) # [num_bbs]
            rotmats_loss = scatter(rotmats_loss, noisy_batch.batch, dim=0, reduce='mean') # [batch_size]

            losses['rotmats_loss'] = rotmats_loss * training_cfg.rotation_loss_weight

        # Lattice loss
        if 'pred_lattice' in pred_dict:
            gt_lattice = du.lattice_to_nm_radians(noisy_batch.lattice_1)
            pred_lattice = pred_dict['pred_lattice']
            lattice_error = (gt_lattice - pred_lattice) # [batch_size, 6]
            lattice_loss = torch.sum(lattice_error ** 2, dim=-1) # [batch_size]

            losses['lattice_loss'] = lattice_loss * training_cfg.lattice_loss_weight
        
        # Torsion loss
        if 'pred_torsion' in pred_dict:
            gt_torsion = tu.angle_to_cos_sin(noisy_batch.torsion_1) # [num_rotable_bonds, 2]
            raw_torsion = pred_dict['pred_torsion']
            pred_torsion_norm = torch.norm(raw_torsion, dim=-1, keepdim=True) # [num_rotable_bonds, 1]
            pred_torsion = raw_torsion / (pred_torsion_norm + 1e-8) # [num_rotable_bonds, 2]

            if (
                mode == 'train' 
                and self.logger is not None 
                and visualize_freq 
                and self.global_step % visualize_freq == 0
            ):                
                # Visualize raw torsion
                visualize_torsion(
                    gt_torsion=gt_torsion,
                    pred_torsion=raw_torsion,
                    global_step=self.global_step,
                    tag = 'img/raw_torsion',
                    logger=self.logger,
                    is_global_zero=self.trainer.is_global_zero
                )
                # Visualize normalized torsion
                visualize_torsion(
                    gt_torsion=gt_torsion,
                    pred_torsion=pred_torsion,
                    global_step=self.global_step,
                    tag = 'img/normalized_torsion',
                    logger=self.logger,
                    is_global_zero=self.trainer.is_global_zero
                )

            # Angle loss
            if self._model_cfg.output_model_cfg.torsion.normalize:
                torsion_error = (gt_torsion - pred_torsion) # [num_rotable_bonds, 2]
            else:
                torsion_error = (gt_torsion - raw_torsion) # [num_rotable_bonds, 2]
            torsion_angle_loss = torch.sum(torsion_error ** 2, dim=-1) # [num_rotable_bonds]

            # Average within each datapoint
            rotable_bond_index = noisy_batch.bond_index[:, noisy_batch.rotable_bond_mask] # [2, num_rotable_bonds]
            atoms_vec = du.repeat_interleave(noisy_batch.num_atoms) # [num_atoms]
            rotable_bond_vec = atoms_vec[rotable_bond_index[0]] # [num_rotable_bonds]
            torsion_angle_loss = scatter(torsion_angle_loss, rotable_bond_vec, dim=0, reduce='mean')
            losses['torsion_angle_loss'] = torsion_angle_loss * training_cfg.torsion_angle_loss_weight

            # Torsion norm loss
            if training_cfg.torsion_norm_loss_weight > 0 and self._model_cfg.output_model_cfg.torsion.normalize:
                torsion_norm_loss = torch.abs(pred_torsion_norm - 1.0) # [num_rotable_bonds]
                torsion_norm_loss = scatter(torsion_norm_loss, rotable_bond_vec, dim=0, reduce='mean')
                losses['torsion_norm_loss'] = torsion_norm_loss * training_cfg.torsion_norm_loss_weight

        return losses

    def forward(self, batch):
        return self.model(batch)
    
    def model_step(self, batch, mode=None):
        pred_dict = self(batch)

        # Compute rotation loss only
        batch_loss_dict = self.compute_loss(pred_dict, batch, mode=mode)
        
        return batch_loss_dict
    
    def on_train_start(self):
        self._epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self._epoch_start_time = time.time()
    
    def training_step(self, batch, batch_idx):
        step_start_time = time.time()
        
        # Compute loss
        batch_loss_dict = self.model_step(batch, mode='train')
        total_losses = {
            k: torch.mean(v) for k,v in batch_loss_dict.items()
        }

        ####### Loggings #######
        num_batch = batch.num_graphs

        # Batch size
        self.log('train/batch_size', float(num_batch))

        # Average time
        self.log(f"train/t", np.mean(du.to_numpy(batch.t)), batch_size=num_batch)

        # Loss
        for loss_name, batch_loss in batch_loss_dict.items():
            # Total loss
            self.log(f"train/{loss_name}", total_losses[loss_name], batch_size=num_batch)
        
            # Stratified loss
            stratified_losses = mu.t_stratified_loss(
                batch.t, batch_loss, loss_name=loss_name
            ) 
            for key, value in stratified_losses.items():
                self.log(f"train/{key}", value, batch_size=num_batch)
            
        train_loss = sum(total_losses.values())
        self.log('train/loss', train_loss, batch_size=num_batch)
        
        # Time
        step_time = time.time() - step_start_time
        self.log('train/examples_per_second', num_batch / step_time)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        batch_loss_dict = self.model_step(batch)
        total_losses = {
            k: torch.mean(v) for k,v in batch_loss_dict.items()
        }
        
        ####### Loggings #######
        num_batch = batch.num_graphs
        
        # Average time
        self.log(f"valid/t", np.mean(du.to_numpy(batch.t)), batch_size=num_batch, on_step=False, on_epoch=True, sync_dist=True)
        
        # Loss
        for loss_name, batch_loss in batch_loss_dict.items():
            # Total loss
            self.log(f"valid/{loss_name}", total_losses[loss_name], batch_size=num_batch, on_step=False, on_epoch=True, sync_dist=True)
        
            # Stratified loss
            stratified_losses = mu.t_stratified_loss(
                batch.t, batch_loss, loss_name=loss_name
            ) 
            for key, value in stratified_losses.items():
                self.log(f"valid/{key}", value, batch_size=num_batch, on_step=False, on_epoch=True, sync_dist=True)
        
        valid_loss = sum(total_losses.values())
        self.log('valid/loss', valid_loss, batch_size=num_batch, on_step=False, on_epoch=True, sync_dist=True)
    
    def on_predict_start(self):
        self.sampler = Sampler(self._cfg)
    
    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        batch_cart_coords, batch_num_atoms, batch_atom_types, batch_lattices = [], [], [], []

        for k in range(self._cfg.inference.num_samples):
            init_coords = batch.init_coords if 'init_coords' in batch else None
            coord_traj, lattice_traj = self.sampler.sample(batch=batch, init_coords=init_coords, model=self.model)
            coords = coord_traj[-1]
            lattice = lattice_traj[-1]

            batch_cart_coords.append(coords.detach().cpu())
            batch_lattices.append(lattice.detach().cpu())
            batch_num_atoms.append(batch.num_atoms.detach().cpu())
            batch_atom_types.append(batch.atom_types.detach().cpu())

        return {
            "cart_coords": torch.stack(batch_cart_coords, dim=0),   # [k, num_atoms, 3]
            "num_atoms": torch.stack(batch_num_atoms, dim=0),       # [k, num_graphs]
            "atom_types": torch.stack(batch_atom_types, dim=0),     # [k, num_atoms]
            "lattices": torch.stack(batch_lattices, dim=0),         # [k, num_graphs, 6]
            "gt_data": batch.to_data_list(),
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self._exp_cfg.optimizer
        )

        scheduler_cfg = self._exp_cfg.lr_scheduler
        scheduler_type = self._exp_cfg.lr_scheduler_type

        if scheduler_type == 'linear_warmup':
            scheduler = mu.LinearWarmupScheduler(optimizer, **scheduler_cfg.linear_warmup)
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_cfg.reduce_on_plateau
            )
        else:
            return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valid/loss'
        }
