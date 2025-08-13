import logging
import math
import numpy as np
import torch
import random
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from utils.pyg_data import MOFDataLoader
from utils.batching import DynamicBatchSampler, OverfitSampler


class MOFDatamodule(LightningDataModule):

    def __init__(self, *, data_cfg, train_dataset, valid_dataset, pred_dataset=None):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self._datasets = {
            'train': train_dataset,
            'val': valid_dataset,
            'pred': pred_dataset
        }
    
    def _build_dynamic_batch_sampler(self, dataset, is_train=False):
        is_distributed = is_train and torch.distributed.is_initialized()

        dynamic_cfg = self.loader_cfg.dynamic
        return DynamicBatchSampler(
            dataset=dataset,
            max_batch_units=dynamic_cfg.max_num_atoms,
            max_batch_size=dynamic_cfg.max_batch_size,
            distributed=is_distributed,
            shuffle=is_train,
            sort_key=lambda i: dataset.num_units[i],
        )
    
    def _build_dataloader(self, split='train'):
        dataset = self._datasets[split]
        batch_size = self.loader_cfg.batch_size.train # Ignored if batch_sampler is not None
        sampler_type = self.loader_cfg.sampler

        # Worker settings
        num_workers = self.loader_cfg.num_workers
        prefetch_factor = None if num_workers == 0 else self.loader_cfg.prefetch_factor
        persistent_workers = num_workers > 0

        is_train = split == 'train'

        # Set sampler
        sampler = None
        batch_sampler = None
        if sampler_type == 'overfit' and is_train:
            sampler = OverfitSampler(
                num_samples=self.loader_cfg.overfit.num_samples,
                dataset_len=len(dataset)
            )
        elif sampler_type == 'dynamic':
            batch_sampler = self._build_dynamic_batch_sampler(
                dataset,
                is_train=is_train
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size if batch_sampler is None else 1,
            sampler=sampler,
            batch_sampler=batch_sampler,
            shuffle=is_train if sampler is None and batch_sampler is None else False,
            exclude_keys=['rotable_atom_mask'],
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=persistent_workers
        )
        
    def train_dataloader(self):
        return self._build_dataloader('train')

    def val_dataloader(self):
        return self._build_dataloader('val')

    def predict_dataloader(self):
        # Custom dataloader that diagonally concatenates 'rotable_atom_mask' for batch processing
        return MOFDataLoader(
            self._datasets['pred'],
            batch_size=self.loader_cfg.batch_size.predict,
            shuffle=False,
            exclude_keys=['rotable_atom_mask'],
            num_workers=self.loader_cfg.num_workers,
            prefetch_factor=self.loader_cfg.prefetch_factor,
            pin_memory=True,
            persistent_workers=self.loader_cfg.num_workers > 0
        )