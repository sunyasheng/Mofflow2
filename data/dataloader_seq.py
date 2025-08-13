import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from utils.batching import DynamicBatchSampler, collate_fn


class MOFSequenceDatamodule(LightningDataModule):

    def __init__(self, *, data_cfg, train_dataset, valid_dataset, test_dataset=None):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self._datasets = {
            'train': train_dataset,
            'val': valid_dataset,
            'test': test_dataset
        }
    
    def _build_batch_sampler(self, dataset, is_train=False):
        is_distributed = is_train and torch.distributed.is_initialized()

        return DynamicBatchSampler(
            dataset=dataset,
            max_batch_units=self.loader_cfg.max_tokens,
            max_batch_size=self.loader_cfg.max_batch_size,
            distributed=is_distributed,
            shuffle=is_train,
            sort_key=lambda i: len(dataset[i][0]),
            use_heap=True
        )

    def _build_dataloader(self, split='train'):
        dataset = self._datasets[split]

        # Worker settings
        num_workers = self.loader_cfg.num_workers
        prefetch_factor = None if num_workers == 0 else self.loader_cfg.prefetch_factor
        persistent_workers = num_workers > 0

        is_train = split == 'train'

        # Set sampler
        sampler = self._build_batch_sampler(dataset, is_train=is_train)

        return DataLoader(
            dataset,
            batch_size=1,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(self):
        return self._build_dataloader('train')

    def val_dataloader(self):
        return self._build_dataloader('val')

    def test_dataloader(self):
        return self._build_dataloader('test')