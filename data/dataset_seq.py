import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from utils.lmdb import read_lmdb
from data.tokenizer import SmilesTokenizer
from pytorch_lightning.utilities.rank_zero import rank_zero_info


class MOFSequenceDataset(Dataset):
    def __init__(self, *, dataset_cfg, split='train'):
        self._dataset_cfg = dataset_cfg

        self.prefix = dataset_cfg.dataset_prefix
        self.max_len = dataset_cfg.max_len
        self.seq_dir = dataset_cfg.seq_dir
        self.vocab_path = dataset_cfg.vocab_path

        # Load dataset
        self._load_dataset(split=split)

        # Load tokenizer
        self._load_tokenizer()

    def _load_tokenizer(self):
        tokenizer = SmilesTokenizer()
        if not os.path.exists(self.vocab_path):
            print(f"INFO:: Building vocab to {self.vocab_path}...")
            tokenizer.build_vocab(self._load_dataset(split='train'))
            tokenizer.save_vocab(self.vocab_path)
        else:
            print(f"INFO:: Loading vocab from {self.vocab_path}...")
            tokenizer.load_vocab(self.vocab_path)
        self.tokenizer = tokenizer
   
    def _load_dataset(self, split):
        with open(f"{self.seq_dir}/{self.prefix}_{split}.json", 'r') as f:
            dataset = json.load(f)
        self.dataset = list(dataset.values())
        print(f"INFO:: {len(self.dataset)} datapoints in {split} split.")

        if split == 'train':
            sample_limit = self._dataset_cfg.train_sample_limit
        elif split == 'val':
            sample_limit = self._dataset_cfg.val_sample_limit
        else:
            sample_limit = None
        
        if sample_limit is not None:
            self.dataset = self.dataset[:sample_limit]
            rank_zero_info(f"INFO:: Limiting {split} split to {sample_limit} samples.")

        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mof_seq = self.dataset[idx]
        token_ids = self.tokenizer.encode(mof_seq)

        if self.max_len is not None:
            token_ids = token_ids[:self.max_len]

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)

        return input_ids, target_ids
