import math
import heapq
import torch
import random
import numpy as np
from collections import deque
from torch import distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
from typing import Any, Callable, Iterator, List, Optional


def collate_fn(batch, pad_token_id: int = 0):
    """
    Efficient collate function for MOFSequenceDataset.

    Args:
        batch: List of (input_ids, target_ids) tuples
        pad_token_id: int, padding index from tokenizer vocab

    Returns:
        dict with input_ids, target_ids, attention_mask
        - input_ids: Tensor of shape (batch_size, max_seq_len)
        - target_ids: Tensor of shape (batch_size, max_seq_len)
        - attention_mask: Tensor of shape (batch_size, max_seq_len)
    """
    input_ids, target_ids, props = zip(*batch)

    # Pad to the longest sequence in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=-100)  # For loss masking

    # Build attention mask (1 for real token, 0 for pad)
    attention_mask = (input_ids_padded != pad_token_id).bool()

    # Stack properties
    props_tensor = torch.stack(props, dim=0)

    return {
        "input_ids": input_ids_padded,
        "target_ids": target_ids_padded,
        "attention_mask": attention_mask,
        "props": props_tensor
    }

class OverfitSampler(Sampler):
    def __init__(self, num_samples: int, dataset_len: int):
        self.num_samples = num_samples
        self.dataset_len = dataset_len

    def __iter__(self):
        idx_list = list(range(self.dataset_len))
        num_repeats = math.ceil(self.num_samples / self.dataset_len)
        full_list = (idx_list * num_repeats)[:self.num_samples]
        return iter(random.sample(full_list, len(full_list)))

    def __len__(self):
        return self.num_samples
    
class DynamicBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        max_batch_units: int,
        max_batch_size: Optional[int] = None,
        drop_last: bool = False,
        distributed: bool = False,
        sort_key: Callable = None,
        buffer_size_multiplier: int = 100,
        shuffle: bool = False,
        use_heap: bool = False,
    ):
        """
        Batch sampler that dynamically groups samples into batches based on a user-defined size metric (e.g., number of tokens, atoms, nodes).
        Each batch is constructed to stay within a maximum total unit count (`max_batch_units`), allowing for efficient batching of variable-sized data.

        Examples:
        - For tokenized text sequences: sort_key = lambda i: len(dataset[i][0])  # number of tokens
        - For molecular graphs: sort_key = lambda i: dataset[i].num_atoms        # number of atoms
        - For general graphs: sort_key = lambda i: dataset[i].num_nodes          # number of nodes
        """
        self.distributed = distributed
        if distributed:
            self.sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            self.sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

        super().__init__(self.sampler, batch_size=1, drop_last=drop_last)

        self.max_batch_units = max_batch_units
        self.max_batch_size = max_batch_size
        self.sort_key = sort_key
        self.max_buffer_size = max_batch_units * buffer_size_multiplier
        self._epoch = 0
        self.shuffle = shuffle
        self.use_heap = use_heap # Useful if padding is involved (e.g., for tokens)
        self.drop_last = drop_last

        self.bucket_batches = []

    def __len__(self):
        if not self.bucket_batches:
            self._build_batches()
        return len(self.bucket_batches)

    def __iter__(self):
        self._build_batches()
        for batch, _ in self.bucket_batches:
            yield batch

    def _build_batches(self):
        buffer = []
        buffer_deque = deque()  # Use deque for FIFO when use_heap=False
        buffer_size = 0

        batch = []
        batch_units = 0

        bucket_batches = []

        indices = list(self.sampler)
        for index in indices:
            # Add to buffer
            num_units = self.sort_key(index)
            if self.use_heap:
                # Store negative to simulate max-heap (largest first)
                heapq.heappush(buffer, (-num_units, index))
            else:
                buffer_deque.append((num_units, index))
            buffer_size += num_units

            # Flush buffer if exceeds max buffer size
            while buffer_size > self.max_buffer_size:
                if self.use_heap:
                    neg_units, index = heapq.heappop(buffer)
                    num_units = -neg_units
                else:
                    num_units, index = buffer_deque.popleft()
                buffer_size -= num_units

                # Check batch constraints
                if (batch_units + num_units > self.max_batch_units) or \
                   (self.max_batch_size and len(batch) >= self.max_batch_size):
                    bucket_batches.append((batch, batch_units))
                    batch, batch_units = [], 0
                batch.append(index)
                batch_units += num_units

        # Process remaining elements in buffer
        while buffer if self.use_heap else buffer_deque:
            if self.use_heap:
                neg_units, index = heapq.heappop(buffer)
                num_units = -neg_units
            else:
                num_units, index = buffer_deque.popleft()
            if (batch_units + num_units > self.max_batch_units) or \
               (self.max_batch_size and len(batch) >= self.max_batch_size):
                bucket_batches.append((batch, batch_units))
                batch, batch_units = [], 0

            batch.append(index)
            batch_units += num_units

        # Handle last batch
        if batch and not self.drop_last:
            bucket_batches.append((batch, batch_units))

        # Extra randomization for use_heap
        if self.shuffle and self.use_heap:
            np.random.shuffle(bucket_batches)

        # DDP synchronization
        if self.distributed:
            # Communicate the number of batches across processes
            num_batches = torch.tensor(len(bucket_batches), device='cuda')
            dist.all_reduce(num_batches, op=dist.ReduceOp.MIN)
            num_batches = num_batches.item()

            # Truncate to the minimum number of batches across all processes
            if len(bucket_batches) > num_batches:
                bucket_batches = bucket_batches[:num_batches]
        
        self.bucket_batches = bucket_batches

    def set_epoch(self, epoch):
        self._epoch = epoch
        if self.distributed:
            self.sampler.set_epoch(epoch)