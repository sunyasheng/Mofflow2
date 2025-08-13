from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame


# Custom Data class for MOF data
class MOFData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'bond_index':
            return self.num_atoms
        if key == 'canonical_torsion_tuples':
            return self.num_atoms
        return super().__inc__(key, value, *args, **kwargs)

# Custom collate function that pads 'rotable_atom_mask'
def custom_collate_fn(
        data_list: list, 
        follow_batch: Optional[List[str]],
        exclude_keys: Optional[List[str]],
    ) -> Batch:
    """
    Custom collate function to handle the data loading.
    """
    batch = Batch.from_data_list(data_list, follow_batch, exclude_keys)
    
    # Batch 'rotable_atom_mask'
    masks = [data.rotable_atom_mask for data in data_list]  # list of [num_rotable_bonds, num_atoms]

    total_rotable_bonds = sum([m.size(0) for m in masks])
    total_atoms = sum([m.size(1) for m in masks])

    total_rotable_atom_mask = torch.zeros((total_rotable_bonds, total_atoms), dtype=torch.bool, device=masks[0].device)

    rb_offset = 0
    atom_offset = 0
    for mask in masks:
        num_rotable_bonds, num_atoms = mask.shape
        total_rotable_atom_mask[rb_offset:rb_offset + num_rotable_bonds, atom_offset:atom_offset + num_atoms] = mask

        # Update offsets
        rb_offset += num_rotable_bonds
        atom_offset += num_atoms

    batch.rotable_atom_mask = total_rotable_atom_mask
    return batch

class CustomCollater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return custom_collate_fn(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

# DataLoader with custom collate function
class MOFDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=CustomCollater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )
