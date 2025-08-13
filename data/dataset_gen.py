import os
import re
import json
import pickle
import torch
import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from utils import data as du
from utils import torsion as tu
from utils import conformer_matching as cu
from torch.utils.data import Dataset
from utils.pyg_data import MOFData


# Disable warnings
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore", message="SMILES provided.*")


class MOFGenDataset(Dataset):
    def __init__(self, *, metal_lib_path, mof_seqs_path, sample_limit=None):

        self._metal_lib_path = metal_lib_path
        self._mof_seqs_path = mof_seqs_path
        self._sample_limit = sample_limit
    
        self._load_seqs()
        self._load_metal_lib()

        # Load processed data
        suffix = '' if sample_limit is None else f'_{sample_limit}'
        self._processed_path = Path(self._mof_seqs_path).parent / f'processed_data{suffix}.pkl'
        if os.path.exists(self._processed_path):
            with open(self._processed_path, 'rb') as f:
                self._processed_data = pickle.load(f)
            print(f"Loaded processed data from {self._processed_path}")
        else:
            self._preprocess()
    
    def _load_seqs(self):
        with open(self._mof_seqs_path, 'r') as f:
            self._seq_dataset = json.load(f)

        # Limit sample size
        if self._sample_limit is not None:
            print(f"Limiting dataset to {self._sample_limit} samples.")
            self._seq_dataset = self._seq_dataset[:self._sample_limit]
    
    def _load_metal_lib(self):
        with open(self._metal_lib_path, 'rb') as f:
            self._metal_lib = pickle.load(f)['metal_bb_library']

    @staticmethod
    def _parse_gen_seqs(decoded_str):
        """
        Parses a decoded sequence into metal and organic building blocks.

        Returns:
            metal_bbs: list of metal building blocks (SMILES strings)
            organic_bbs: list of organic building blocks (SMILES strings)
        """
        if "<SEP>" not in decoded_str:
            return [], []

        metal_part, organic_part = decoded_str.split("<SEP>", maxsplit=1)

        # Strip special tokens and whitespace
        metal_part = metal_part.replace("<BOS>", "").replace("<EOS>", "").strip()
        organic_part = organic_part.replace("<EOS>", "").replace("<PAD>", "").strip()

        # Split on SMILES separator '.'
        metal_bbs = [s for s in metal_part.split(".") if s]
        organic_bbs = [s for s in organic_part.split(".") if s]

        # Re-insert '.' within metal building blocks
        bracketed_atom_regex = re.compile(r"\[[^\]]+\]")
        metal_bbs = [
            ".".join(bracketed_atom_regex.findall(block)) for block in metal_bbs
        ]

        return metal_bbs, organic_bbs
    
    def intialize_seq_to_mols(self, mof_seq: str):
        metal_str, organic_bbs = self._parse_gen_seqs(mof_seq)

        # Initialize structures
        metal_mols = [self._metal_lib[bb] for bb in metal_str]
        organic_mols = [cu.get_rd_conformer(bb) for bb in organic_bbs]

        # Reorder by MolWt
        metal_mols = sorted(metal_mols, key=lambda x: Descriptors.ExactMolWt(x))
        organic_mols = sorted(organic_mols, key=lambda x: Descriptors.ExactMolWt(x))
        bb_mols = metal_mols + organic_mols

        return bb_mols

    @staticmethod  
    def _featurize_mol_list(bb_mols):
        """
        Featurizes a list of RDKit molecules.

        Args:
            bb_mols: List of RDKit molecule objects.

        Returns:
            feats: Dictionary containing features.
        """
        feats = {}

        bb_num_vec = [mol.GetNumAtoms() for mol in bb_mols]

        # Atom types and coordinates
        all_coords = []
        all_atom_types = []

        for mol in bb_mols:
            coords = mol.GetConformer().GetPositions()  # np.array of shape (N_i, 3)
            atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]  # list of N_i ints

            all_coords.append(torch.tensor(coords, dtype=torch.float))  # (N_i, 3)
            all_atom_types.append(torch.tensor(atom_types, dtype=torch.long))  # (N_i,)

        # Add to feats
        feats['bb_num_vec'] = torch.tensor(bb_num_vec, dtype=torch.long)
        feats['init_coords'] = torch.cat(all_coords, dim=0)  # (N, 3)
        feats['atom_types'] = torch.cat(all_atom_types, dim=0)  # (N,)

        # Set bb_pos_idx
        bb_pos_idx = du.repeat_interleave(feats['bb_num_vec'])
        feats['bb_pos_idx'] = bb_pos_idx

        # Extract rotable bonds
        rotable_bond_data = tu.get_rotable_bond_data(bb_mols, bb_num_vec)
        feats['bb_mols'] = [Chem.MolToMolBlock(bb_mol) for bb_mol in bb_mols]
        for key, value in rotable_bond_data.items():
            feats[key] = value

        return feats
    
    def _process_one(self, seq_data):
        try:
            bb_mols = self.intialize_seq_to_mols(seq_data)
            feats = self._featurize_mol_list(bb_mols)

            return feats
        except Exception as e:
            return None
    
    def _preprocess(self):
        processed_data = []

        # Process sequence
        for data_idx, seq_data in enumerate(tqdm(self._seq_dataset, desc='Processing data')):
            feats = self._process_one(seq_data)
            if feats is not None:
                feats['data_idx'] = data_idx
                processed_data.append(feats)
        
        print(f"Number of valid samples: {len(processed_data)} / {len(self._seq_dataset)}")
        self._processed_data = processed_data
        
        # Save processed data
        with open(self._processed_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"Processed data saved to {self._processed_path}")

    def __len__(self):
        return len(self._processed_data)
    
    def __getitem__(self, idx):
        feats = self._processed_data[idx]
        data = MOFData(
            data_idx=feats['data_idx'], # Save the index for ordering cif files (during test time)
            num_nodes=len(feats['bb_num_vec']),
            num_atoms=feats['init_coords'].shape[0],
            num_bbs=len(feats['bb_num_vec']),
            num_rotable_bonds=feats['rotable_bond_mask'].sum().item(),
            init_coords=feats['init_coords'],
            bb_num_vec=feats['bb_num_vec'],
            bb_pos_idx=feats['bb_pos_idx'],
            atom_types=feats['atom_types'],
            atom_feats=feats['atom_feats'],
            bond_index=feats['bond_index'],
            bond_feats=feats['bond_feats'],
            rotable_bond_mask=feats['rotable_bond_mask'],
            rotable_atom_mask=feats['rotable_atom_mask'],
            canonical_torsion_tuples=feats['canonical_torsion_tuples'],
        )
        return data
