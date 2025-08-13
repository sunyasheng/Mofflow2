import math
import copy
import torch
import itertools
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from torch_geometric.data import Data
from utils import so3 as su
from utils import molecule as mu


# Interpolation
def sym_mod(a, b=(2 * math.pi)):
  return (a + b / 2) % b - b / 2

def log_map(x0, x1):
  return sym_mod(x1 - x0)

def get_xt(x0, x1, t):
  log_x1 = log_map(x0, x1)
  return sym_mod(t * log_x1 + x0)

def get_ut(xt, x1, t):
  log_xt_x1 = sym_mod(x1 - xt)
  return log_xt_x1 / (1 - t)

# Get/set dihedrals
def get_dihedrals(pos: torch.Tensor, torsion_tuples: torch.Tensor) -> torch.Tensor:
    """
    Compute dihedral angles from atom positions and torsion tuples.
    
    Args:
        pos (torch.Tensor): [N, 3] atom positions
        torsion_tuples (torch.Tensor): [num_dihedrals, 4] of atom indices

    Returns:
        torch.Tensor: [num_dihedrals] dihedral angles in [-π, π)
    """
    i, j, k, l = torsion_tuples.unbind(dim=1)

    p0 = pos[i]  # [B, 3]
    p1 = pos[j]
    p2 = pos[k]
    p3 = pos[l]

    r_ij = p1 - p0
    r_jk = p2 - p1
    r_kl = p3 - p2

    n_ijk = torch.cross(r_ij, r_jk, dim=1)  # [B, 3]
    n_jkl = torch.cross(r_jk, r_kl, dim=1)

    m = torch.cross(n_ijk, r_jk, dim=1)

    # Compute norms
    norm_n_ijk = torch.norm(n_ijk, dim=1)
    norm_n_jkl = torch.norm(n_jkl, dim=1)
    norm_m = torch.norm(m, dim=1)

    # Avoid divide-by-zero (numerical stability)
    eps = 1e-8
    norm_n_ijk = norm_n_ijk.clamp(min=eps)
    norm_n_jkl = norm_n_jkl.clamp(min=eps)
    norm_m = norm_m.clamp(min=eps)

    x = (n_ijk * n_jkl).sum(dim=1) / (norm_n_ijk * norm_n_jkl)
    y = (m * n_jkl).sum(dim=1) / (norm_m * norm_n_jkl)

    angles = -torch.atan2(y, x)
    return angles

def get_canonical_order(mol):
    mol_cp = copy.deepcopy(mol)
    ranks = Chem.CanonicalRankAtoms(mol_cp, includeChirality=True)
    
    # Sort atoms by rank
    order = sorted(range(len(ranks)), key=lambda i: ranks[i])
    return order

def get_canonical_torsion_tuples_from_bonds(mol: Chem.Mol, rotable_bonds: torch.Tensor) -> torch.Tensor:
    """
    Return a torch tensor of shape [num_rotable_bonds, 4] with torsion tuples (i, j, k, l),
    where j–k is a rotatable bond, and i/l are canonical neighbors selected using CanonicalRankAtoms.
    """
    if rotable_bonds.numel() == 0:
        return torch.empty(0, 4, dtype=torch.long)

    atom_ranks = Chem.CanonicalRankAtoms(mol, includeChirality=True)
    torsion_tuples = []

    for bond in rotable_bonds.T:
        j, k = bond[0].item(), bond[1].item()
        atom_j = mol.GetAtomWithIdx(j)
        atom_k = mol.GetAtomWithIdx(k)

        j_neighbors = [nbr.GetIdx() for nbr in atom_j.GetNeighbors() if nbr.GetIdx() != k]
        k_neighbors = [nbr.GetIdx() for nbr in atom_k.GetNeighbors() if nbr.GetIdx() != j]

        if not j_neighbors or not k_neighbors:
            continue

        i = min(j_neighbors, key=lambda x: atom_ranks[x])
        l = min(k_neighbors, key=lambda x: atom_ranks[x])

        torsion_tuples.append((i, j, k, l))

    return torch.tensor(torsion_tuples, dtype=torch.long)

# Combine features from multiple building blocks
def get_rotable_bond_data(bb_mols, bb_num_vec):
    total_num_atoms = sum(bb_num_vec)

    total_atom_attr = []
    total_bond_index = []
    total_bond_attr = []
    total_bond_mask = []
    total_atom_mask = []
    total_torsion_tuples = []

    offset = 0
    for i, bb_num_atoms in enumerate(bb_num_vec):
        bb_feats = mu.featurize_mol(bb_mols[i])
        bond_mask, atom_mask = mu.get_transformation_mask(bb_feats, bb_mols[i])

        # Get canonical torsion tuples
        canonical_tuples = get_canonical_torsion_tuples_from_bonds(
            mol=bb_mols[i], 
            rotable_bonds=bb_feats.edge_index[:, bond_mask]
        )
        
        # Get features
        bb_bond_index = bb_feats.edge_index # [2, num_bonds]
        bb_bond_attr = bb_feats.edge_attr # [num_bonds, num_bond_features]
        bb_atom_attr = bb_feats.node_attr # [num_atoms, num_atom_features]
        
        # Add offsets
        bb_bond_index = bb_bond_index + offset
        canonical_tuples = canonical_tuples + offset
        
        # Append
        total_atom_attr.append(bb_atom_attr)
        total_bond_index.append(bb_bond_index)
        total_bond_attr.append(bb_bond_attr)
        total_bond_mask.append(bond_mask)
        total_torsion_tuples.append(canonical_tuples)
        
        # Pad atom_mask from shape [num_rotatable_bonds, bb_num_atoms] to [num_rotatable_bonds, total_num_atoms]
        num_rotatable_bonds = atom_mask.shape[0]
        global_atom_mask = torch.zeros(num_rotatable_bonds, total_num_atoms).bool()
        global_atom_mask[:, offset:offset + bb_num_atoms] = atom_mask
        total_atom_mask.append(global_atom_mask)
        
        # Update offset
        offset += bb_num_atoms

    # Concatenate
    total_atom_attr = torch.cat(total_atom_attr, dim=0)                # [total_num_atoms, num_atom_features]
    total_bond_index = torch.cat(total_bond_index, dim=1)              # [2, total_num_bonds]
    total_bond_attr = torch.cat(total_bond_attr, dim=0)                # [total_num_bonds, num_bond_features]
    total_bond_mask = torch.cat(total_bond_mask, dim=0).bool()         # [total_num_bonds]
    total_atom_mask = torch.cat(total_atom_mask, dim=0).bool()         # [total_num_rotatable_bonds, total_num_atoms]
    total_torsion_tuples = torch.cat(total_torsion_tuples, dim=0)      # [total_num_rotatable_bonds, 4]
    
    return Data(
        atom_feats=total_atom_attr,
        bond_index=total_bond_index,
        bond_feats=total_bond_attr,
        rotable_bond_mask=total_bond_mask,
        rotable_atom_mask=total_atom_mask,
        canonical_torsion_tuples=total_torsion_tuples,
    )

# Apply torsion updates to atom coordinates
def apply_relative_torsion(pos, rotable_bond_index, rotable_atom_mask, torsion_updates):
    """
    Args:
    - pos (torch.Tensor): tensor of atom coordinates [num_atoms, 3]
    - rotable_bond_index (torch.Tensor): tensor of bond indices [2, num_rotable_bonds]
    - rotable_atom_mask (torch.Tensor): tensor of atom rotation masks [num_rotable_bonds, num_atoms]
    - torsion_updates (torch.Tensor): tensor of torsion updates [num_rotable_bonds]

    Returns:
    - torch.Tensor: tensor of updated atom coordinates [num_atoms, 3]
    """
    pos = pos.clone()
    num_rotable_bonds = rotable_bond_index.shape[1]

    for bond_idx in range(num_rotable_bonds):
        if torsion_updates[bond_idx] == 0:
            continue
        
        # Get the atom indices for the current bond
        u = rotable_bond_index[0, bond_idx].item()
        v = rotable_bond_index[1, bond_idx].item()
        
        # Check rotation mask conditions.
        # The convention is that atom 'v' is the pivot for rotation.
        if rotable_atom_mask[bond_idx, u] or (not rotable_atom_mask[bond_idx, v]):
            print("Mask rotate exception")
        
        # Compute the rotation vector (difference between positions).
        rotvec = pos[u] - pos[v]
        # Scale the rotation vector by the torsion update (angle in radians).
        rotvec = rotvec * torsion_updates[bond_idx] / torch.norm(rotvec)
        rotmat = su.rotvec_to_rotmat(rotvec.unsqueeze(0)).squeeze(0)
    
        # Select atoms to be rotated using the provided boolean mask.
        mask = rotable_atom_mask[bond_idx] # shape: [num_atoms]
        # Apply rotation around the pivot (pos[v]):
        pos[mask] = (pos[mask] - pos[v]) @ rotmat.T + pos[v]
    
    return pos

def apply_absolute_torsion(pos, rotable_atom_mask, torsion_targets, torsion_tuples):
    pos = pos.clone()
    num_rotable_bonds = torsion_tuples.shape[0]

    current_angles = get_dihedrals(pos, torsion_tuples)

    for bond_idx in range(num_rotable_bonds):
        _, j, k, _ = torsion_tuples[bond_idx]
        j, k = j.item(), k.item()

        current_angle = current_angles[bond_idx]
        target_angle = torsion_targets[bond_idx]
        delta = sym_mod(target_angle - current_angle)

        if abs(delta) < 1e-6:
            continue

        if rotable_atom_mask[bond_idx, j] or not rotable_atom_mask[bond_idx, k]:
            print("Mask rotate exception")

        axis = pos[j] - pos[k]
        axis = axis / axis.norm()
        rotvec = axis * delta
        rotmat = su.rotvec_to_rotmat(rotvec.unsqueeze(0)).squeeze(0)

        mask = rotable_atom_mask[bond_idx]
        pos[mask] = (pos[mask] - pos[k]) @ rotmat + pos[k]

    return pos

# Convert angle to cos/sin representation
def angle_to_cos_sin(angle):
    """
    Args:
    - angle (torch.Tensor): [B], tensor of angles in radians

    Returns:
    - torch.Tensor: [B, 2], tensor of sin/cos representation of angles
    """
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)

# Convert cos/sin representation to angle
def cos_sin_to_angle(cos_sin):
    """
    Args:
    - cos_sin (torch.Tensor): [B, 2], tensor of sin/cos representation of angles

    Returns:
    - torch.Tensor: [B], tensor of angles in radians
    """
    return torch.atan2(cos_sin[:, 1], cos_sin[:, 0])

# Get symmetry for each rotable bond
def get_torsion_symmetry_scores(
    bb_mol: Chem.rdchem.Mol,
    bond_index: torch.Tensor,
    rotable_bond_mask: torch.Tensor,
    rotable_atom_mask: torch.Tensor,
    RMSD_THRESHOLD: float = 0.1,
) -> torch.Tensor:
    """
    Returns:
    - symmetry_scores (torch.Tensor): [num_rotable_bonds], tensor of symmetry scores for each rotable bond
        1: no symmetry
        2: pi symmetry
        3: 2*pi/3 symmetry (e.g., CH3)
    """
    REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)

    coords = torch.tensor(bb_mol.GetConformer().GetPositions()).float()
    atom_types = torch.tensor([atom.GetAtomicNum() for atom in bb_mol.GetAtoms()])

    rotable_bonds = bond_index[:, rotable_bond_mask]

    symmetry_scores = []
    for bond_idx, rotable_bond in enumerate(rotable_bonds.T):
        sym_score = 1 # default: no symmetry
        rotating_mask = rotable_atom_mask[bond_idx]

        # Check for pi and 2*pi/3 symmetry
        try:
            for angle, val in zip([np.pi, 2 * np.pi / 3], [2, 3]):
                # Check for symmetry
                twist_coords = apply_relative_torsion(
                    pos=coords,
                    rotable_bond_index=rotable_bond[:, None],
                    rotable_atom_mask=rotating_mask[None, :],
                    torsion_updates=torch.tensor([angle])
                )
                twist_mol = mu.ob_mol_from_data(atom_types=atom_types, positions=twist_coords)
                twist_mol = mu.ob_to_rd(twist_mol)

                # Calculate RMSD
                rmsd = AllChem.GetBestRMS(REMOVE_HS(twist_mol), REMOVE_HS(bb_mol))

                if rmsd < RMSD_THRESHOLD:
                    sym_score = val
                    break
        except Exception as e:
            pass

        symmetry_scores.append(sym_score)

    return torch.tensor(symmetry_scores).float()
