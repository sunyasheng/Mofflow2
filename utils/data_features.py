"""
Utility functions from extracting tensor features from MOF data.
"""
import torch
import numpy as np
from utils import data as du
from utils import molecule as mu
from torch_geometric.utils import scatter
from pymatgen.core import Lattice, Molecule, SymmOp
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from mofdiff.common.data_utils import frac_to_cart_coords


def get_canonical_rotmat(lattice):
    """
    Returns:
        rotmat: numpy array of shape (3, 3)
            Rotation matrix to apply to Cartesian coordinates
        lattice_std: pymatgen.core.lattice.Lattice
            Standardized lattice
    """
    lattice_std = Lattice.from_parameters(*lattice.parameters)
    rotmat = np.linalg.inv(lattice_std.matrix) @ lattice.matrix
    
    return rotmat, lattice_std

def get_gt_coords(data, c_rotmat):
    """
    Returns:
        gt_coords: numpy array of shape (n_atoms, 3)
    """
    def _get_cart_coords_from_bb(bb):
        cart_coords = frac_to_cart_coords(
            bb.frac_coords, 
            bb.lengths,
            bb.angles,
            bb.num_atoms
        )
        return cart_coords

    # Gather ground-truth Cartesian coordinates
    gt_coords = [_get_cart_coords_from_bb(bb) for bb in data.pyg_mols]
    gt_coords = torch.cat(gt_coords, dim=0) # [num_atoms, 3]
    gt_coords = gt_coords @ c_rotmat.T

    # Compute building block centroids
    bb_num_vec = torch.tensor([bb.num_atoms for bb in data.pyg_mols])
    bb_vec = du.repeat_interleave(bb_num_vec)
    bb_centroids = scatter(gt_coords, bb_vec, dim=0, reduce='mean')
    bb_centroid_mean = bb_centroids.mean(dim=0, keepdim=True) # [1, 3]

    # Recenter coordinates
    gt_coords = gt_coords - bb_centroid_mean

    return gt_coords

def get_pga_info(gt_coords, atom_types, bb_num_vec):
    num_bbs = bb_num_vec.shape[0]
    centered_coords = du.apply_rototranslation(
        X_atoms=gt_coords,
        rotmats=torch.eye(3).expand(num_bbs, 3, 3), 
        trans=torch.zeros(num_bbs, 3), 
        bb_num_vec=bb_num_vec
    )
    
    # Split
    bb_centered_coords = torch.split(centered_coords, bb_num_vec.tolist())
    bb_atom_types = torch.split(atom_types, bb_num_vec.tolist())
    
    pg_types, symops = [], []
    for bb_atoms, bb_coords in zip(bb_atom_types, bb_centered_coords):
        C_1 = Molecule(bb_atoms, bb_coords)
        pga = PointGroupAnalyzer(C_1)
        pg_types.append(pga.sch_symbol)
        if pga.sch_symbol == "Kh": # single atom
            symops.append([SymmOp(np.eye(4))])
        else:
            symops.append(pga.get_symmetry_operations())
    return pg_types, symops, bb_centered_coords

def get_equiv_coords(pg_types, symops, bb_centered_coords):
    all_equiv_coords = []
    for i, bb_coords in enumerate(bb_centered_coords):
        bb_symops = symops[i]
        bb_sch_symbol = pg_types[i]
        equiv_coords = []
        if bb_sch_symbol == "D*h":
            # Identity
            equiv_coords.append(bb_coords)
            # Rotation by 180 degrees
            equiv_coords.append(bb_coords.flip(0))
        else:
            for sym_op in bb_symops:
                if np.abs(np.linalg.det(sym_op.rotation_matrix) - 1) < 1e-4:
                    bb_coords_1_equiv = sym_op.operate_multi(du.to_numpy(bb_coords))
                    equiv_coords.append(torch.tensor(bb_coords_1_equiv).float())
        equiv_coords = torch.stack(equiv_coords, dim=0) # [num_equiv, num_atoms, 3]
        all_equiv_coords.append(equiv_coords)
    return all_equiv_coords

def get_bb_mols_from_feats(feats):
    atom_type_list = torch.split(feats['atom_types'], feats['bb_num_vec'].tolist())
    coords_list = torch.split(feats['gt_coords'], feats['bb_num_vec'].tolist())
    is_metal_list = [mu.is_metal_bb(bb_atom_types) for bb_atom_types in atom_type_list]
    bb_mols = []
    for bb_atom_types, bb_coords, is_metal in zip(atom_type_list, coords_list, is_metal_list):
        bb_mol = mu.ob_mol_from_data(bb_atom_types, bb_coords, is_metal)
        bb_mol = mu.ob_to_rd(bb_mol)
        bb_mols.append(bb_mol)
    return bb_mols