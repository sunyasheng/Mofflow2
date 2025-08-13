import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from openbabel import openbabel as ob
from openbabel import pybel
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from mofdiff.common.constants import METALS


DIHEDRAL_PATTERN = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
CHIRALITY = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
METALS = torch.tensor(METALS).long()


def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def get_atom_features(atom: Chem.Atom, ring_info: Chem.RingInfo) -> list:
    """
    Args:
        atom (Chem.Atom): The RDKit atom object.
        ring_info (Chem.RingInfo): The RDKit ring information for the molecule.
    
    Returns:
        list: A list of features representing the atom.
    """
    features = []
    
    # Atomic number, aromaticity
    features.extend([atom.GetAtomicNum(), 1 if atom.GetIsAromatic() else 0])
    
    # Atom degree
    features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
    
    # Hybridization type
    features.extend(one_k_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]))
    
    # Implicit valence
    features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
    
    # Formal charge
    features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
    
    # Ring-related features
    atom_idx = atom.GetIdx()
    features.extend([
        int(ring_info.IsAtomInRingOfSize(atom_idx, size)) for size in range(3, 9)
    ])
    features.extend(one_k_encoding(int(ring_info.NumAtomRings(atom_idx)), [0, 1, 2, 3]))
    
    return features

def featurize_mol(mol: Chem.Mol):
    num_atoms = mol.GetNumAtoms()
    
    # Atom features
    atom_features = []
    atomic_number = []
    chiral_tag = []
    ring = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        chiral_tag.append(CHIRALITY[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())        
        atom_features.extend(get_atom_features(atom, ring))
    node_attr = torch.tensor(atom_features).view(num_atoms, -1).float()

    # Bond features
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BONDS[bond.GetBondType()]]

    edge_index = torch.tensor([row, col]).long()
    edge_attr = F.one_hot(torch.tensor(edge_type).long(), num_classes=len(BONDS)).float()
        
    return Data(
        node_attr=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

def get_transformation_mask(pyg_data: Data, mol: Chem.rdchem.Mol):

    # No rotable bonds in linear molecules
    if is_molecule_linear(mol):
        num_bonds = pyg_data.edge_index.shape[1]
        mask_edges = torch.zeros(num_bonds).bool()
        mask_rotate = torch.zeros((0, mol.GetNumAtoms())).bool()
        return mask_edges, mask_rotate

    G = to_networkx(pyg_data, to_undirected=False)
    to_rotate = []
    edges = pyg_data.edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    mask_edges = torch.tensor(mask_edges).bool()
    mask_rotate = torch.tensor(mask_rotate).bool()
    return mask_edges, mask_rotate

def ob_mol_from_data(atom_types: torch.Tensor, positions: torch.Tensor, is_metal=False):
    """
    Create an OpenBabel molecule object from atom types and positions.
    """
    atom_types = atom_types.numpy()
    positions = positions.numpy()
    
    mol = pybel.ob.OBMol()
    for atom_type, position in zip(atom_types, positions):
        atom = pybel.ob.OBAtom()
        atom.SetAtomicNum(int(atom_type))
        atom.SetVector(*position.tolist())
        mol.AddAtom(atom)

    # Add bonds for organic molecules
    if not is_metal:
        # Determine connectivity based on distance
        mol.ConnectTheDots()
        
        # Identify aromatic nitrogens
        for atom in ob.OBMolAtomIter(mol):
            if "N" in atom.GetType() and atom.IsInRing():
                atom.SetAromatic()
        
        # Assign bond orders
        mol.PerceiveBondOrders()
    return pybel.Molecule(mol)

def ob_to_rd(ob_mol):
    """
    Convert an OpenBabel molecule to an RDKit molecule.
    """
    rd_mol = Chem.MolFromMol2Block(ob_mol.write("mol2"), removeHs=False)
    return rd_mol

def rd_to_ob(rd_mol):
    """
    Convert an RDKit molecule to an OpenBabel molecule.
    """
    return pybel.readstring("mol", Chem.MolToMolBlock(rd_mol))

def is_molecule_linear(ob_mol: pybel.Molecule, angle_tolerance=10.0):
    """
    Check whether a molecule is linear. Adapted from _is_molecule_linear function of InchiMolAtom Mapper class (pymatgen). 

    Args:
        ob_mol: OpenBabel pybel.Molecule object.
        angle_tolerance: Tolerance for angle deviation from 180 degrees to consider a molecule linear.
    
    Returns:
        bool: True if molecule is linear, False otherwise.
    """
    if isinstance(ob_mol, Chem.rdchem.Mol):
        ob_mol = rd_to_ob(ob_mol)

    if ob_mol.OBMol.NumAtoms() < 3:
        return True
    a1 = ob_mol.OBMol.GetAtom(1)
    a2 = ob_mol.OBMol.GetAtom(2)
    for idx in range(3, ob_mol.OBMol.NumAtoms() + 1):
        angle = float(ob_mol.OBMol.GetAtom(idx).GetAngle(a2, a1))
        if angle < 0.0:
            angle = -angle
        if angle > 90.0:
            angle = 180.0 - angle
        if angle > angle_tolerance:
            return False
    return True

def is_metal_bb(bb_mol: Chem.rdchem.Mol) -> bool:
    """
    Returns:
    - bool: True if the molecule contains a metal atom, False otherwise
    """
    if isinstance(bb_mol, Chem.rdchem.Mol):
        bb_atom_types = torch.tensor([atom.GetAtomicNum() for atom in bb_mol.GetAtoms()]).long()
    elif isinstance(bb_mol, torch.Tensor):
        bb_atom_types = bb_mol
    else:
        raise ValueError(f"Invalid type: {type(bb_mol)}")
    return torch.any(torch.isin(bb_atom_types, METALS)).item()
        