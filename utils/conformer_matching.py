import copy
import warnings
import networkx as nx
import numpy as np
import torch
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdDistGeom
from scipy.optimize import differential_evolution
from utils import molecule as mu


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def SetDihedral(conf, atom_idx, new_value):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_value)

def apply_changes(mol, values, rotable_bonds, conf_id):
    opt_mol = copy.copy(mol)
    [SetDihedral(opt_mol.GetConformer(conf_id), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]
    return opt_mol

def get_torsion_angles(mol):
    torsions_list = []
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)
    nodes = set(G.nodes())
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0])
        )
    return torsions_list

def get_rmsd(q: Chem.rdchem.Mol, p: Chem.rdchem.Mol, q_conf_id=0, p_conf_id=0):
    """
    Using the Kabsch algorithm the alignment of two molecules (P, Q)
    happens in three steps:
    - translate the P and Q into their centroid
    - compute of the optimal rotation matrix (U) using Kabsch algorithm
    - compute the translation (V) and rmsd.

    The function returns the rotation matrix (U), translation vector (V),
    and RMSD between Q and P', where P' is:

        P' = P * U + V

    Args:
        p: a `Molecule` object what will be matched with the target one.

    Returns:
        U: Rotation matrix (D,D)
        V: Translation vector (D)
        RMSD : Root mean squared deviation between P and Q
    """
    def kabsch(P: np.ndarray, Q: np.ndarray):
        """
        Kabsch alignment of two sets of points.
        """
        C = np.dot(P.T, Q)
        V, _S, WT = np.linalg.svd(C)
        det = np.linalg.det(np.dot(V, WT))
        return np.dot(np.dot(V, np.diag([1, 1, det])), WT)

    if q.GetNumAtoms() != p.GetNumAtoms():
        raise ValueError("Molecules have different number of atoms.")
    
    q_coord = q.GetConformer(q_conf_id).GetPositions()
    p_coord = p.GetConformer(p_conf_id).GetPositions()
    
    p_trans, q_trans = p_coord.mean(axis=0), q_coord.mean(axis=0)
    p_centroid, q_centroid = p_coord - p_trans, q_coord - q_trans
    
    U = kabsch(p_centroid, q_centroid)
    
    p_prime_centroid = np.dot(p_centroid, U)
    rmsd = np.sqrt(np.mean(np.square(p_prime_centroid - q_centroid)))
    
    V = q_trans - np.dot(p_trans, U)
    
    return U, V, rmsd
    
class OptimizeConformer:
    def __init__(self, mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=None):
        super(OptimizeConformer, self).__init__()
        if seed:
            np.random.seed(seed)
        self.rotable_bonds = rotable_bonds
        self.mol = mol
        self.true_mol = true_mol
        self.probe_id = probe_id
        self.ref_id = ref_id

    def score_conformation(self, values):
        for i, r in enumerate(self.rotable_bonds):
            SetDihedral(self.mol.GetConformer(self.probe_id), r, values[i])
        _, _, rmsd = get_rmsd(self.true_mol, self.mol, self.ref_id, self.probe_id)
        return rmsd

def optimize_rotable_bonds(mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=0, popsize=15, maxiter=500,
                             mutation=(0.5, 1), recombination=0.8):
    mol = copy.deepcopy(mol) # Avoid modifying mol
    opt = OptimizeConformer(mol, true_mol, rotable_bonds, seed=seed, probe_id=probe_id, ref_id=ref_id)
    max_bound = [np.pi] * len(opt.rotable_bonds)
    min_bound = [-np.pi] * len(opt.rotable_bonds)
    bounds = (min_bound, max_bound)
    bounds = list(zip(bounds[0], bounds[1]))

    # Optimize conformations
    result = differential_evolution(opt.score_conformation, bounds,
                                    maxiter=maxiter, popsize=popsize,
                                    mutation=mutation, recombination=recombination, disp=False, seed=seed)
    opt_mol = apply_changes(opt.mol, result['x'], opt.rotable_bonds, conf_id=probe_id)

    return opt_mol

def get_rd_conformer(gt_mol, steps=200):

    def get_failure_causes(counts):
        messages = []
        for i, k in enumerate(rdDistGeom.EmbedFailureCauses.names):
            messages.append(f"{k}: {counts[i]}")
        return "\n".join(messages) 
    
    if isinstance(gt_mol, Chem.rdchem.Mol):
        rd_mol = copy.deepcopy(gt_mol)
        rd_mol = Chem.AddHs(rd_mol)
    elif isinstance(gt_mol, pybel.Molecule):
        rd_mol = Chem.MolFromMol2Block(gt_mol.write("mol2"), removeHs=False)
    elif isinstance(gt_mol, str):
        warnings.warn("SMILES provided. Atom orders will not be preserved.", UserWarning)
        rd_mol = Chem.MolFromSmiles(gt_mol)
        rd_mol = Chem.AddHs(rd_mol)
    else:
        raise ValueError("Invalid molecule object.")
    
    # Remove Conformers
    [rd_mol.RemoveConformer(i) for i in range(rd_mol.GetNumConformers())]
    
    # Embed molecule
    # AllChem.EmbedMolecule(rd_mol)
    ps = rdDistGeom.ETKDGv3()
    ps.trackFailures = True
    val = rdDistGeom.EmbedMolecule(rd_mol, ps)
    if val < 0:
        counts = ps.GetFailureCounts()
        msg = get_failure_causes(counts)
        raise ValueError(f"Embedding molecule failed:\n{msg}")

    # Optimize with MMFF
    AllChem.MMFFOptimizeMolecule(rd_mol, maxIters=steps)
    
    return rd_mol

def conformer_matching(mol_list, steps=200, popsize=15, maxiter=15):
    REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)
    
    def align_mol(mol, U, V):
        pred_coords = mol.GetConformer().GetPositions()
        pred_coords = np.dot(pred_coords, U) + V
        mol.GetConformer().SetPositions(pred_coords)

    rmsd_list = []
    conformer_list = []
    
    for mol in mol_list:
        # Generate a conformer
        pred_mol = get_rd_conformer(mol, steps=steps)
        rotable_bonds = get_torsion_angles(pred_mol)
        
        # Skip optimization if no rotable bonds or linear molecule
        if len(rotable_bonds) == 0 or mu.is_molecule_linear(pred_mol):
            U, V, rmsd = get_rmsd(REMOVE_HS(mol), REMOVE_HS(pred_mol))
            align_mol(pred_mol, U, V)

            rmsd_list.append(rmsd)
            conformer_list.append(pred_mol)
            continue
        
        try:
            # Optimize torsion angles
            opt_mol = optimize_rotable_bonds(pred_mol, mol, rotable_bonds, popsize=popsize, maxiter=maxiter)
            U, V, rmsd = get_rmsd(REMOVE_HS(mol), REMOVE_HS(opt_mol))
            align_mol(opt_mol, U, V)
            
            rmsd_list.append(rmsd)
            conformer_list.append(opt_mol)
        except Exception as e:
            print(f"Torsion angle optimization failed: {e}")

            rmsd_list.append(None)
            conformer_list.append(pred_mol)
    
    return rmsd_list, conformer_list

def conformer_matching_single(mol, steps=200, popsize=15, maxiter=15):
    REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)
    
    def align_mol(mol_to_align, U, V):
        coords = mol_to_align.GetConformer().GetPositions()
        coords = np.dot(coords, U) + V
        mol_to_align.GetConformer().SetPositions(coords)

    # Generate an initial conformer
    pred_mol = get_rd_conformer(mol, steps=steps)
    rotable_bonds = get_torsion_angles(pred_mol)
    
    # Skip optimization if no rotable bonds or linear molecule
    if len(rotable_bonds) == 0 or mu.is_molecule_linear(pred_mol):
        U, V, rmsd = get_rmsd(REMOVE_HS(mol), REMOVE_HS(pred_mol))
        align_mol(pred_mol, U, V)
        return rmsd, pred_mol
    
    # Try to optimize the torsion angles
    try:
        opt_mol = optimize_rotable_bonds(
            pred_mol,
            mol,
            rotable_bonds,
            popsize=popsize,
            maxiter=maxiter
        )
        U, V, rmsd = get_rmsd(REMOVE_HS(mol), REMOVE_HS(opt_mol))
        align_mol(opt_mol, U, V)
        return rmsd, opt_mol
    except Exception as e:
        print(f"Torsion angle optimization failed: {e}")
        return None, pred_mol
    
# Assemble coordinates
def get_matched_coords(bb_mol_list):
    matched_coords = []
    for bb_mol in bb_mol_list:
        bb_coords = torch.tensor(bb_mol.GetConformer().GetPositions()).float()
        matched_coords.append(bb_coords)
    return torch.cat(matched_coords)

def mof_matching(feats, match_metal=True, match_organic=True, metal_bb_library=None, steps=200, popsize=15, maxiter=15):
    bb_mols = [Chem.MolFromMolBlock(mol_str, removeHs=False) for mol_str in feats['bb_mols']]

    matched_mols = []
    for bb_mol in bb_mols:
        if mu.is_metal_bb(bb_mol):
            if match_metal:
                smi = Chem.MolToSmiles(bb_mol, canonical=True)
                avg_mol = copy.deepcopy(metal_bb_library[smi])
                _, trans_mat, atom_map = AllChem.GetBestAlignmentTransform(prbMol=avg_mol, refMol=bb_mol)
                
                # Apply transformation
                coords = avg_mol.GetConformer().GetPositions()
                coords = np.append(coords, np.ones((coords.shape[0], 1)), axis=1)
                coords = coords.dot(trans_mat.T)[:, :3]
                avg_mol.GetConformer().SetPositions(coords)
                
                # Assign new atom orders
                atom_map = sorted(atom_map, key=lambda x: x[1])
                order = [prb_idx for prb_idx, ref_idx in atom_map]
                avg_mol = Chem.RenumberAtoms(avg_mol, order)

                matched_mols.append(avg_mol)
            else:
                matched_mols.append(bb_mol)
        else:
            if match_organic:
                rmsd, matched_mol = conformer_matching_single(bb_mol, steps=steps, popsize=popsize, maxiter=maxiter)
                matched_mols.append(matched_mol)
            else:
                matched_mols.append(bb_mol)
    return matched_mols