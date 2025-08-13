import torch
from torch_geometric.utils import scatter


# Global map from chain characters to integers.
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


def to_numpy(x):
    return x.detach().cpu().numpy()

def repeat_interleave(repeats):
    """
    Args:
    - repeats (torch.Tensor): tensor of integers, each representing the number of times to repeat the corresponding index
    - torch.Tensor: tensor of indices with the specified repeats
    """
    outs = torch.repeat_interleave(
        input=torch.arange(len(repeats), device=repeats.device),
        repeats=repeats
    )
    return outs

def lattice_to_nm_radians(lattice):
    lattice = lattice.clone()
    lattice[:, :3] *= ANG_TO_NM_SCALE
    lattice[:, 3:] *= torch.pi / 180.0
    return lattice

def lattice_to_ang_degrees(lattice):
    lattice = lattice.clone()
    lattice[:, :3] *= NM_TO_ANG_SCALE
    lattice[:, 3:] *= 180.0 / torch.pi
    return lattice

def apply_rototranslation(X_atoms, rotmats, trans, bb_num_vec):
    """
    Apply block-wise rototranslation to atom coordinates
    Args:
    - X_atoms (torch.Tensor): tensor of atom coordinates [num_atoms, 3]
    - trans (torch.Tensor): tensor of translations [num_bbs, 3]
    - rotmats (torch.Tensor): tensor of rotations [num_bbs, 3, 3]
    - bb_num_vec (torch.Tensor): tensor of number of atoms in each building block [num_bbs]
    """    
    # Center building block coordinates
    bb_ids = repeat_interleave(bb_num_vec) # [num_atoms]
    bb_centroids = scatter(X_atoms, bb_ids, dim=0, reduce='mean') # [num_bbs, 3]
    X_centered = X_atoms - bb_centroids[bb_ids] # [num_atoms, 3]
    
    # Gather corresponding R, t for each atom
    R_atoms = rotmats[bb_ids] # [num_atoms, 3, 3]
    t_atoms = trans[bb_ids] # [num_atoms, 3]
    
    # Apply rototranslation
    X_final = torch.bmm(R_atoms, X_centered.unsqueeze(-1)).squeeze(-1) # [num_atoms, 3]
    X_final = X_final + t_atoms
    
    return X_final

def kabsch_match(q_coord: torch.Tensor, p_coord: torch.Tensor):
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
        p_coord (torch.Tensor): [n_atoms, 3] tensor of coordinates for molecule P
        q_coord (torch.Tensor): [n_atoms, 3] tensor of coordinates for molecule Q (target)

    Returns:
        U: Rotation matrix (D,D)
        V: Translation vector (D)
        RMSD : Root mean squared deviation between P and Q
    """
    def kabsch(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        C = torch.matmul(P.T, Q)
        U, S, Vt = torch.linalg.svd(C)
        det = torch.det(torch.matmul(U, Vt))
        D = torch.diag(P.new_tensor([1.0, 1.0, det]))
        R = torch.matmul(torch.matmul(U, D), Vt)
        return R

    if q_coord.shape[0] != p_coord.shape[0]:
        raise ValueError("Number of atoms in P and Q must be the same.")
    
    p_trans, q_trans = p_coord.mean(dim=0), q_coord.mean(dim=0)
    p_centroid, q_centroid = p_coord - p_trans, q_coord - q_trans
    
    U = kabsch(p_centroid, q_centroid)
    
    p_prime_centroid = torch.matmul(p_centroid, U)
    rmsd = torch.sqrt(torch.mean(torch.square(p_prime_centroid - q_centroid)))
    
    V = q_trans - torch.matmul(p_trans, U)
    
    return U, V, rmsd

def pairwise_rmsd(position_1: torch.Tensor, positions_2: torch.Tensor):
    """
    Args:
    - position_1 (torch.Tensor): (N, 3) tensor of positions
    - positions_2 (torch.Tensor): (M, N, 3) tensor of positions
    Returns:
    - rmsd (torch.Tensor): (M,) tensor of rmsd values
    """
    diff = position_1.unsqueeze(0) - positions_2 # (M, N, 3)
    rmsds = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1), dim=-1)) # (M,)
    return rmsds