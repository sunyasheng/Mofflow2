# Output modules
import math
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_cluster import radius
from torch_geometric.utils import scatter, softmax
from utils import data as du
from utils.model import GaussianSmearing, custom_xavier_init


# Rotation
class AtomfeatsToRotmat(nn.Module):
    def __init__(self, rotation_cfg):
        super().__init__()
        self.use_svd = rotation_cfg.use_svd
        self.rot_output = nn.Sequential(
            nn.Linear(rotation_cfg.node_embed_dim, rotation_cfg.node_embed_dim),
            nn.GELU(),
            nn.Linear(rotation_cfg.node_embed_dim, 9)
        )
        
        self.init_weights()
        
    def init_weights(self):
        # Initialize weights
        custom_xavier_init(self.rot_output[0].weight, mode='default')
        custom_xavier_init(self.rot_output[2].weight, mode='default')
        # Initialize biases
        if self.rot_output[0].bias is not None:
            nn.init.zeros_(self.rot_output[0].bias)
        if self.rot_output[2].bias is not None:
            nn.init.zeros_(self.rot_output[2].bias)
    
    def forward(self, bb_feats):
        raw_rotmats = self.rot_output(bb_feats).view(-1, 3, 3)

        if self.use_svd:
            return roma.special_procrustes(raw_rotmats)
        return raw_rotmats

# Translation
class AtomfeatsToTrans(nn.Module):
    def __init__(self, node_embed_dim):
        super().__init__()
        self.trans_output = nn.Sequential(
            nn.Linear(node_embed_dim, node_embed_dim),
            nn.GELU(),
            nn.Linear(node_embed_dim, 3)
        )

    def forward(self, bb_feats, batch):
        # Convert to translation
        trans = self.trans_output(bb_feats) # [num_bbs, 3]

        # Remove mean
        trans_mean = scatter(trans, batch.batch, dim=0, reduce='mean') # [batch_size, 3]
        trans_mean = trans_mean.repeat_interleave(batch.num_bbs, dim=0)
        trans = trans - trans_mean
        
        return trans

# Lattice    
class AtomfeatsToLattice(nn.Module):
    def __init__(self, node_embed_dim):
        super().__init__()
        self.lattice_output = nn.Sequential(
            nn.Linear(node_embed_dim, node_embed_dim),
            nn.GELU(),
            nn.Linear(node_embed_dim, 6)
        )

    def forward(self, bb_feats, batch):
        batch_feats = scatter(bb_feats, batch.batch, dim=0, reduce='mean') # [batch_size, node_embed_dim]

        # Convert to lattice
        lattice = F.softplus(self.lattice_output(batch_feats))

        return lattice
    
# Torsion angles
class AtomfeatsToTorsion(nn.Module):
    def __init__(self, torsion_cfg):
        super().__init__()
        self.torsion_cfg = torsion_cfg
        self.edge_dist_embedder = GaussianSmearing(0.0, torsion_cfg.gaussian.max_radius, torsion_cfg.gaussian.dist_embed_dim)
        
        self.no_heads = torsion_cfg.no_heads
        self.embed_dim = torsion_cfg.node_embed_dim
        self.head_dim = self.embed_dim // self.no_heads
        assert (
            self.head_dim * self.no_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        hc = self.no_heads * self.head_dim

        self.dihedral_proj = nn.Linear(4 * self.embed_dim, self.embed_dim)
        self.linear_q = nn.Linear(self.embed_dim, hc)
        self.linear_k = nn.Linear(self.embed_dim + torsion_cfg.gaussian.dist_embed_dim, hc)
        self.linear_v = nn.Linear(self.embed_dim + torsion_cfg.gaussian.dist_embed_dim, hc)
        self.linear_out = nn.Linear(hc, self.embed_dim)
        self.torsion_output = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 2)
        )

        self.init_weights()

    def init_weights(self):
        # Initialize weights
        custom_xavier_init(self.linear_q.weight, mode='1in5out')
        custom_xavier_init(self.linear_k.weight, mode='1in5out')
        custom_xavier_init(self.linear_v.weight, mode='1in5out')
        custom_xavier_init(self.linear_out.weight, mode='1in5out')
        custom_xavier_init(self.torsion_output[0].weight, mode='default')
        custom_xavier_init(self.torsion_output[2].weight, mode='default')
        # Initialize biases
        if self.linear_q.bias is not None:
            nn.init.zeros_(self.linear_q.bias)
        if self.linear_k.bias is not None:
            nn.init.zeros_(self.linear_k.bias)
        if self.linear_v.bias is not None:
            nn.init.zeros_(self.linear_v.bias)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)
        if self.torsion_output[0].bias is not None:
            nn.init.zeros_(self.torsion_output[0].bias)
        if self.torsion_output[2].bias is not None:
            nn.init.zeros_(self.torsion_output[2].bias)

    def forward(self, atom_feats, batch):
        # Get rotable bond index
        rotable_bond_index = batch.bond_index[:, batch.rotable_bond_mask] # [2, num_rotable_bonds]
        rotable_bond_pos = (batch.coords_t[rotable_bond_index[0]] + batch.coords_t[rotable_bond_index[1]]) / 2 # [num_rotable_bonds, 3]
        
        # Get batch index of rotatable bond
        atom_vec = du.repeat_interleave(batch.num_atoms) # [num_atoms]
        rotable_bond_vec = atom_vec[rotable_bond_index[0]] # [num_rotable_bonds]

        # Construct edges for rotable bonds
        edge_index = radius(
            x=batch.coords_t,
            y=rotable_bond_pos,
            r=self.torsion_cfg.max_radius,
            batch_x=atom_vec,
            batch_y=rotable_bond_vec,
            max_num_neighbors=self.torsion_cfg.max_neighbors
        )
        tgt, src = edge_index

        # Distance embedding
        edge_vec = batch.coords_t[src] - rotable_bond_pos[tgt] # [num_edges, 3]
        edge_dist_embed = self.edge_dist_embedder(edge_vec.norm(dim=-1)) # [num_edges, dist_embed_dim]

        # Rotable bond features
        canonical_torsion_tuples = batch.canonical_torsion_tuples.T # [4, num_rotable_bonds]

        feats_i = atom_feats[canonical_torsion_tuples[0]] # [num_rotable_bonds, node_embed_dim]
        feats_j = atom_feats[canonical_torsion_tuples[1]] # [num_rotable_bonds, node_embed_dim]
        feats_k = atom_feats[canonical_torsion_tuples[2]] # [num_rotable_bonds, node_embed_dim]
        feats_l = atom_feats[canonical_torsion_tuples[3]] # [num_rotable_bonds, node_embed_dim]

        dihedral_feats_forward = torch.cat([feats_i, feats_j, feats_k, feats_l], dim=-1)
        dihedral_feats_backward = torch.cat([feats_l, feats_k, feats_j, feats_i], dim=-1)

        rotable_bond_feats = (
            self.dihedral_proj(dihedral_feats_forward) +
            self.dihedral_proj(dihedral_feats_backward)
        )
                
        # Query = rotable bond features
        q = self.linear_q(rotable_bond_feats) # [num_rotable_bonds, hc]
        q = rearrange(q, 'n (h c) -> n h c', h=self.no_heads) # [num_rotable_bonds, no_heads, head_dim]

        # Key/Value = neighboring atom features & distances
        atom_input = torch.cat([atom_feats[src], edge_dist_embed], dim=-1) # [num_edges, node_embed_dim + dist_embed_dim]
        k = self.linear_k(atom_input) # [num_edges, hc]
        v = self.linear_v(atom_input) # [num_edges, hc]
        k = rearrange(k, 'n (h c) -> n h c', h=self.no_heads) # [num_edges, no_heads, head_dim]
        v = rearrange(v, 'n (h c) -> n h c', h=self.no_heads) # [num_edges, no_heads, head_dim]

        # Compute attention
        attn = torch.einsum('ehc, ehc -> eh', q[tgt], k) / math.sqrt(self.head_dim) # [num_edges, no_heads]
        attn = softmax(attn, tgt, dim=0) # [num_edges, no_heads]

        # Compute message
        msg = attn[..., None] * v # [num_edges, no_heads, head_dim]
        output = scatter(msg, tgt, dim=0, reduce='sum') # [num_rotable_bonds, no_heads, head_dim]
        output = rearrange(output, 'n h c -> n (h c)')
        output = self.linear_out(output) # [num_rotable_bonds, embed_dim]
        torsions = self.torsion_output(output) # [num_rotable_bonds, 2]

        return torsions
        