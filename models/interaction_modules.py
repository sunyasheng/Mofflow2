# Modules for interaction layers
import math
import torch
import torch.nn as nn
from einops import rearrange
from torch_cluster import radius_graph, radius
from torch_geometric.utils import scatter, softmax
from models.transformers import TransformerEncoder
from utils.data import repeat_interleave
from utils.model import GaussianSmearing, custom_xavier_init


# Message passing within building block
class InteractionModule(nn.Module):
    def __init__(self, interaction_cfg):
        super().__init__()
        self._inter_cfg = interaction_cfg
        self.edge_dist_embedder = GaussianSmearing(
            start=0.0,
            stop=self._inter_cfg.max_radius,
            num_gaussians=self._inter_cfg.dist_embed_dim
        )
        self.interaction_layer = TransformerEncoder(self._inter_cfg.trans_encoder_cfg)

    def forward(self, atom_feats, batch):
        ###### Construct edge index ######
        atom_vec = repeat_interleave(batch.num_atoms)
        radius_edges = radius_graph(
            x=batch.coords_t,
            r=self._inter_cfg.max_radius,
            batch=atom_vec,
            loop=True,
            max_num_neighbors=self._inter_cfg.max_neighbors
        ) # [2, E_r]
        edge_index = torch.cat([batch.bond_index, radius_edges], dim=-1) # [2, E_b + E_r]
        # TODO: Fully connected edges for metal atoms

        ###### Construct edge index ######
        # Bond features
        bond_feats = torch.cat([
            batch.bond_feats,
            torch.zeros(radius_edges.shape[-1], 4, device=batch.bond_feats.device)
        ], dim=0) # [E_b + E_r, 4]
        
        # Distance features
        src, tgt = edge_index
        edge_vec = batch.coords_t[tgt] - batch.coords_t[src] # [E_b + E_r, 3]
        edge_dist_embed = self.edge_dist_embedder(edge_vec.norm(dim=-1)) # [E_b + E_r, dist_embed_dim]

        # Concatenate bond and distance features
        edge_feats = torch.cat([bond_feats, edge_dist_embed], dim=-1) # [E_b + E_r, 4 + dist_embed_dim]
        
        atom_feats = self.interaction_layer(atom_feats, edge_feats, edge_index)
        
        return atom_feats
    
# Attention pooling to building block level
class AttentionPoolBuildingBlock(nn.Module):
    def __init__(self, pool_cfg):
        super().__init__()
        self.bb_pool_cfg = pool_cfg
        self.edge_dist_embedder = GaussianSmearing(0.0, pool_cfg.gaussian.max_radius, pool_cfg.gaussian.dist_embed_dim)

        self.no_heads = pool_cfg.no_heads
        self.embed_dim = pool_cfg.node_embed_dim
        self.head_dim = self.embed_dim // self.no_heads
        assert (
            self.head_dim * self.no_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        hc = self.no_heads * self.head_dim

        self.linear_q = nn.Linear(self.embed_dim, hc)
        self.linear_k = nn.Linear(self.embed_dim + pool_cfg.gaussian.dist_embed_dim, hc)
        self.linear_v = nn.Linear(self.embed_dim + pool_cfg.gaussian.dist_embed_dim, hc)
        self.linear_out = nn.Linear(hc, self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # Initialize weights
        custom_xavier_init(self.linear_q.weight, mode='1in5out')
        custom_xavier_init(self.linear_k.weight, mode='1in5out')
        custom_xavier_init(self.linear_v.weight, mode='1in5out')
        custom_xavier_init(self.linear_out.weight, mode='default')
        # Initialize biases
        if self.linear_q.bias is not None:
            nn.init.zeros_(self.linear_q.bias)
        if self.linear_k.bias is not None:
            nn.init.zeros_(self.linear_k.bias)
        if self.linear_v.bias is not None:
            nn.init.zeros_(self.linear_v.bias)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)

    def forward(self, atom_feats, batch):
        num_bbs = len(batch.bb_num_vec)
        bb_vec = repeat_interleave(batch.bb_num_vec)

        bb_center_pos = scatter(batch.coords_t, bb_vec, dim=0, reduce='mean') # [num_bbs, 3]
        bb_center_feats = scatter(atom_feats, bb_vec, dim=0, reduce='sum') # [num_bbs, node_embed_dim]

        # Construct edge index (within building block)
        edge_index = radius(
            x=batch.coords_t,
            y=bb_center_pos,
            r=self.bb_pool_cfg.max_radius,
            batch_x=bb_vec,
            batch_y=torch.arange(num_bbs, device=bb_vec.device),
            max_num_neighbors=self.bb_pool_cfg.max_neighbors,
        )
        tgt, src = edge_index

        # Distance embedding
        edge_vec = batch.coords_t[src] - bb_center_pos[tgt]
        edge_dist_embed = self.edge_dist_embedder(edge_vec.norm(dim=-1))

        # Query = bb center features
        q = self.linear_q(bb_center_feats)
        q = rearrange(q, 'n (h c) -> n h c', h=self.no_heads) # [num_bbs, no_heads, head_dim]

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
        output = scatter(msg, tgt, dim=0, reduce='sum') # [num_bbs, no_heads, head_dim]
        output = rearrange(output, 'n h c -> n (h c)') # [num_bbs, embed_dim]
        output = self.linear_out(output) # [num_bbs, embed_dim]

        return output