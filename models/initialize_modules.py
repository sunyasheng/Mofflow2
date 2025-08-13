import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import data as du
from utils.model import custom_xavier_init


class InitializeAtomEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.atom_type_embed = nn.Embedding(cfg.max_atom_types, cfg.atom_type_dim)
        self.linear_out = nn.Linear(cfg.atom_type_dim + cfg.atom_feats_dim - 1 + 3 + 6 + cfg.time_embed_dim, cfg.node_embed_dim)
        
        self.init_weights()
    
    def init_weights(self):
        # Initialize weights
        custom_xavier_init(self.linear_out.weight, mode='default')
        # Initialize biases
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)  
    
    @staticmethod
    def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
        # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
        assert len(timesteps.shape) == 1
        timesteps = timesteps * max_positions
        half_dim = embedding_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb
    
    def embed_t(self, timesteps):
        timestep_embedding = self.get_time_embedding(
            timesteps[:, 0],
            self.cfg.time_embed_dim,
            max_positions=2056
        )
        return timestep_embedding
    
    @staticmethod
    def get_index_embedding(indices, emb_dim, max_len=2048):
        """Creates sine / cosine positional embeddings from a prespecified indices.

        Args:
            indices: offsets of size [..., num_tokens] of type integer
            emb_dim: dimension of the embeddings to create
            max_len: maximum length

        Returns:
            positional embedding of shape [..., num_tokens, emb_dim]
        """
        K = torch.arange(emb_dim // 2, device=indices.device)
        pos_embedding_sin = torch.sin(
            indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
        ).to(indices.device)
        pos_embedding_cos = torch.cos(
            indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
        ).to(indices.device)
        pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
        return pos_embedding

    def forward(self, batch):
        # Atomic features (A)
        atom_type_embed = self.atom_type_embed(batch['atom_types']) # [num_atoms, atom_type_dim]
        atom_feats = batch['atom_feats'][:, 1:] # [num_atoms, atom_feats_dim - 1]
        atom_feats = torch.cat([atom_type_embed, atom_feats], dim=-1)

        # Noisy coordinates (X_t)
        coords_t = batch['coords_t'] * du.ANG_TO_NM_SCALE

        # Noisy lattice (L_t)
        lattice_t = du.lattice_to_nm_radians(batch['lattice_t']) # [B, 6]
        lattice_bbs_t = torch.repeat_interleave(lattice_t, batch.num_bbs, dim=0) # [num_bbs, 6]
        lattice_atoms_t = torch.repeat_interleave(lattice_bbs_t, batch.bb_num_vec, dim=0) # [num_atoms, 6]

        # Time embedding (t)
        atom_time_emb = self.embed_t(batch['t_atoms'])

        # Concatenate A, X_t, L_t, t
        atom_feats = torch.cat([atom_feats, coords_t, lattice_atoms_t, atom_time_emb], dim=-1) # [num_atoms, atom_feats_dim + 3 + 6 + time_embed_dim]
        atom_feats = self.linear_out(atom_feats) # [num_atoms, node_embed_dim]

        # Positional encoding to atom_feats
        if self.cfg.add_pos_embed:
            bb_pos_embed = self.get_index_embedding(
                indices=batch.bb_pos_idx, 
                emb_dim=self.cfg.node_embed_dim
            ) # [num_atoms, node_embed_dim]

            atom_feats = atom_feats + bb_pos_embed

        return atom_feats