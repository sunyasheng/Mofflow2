import math
import torch
import torch.nn as nn
from einops import rearrange
from torch_geometric.utils import softmax, scatter
from utils.model import custom_xavier_init, GaussianSmearing


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x / math.sqrt(self.dim)
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_cfg = cfg.attention
        self.init_mode = cfg.init_mode

        # Check if node_embed_dim is divisible by num_heads
        self.node_embed_dim = cfg.attention.node_embed_dim
        self.edge_embed_dim = 4 + cfg.attention.dist_embed_dim # 4 for bond features
        self.no_heads = cfg.attention.no_heads
        self.head_dim = self.node_embed_dim // self.no_heads
        assert (
            self.head_dim * self.no_heads == self.node_embed_dim
        ), "node_embed_dim must be divisible by num_heads"
        
        hc = self.no_heads * self.head_dim
        self.linear_q = nn.Linear(self.node_embed_dim, hc)
        self.linear_k = nn.Linear(self.node_embed_dim + self.edge_embed_dim, hc)
        self.linear_v = nn.Linear(self.node_embed_dim + self.edge_embed_dim, hc)
        self.linear_out = nn.Linear(hc, self.node_embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        
        self.init_weights()
    
    def init_weights(self):
        # Initialize weights
        custom_xavier_init(self.linear_q.weight, mode=self.init_mode)
        custom_xavier_init(self.linear_k.weight, mode=self.init_mode)
        custom_xavier_init(self.linear_v.weight, mode=self.init_mode)
        custom_xavier_init(self.linear_out.weight, mode=self.init_mode)
        
        # Initialize biases
        if self.linear_q.bias is not None:
            nn.init.zeros_(self.linear_q.bias)
        if self.linear_k.bias is not None:
            nn.init.zeros_(self.linear_k.bias)
        if self.linear_v.bias is not None:
            nn.init.zeros_(self.linear_v.bias)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)
    
    def forward(self, node_feats, edge_feats, edge_index):        
        src, tgt = edge_index

        # Query = node features
        q = self.linear_q(node_feats[tgt]) # [num_edges, hc]
        q = rearrange(q, 'n (h c) -> n h c', h=self.no_heads) # [num_edges, h, c]

        # Key/Value = node features + edge features
        kv_input = torch.cat([node_feats[src], edge_feats], dim=-1) # [num_edges, node_embed_dim + edge_embed_dim]
        k = self.linear_k(kv_input) # [num_edges, hc]
        v = self.linear_v(kv_input) # [num_edges, hc]
        k = rearrange(k, 'n (h c) -> n h c', h=self.no_heads) # [num_edges, h, c]
        v = rearrange(v, 'n (h c) -> n h c', h=self.no_heads) # [num_edges, h, c]
        
        # Compute attention
        a = torch.einsum('ehc, ehc -> eh', q, k)
        a = a / math.sqrt(self.head_dim)
        a = softmax(a, tgt, dim=0) # [num_edges, h]
        
        # Compute value
        o_msg = a[..., None] * v # [num_edges, h, c]
        output = scatter(o_msg, tgt, dim=0, reduce='sum') # [num_nodes, h, c]
        output = rearrange(output, 'n h c -> n (h c)') # [num_nodes, hc]
        output = self.linear_out(output) # [num_nodes, node_embed_dim]
        output = self.dropout(output)
        
        return output
    
class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln_mode = cfg.ln_mode
        self.init_mode = cfg.init_mode
        self.node_embed_dim = cfg.attention.node_embed_dim
        
        # Normalization layers
        if cfg.norm == 'layernorm':
            self.attn_norm = nn.LayerNorm(self.node_embed_dim)
            self.ffn_norm = nn.LayerNorm(self.node_embed_dim)
        elif cfg.norm == 'rmsnorm':
            self.attn_norm = RMSNorm(self.node_embed_dim)
            self.ffn_norm = RMSNorm(self.node_embed_dim)
        else:
            raise ValueError(f"Unknown normalization type: {cfg.norm}")
        
        # Attention and Feed Forward Network
        self.self_attn = MultiHeadAttention(cfg)
        self.ffn = nn.Sequential(
            nn.Linear(self.node_embed_dim, cfg.encoder.ffn_dim, bias=cfg.ffn_bias),
            nn.GELU(),
            nn.Linear(cfg.encoder.ffn_dim, self.node_embed_dim, bias=cfg.ffn_bias),
            nn.Dropout(cfg.dropout)
        )
        
        self.init_weights()
        self.norm_inputs = []
        
    def init_weights(self):
        # Initialize weights
        custom_xavier_init(self.ffn[0].weight, mode=self.init_mode)
        custom_xavier_init(self.ffn[2].weight, mode=self.init_mode)
        
        # Initialize biases
        if self.ffn[0].bias is not None:
            nn.init.zeros_(self.ffn[0].bias)
        if self.ffn[2].bias is not None:
            nn.init.zeros_(self.ffn[2].bias)
    
    def forward(self, node_feats, edge_feats, edge_index):
        ###### Pre-LN ######
        if self.ln_mode == 'pre':
            residual = node_feats
            node_feats = self.attn_norm(node_feats)
            node_feats = self.self_attn(node_feats, edge_feats, edge_index)
            node_feats = residual + node_feats
            
            residual = node_feats
            pre_ffn_norm = node_feats.detach() # for logging
            node_feats = self.ffn_norm(node_feats)
            node_feats = self.ffn(node_feats)
            node_feats = residual + node_feats
            
            # Store norms for logging
            if hasattr(self, 'norm_inputs'):
                self.norm_inputs.append(pre_ffn_norm.norm(p=2, dim=-1).mean().item())
        ###### Post-LN ######
        elif self.ln_mode == 'post':
            residual = node_feats
            node_feats = self.self_attn(node_feats, edge_feats, edge_index)
            node_feats = node_feats + residual
            node_feats = self.attn_norm(node_feats)

            residual = node_feats
            node_feats = self.ffn(node_feats)
            node_feats = node_feats + residual
            pre_ffn_norm = node_feats.detach() # for logging
            node_feats = self.ffn_norm(node_feats)

            # Store norms for logging
            if hasattr(self, 'norm_inputs'):
                self.norm_inputs.append(pre_ffn_norm.norm(p=2, dim=-1).mean().item())
        else:
            raise ValueError(f"Unknown layer normalization mode: {self.ln_mode}")
        
        return node_feats

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(cfg)
            for _ in range(cfg.num_layers)
        ])
    
    def forward(self, init_node_feats, edge_feats, edge_index):
        node_feats = init_node_feats
        for layer in self.transformer_layers:
            node_feats = layer(node_feats, edge_feats, edge_index)
        
        return node_feats