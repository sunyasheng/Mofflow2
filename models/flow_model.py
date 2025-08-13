import torch
import torch.nn as nn
from models.initialize_modules import InitializeAtomEmbedding
from models.interaction_modules import InteractionModule, AttentionPoolBuildingBlock
from models.output_modules import AtomfeatsToRotmat, AtomfeatsToTrans, AtomfeatsToLattice, AtomfeatsToTorsion


class FlowModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.initial_embed = InitializeAtomEmbedding(model_cfg.initialize_cfg)
        self.interaction_layers = nn.ModuleList([
            InteractionModule(model_cfg.interaction_cfg)
            for _ in range(model_cfg.interaction_cfg.num_layers)
        ])
        self.final_pool = AttentionPoolBuildingBlock(model_cfg.pool_cfg)

        # Output modules
        if model_cfg.corrupt_rots:
            self.rotation_output = AtomfeatsToRotmat(rotation_cfg=model_cfg.output_model_cfg.rotation)
        if model_cfg.corrupt_trans:
            self.translation_output = AtomfeatsToTrans(node_embed_dim=model_cfg.node_embed_dim)
        if model_cfg.corrupt_lattice:
            self.lattice_output = AtomfeatsToLattice(node_embed_dim=model_cfg.node_embed_dim)
        if model_cfg.corrupt_torsions:
            self.torsion_output = AtomfeatsToTorsion(torsion_cfg=model_cfg.output_model_cfg.torsion)

    def forward(self, batch):
        # Initialize atom embeddings
        atom_feats = self.initial_embed(batch)  # [N, D]

        # Interaction layers
        for layer in self.interaction_layers:
            atom_feats = layer(atom_feats, batch)

        # Final pooling
        bb_feats = self.final_pool(atom_feats, batch)

        # Output modules
        outputs = {}
        if self.model_cfg.corrupt_rots:
            outputs['pred_rotmats'] = self.rotation_output(bb_feats)
        if self.model_cfg.corrupt_trans:
            outputs['pred_trans'] = self.translation_output(bb_feats, batch)
        if self.model_cfg.corrupt_lattice:
            outputs['pred_lattice'] = self.lattice_output(bb_feats, batch)
        if self.model_cfg.corrupt_torsions:
            outputs['pred_torsion'] = self.torsion_output(atom_feats, batch)
        
        return outputs