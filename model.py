import torch.nn as nn
from multitask_ti_classification.encoders.egnn_encoder import RealSpaceEGNN
from multitask_ti_classification.encoders.kspace_transformer_encoder import KSpaceCTGNN
from multitask_ti_classification.encoders.ph_token_encoder import PHTokenEncoder
from multitask_ti_classification.fusion.fusion_head import FusionHead

class TopologicalMaterialModel(nn.Module):
    def __init__(self, 
                 real_space_node_dim, # e.g., original atom features + ct_uae_dim
                 k_space_node_dim,    # e.g., irrep_id (one-hot/embedding) + energy_rank + pos_enc
                 ph_dim=256,
                 real_space_hidden_dim=512,
                 k_space_hidden_dim=512,
                 fusion_attn_proj_dim=512,
                 mlp_dims=[512, 256, 64, 1] # MLP after fusion (768 -> 512 -> 256 -> 64 -> 1)
                ):
        super().__init__()
        
        self.real_space_encoder = RealSpaceEGNN(
            input_node_dim=real_space_node_dim,
            hidden_dim=real_space_hidden_dim
        )
        self.k_space_encoder = KSpaceCTGNN(
            input_node_dim=k_space_node_dim,
            hidden_dim=k_space_hidden_dim
        )
        self.ph_encoder = PHTokenEncoder(input_dim=ph_dim)
        
        self.fusion_head = FusionHead(
            d_real=real_space_hidden_dim,
            d_k=k_space_hidden_dim,
            d_ph=ph_dim,
            d_attn_proj=fusion_attn_proj_dim,
            classifier_dims=mlp_dims
        )

    def forward(self, data):
        # Extract sub-graphs for each encoder
        # PyG automatically handles batching within HeteroData for each node type
        h_real = self.real_space_encoder(data['atom'])
        h_kspace = self.k_space_encoder(data['kpoint'])
        h_ph = self.ph_encoder(data['ph'])
        
        # Fuse and classify
        logits = self.fusion_head(h_real, h_kspace, h_ph)
        
        return logits # Raw logits, apply sigmoid in loss or after for prediction