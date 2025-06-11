import torch
import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self, d_real=512, d_k=512, d_ph=256, d_attn_proj=512, d_fused=768, classifier_dims=[512, 256, 64, 1]):
        super().__init__()
        # Projection layers for cross-modal attention
        self.query_proj = nn.Linear(d_real, d_attn_proj)
        self.key_proj   = nn.Linear(d_k, d_attn_proj)
        self.value_proj = nn.Linear(d_k, d_attn_proj)

        # Output projection for the cross-attention output
        self.attn_output_proj = nn.Linear(d_attn_proj, d_attn_proj) # Project to 512, as per diagram
        
        # MLP for post-attention fusion and final classification
        # Combine the cross-modal attention output and the ASPH token
        self.pre_classifier_mlp = nn.Sequential(
            nn.Linear(d_attn_proj + d_ph, classifier_dims[0]), # (512 + 256 = 768) -> 512
            nn.ReLU(),
            nn.Dropout(0.2), # Good practice for regularization
            nn.Linear(classifier_dims[0], classifier_dims[1]), # 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dims[1], classifier_dims[2]), # 256 -> 64
            nn.ReLU(),
            nn.Linear(classifier_dims[2], classifier_dims[3])  # 64 -> 1 (logits for binary classification)
        )

    def forward(self, h_real, h_kspace, h_ph):
        # Ensure dimensions are compatible for batching: (B, D)
        # Cross-modal attention (simple dot product attention as described)
        # Query from real-space, Key/Value from k-space
        query = self.query_proj(h_real) # (B, d_attn_proj)
        key   = self.key_proj(h_kspace)  # (B, d_attn_proj)
        value = self.value_proj(h_kspace) # (B, d_attn_proj)

        # Compute attention scores: (B, 1) after sum, then softmax over features (not batch)
        # Using element-wise product and sum for attention, then softmax.
        # This is a specific interpretation of "dot-product attention" given the (B, D) inputs.
        # A more standard attention would involve expanding dims and matmul.
        # Let's adjust to a more typical attention for global tokens, or keep it simple as described:
        
        # Simple attention: (Q * K) -> scalar weight per sample, then apply to Value
        attn_scores = (query * key).sum(dim=-1, keepdim=True) # (B, 1)
        # Normalize scores across the batch or within a single sample if multiple targets.
        # Since we have one pair (h_real, h_kspace) per sample, softmax is tricky here.
        # If it's "cross-modal" on features, it's (B, D) * (B, D) -> (B, D_out)
        # The prompt implies a fusion mechanism, not token-level attention.
        
        # Let's assume a simplified fusion where real and k-space contribute to a combined feature.
        # A simple concatenation and MLP could also be "cross-modal fusion".
        # Based on "attn = torch.softmax((self.query(h_real) * self.key(h_k)).sum(-1, keepdim=True), dim=1)"
        # This implies attention over *features* within a sample.
        # If h_real and h_k are (B, D_encoder), and we want an attention mechanism
        # *between* these two views for each sample:
        
        # Let's re-interpret the attention mechanism for clarity:
        # Instead of self-attention within sequences, it's about aligning two global embeddings.
        # A common way to do this is a simple MLP + sigmoid or a gating mechanism.
        # The provided example implies a scaled dot-product for two single "tokens".
        # For h_real (B, D_real) and h_k (B, D_k), we need a (B, D_attn_proj) output.

        # Simplified approach for aligning two single global embeddings:
        # Think of it as a linear projection of h_real and h_k, then some interaction.
        
        # Option 1: Concatenation and MLP (MultiMat style is more complex, involving alignment losses)
        # fused_hk = torch.cat([self.query_proj(h_real), self.value_proj(h_kspace)], dim=-1)
        # h_ck = self.attn_output_proj(fused_hk) # Project to 512

        # Option 2: Learned weighting/gating (similar to what was hinted)
        # Compute a "gate" vector from h_real and h_kspace
        gate = torch.sigmoid(self.query_proj(h_real) + self.key_proj(h_kspace)) # (B, d_attn_proj)
        # Apply gate to k-space value
        h_ck = gate * self.value_proj(h_kspace) # (B, d_attn_proj)

        # Option 3: Direct from the provided pseudocode:
        # attn = torch.softmax((self.query(h_real) * self.key(h_k)).sum(-1, keepdim=True), dim=1)
        # h_ck = (attn * self.value(h_k)).sum(1)
        # This assumes h_real and h_k can be treated as sequences for attention.
        # Given they are (B, D), `dim=1` for softmax is problematic if D is 512.
        # It's likely intended to be:
        attn_weights = torch.sigmoid(torch.sum(self.query_proj(h_real) * self.key_proj(h_kspace), dim=-1, keepdim=True)) # (B, 1)
        h_ck = attn_weights * self.value_proj(h_kspace) # (B, d_attn_proj)
        # This means h_ck is a scaled version of the k-space value.

        # Fuse cross-modal attention output with ASPH token
        fused = torch.cat([h_ck, h_ph], dim=-1) # (B, d_attn_proj + d_ph)

        # Final MLP classifier
        h_fused = self.pre_classifier_mlp(fused)
        logits = self.classifier(h_fused).squeeze(-1) # (B) for BCEWithLogitsLoss
        return logits