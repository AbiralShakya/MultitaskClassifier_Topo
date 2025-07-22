import torch.nn as nn

class PHTokenEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dims = 128, output_dim=256):
        super().__init__()
        # Optionally, a small MLP if you want to transform the token
        # self.mlp = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, data):
        # data.x will be (B, 1, 256) due to HeteroData and batching
        return data.x.squeeze(1) # (B, 256)