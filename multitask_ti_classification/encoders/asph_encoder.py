import torch
import torch.nn as nn

class ASPHEncoder(nn.Module):
    def __init__(self, input_dim=3115, hidden_dims = 1024, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.fc(x) 