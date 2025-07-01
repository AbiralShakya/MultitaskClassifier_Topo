import torch
import torch.nn as nn

class KSpacePhysicsEncoder(nn.Module):
    def __init__(self, input_dim, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.fc(x) 