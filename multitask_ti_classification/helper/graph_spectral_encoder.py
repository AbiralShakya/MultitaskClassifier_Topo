# graph_spectral_encoder.py
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse.linalg

class GraphSpectralEncoder(nn.Module):
    """
    Given a PyG edge_index and num_nodes (and optionally batch),
    builds the normalized graph Laplacian, extracts the first K_LAPLACIAN_EIGS
    nonzero eigenvalues, and embeds them via an MLP.
    """
    def __init__(self, k_eigs: int, hidden: int):
        super().__init__()
        self.k = k_eigs
        self.mlp = nn.Sequential(
            nn.Linear(k_eigs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, edge_index, num_nodes, batch=None):
        # Build normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
        row, col, val = get_laplacian(edge_index,
                                      normalization='sym',
                                      num_nodes=num_nodes)
        # Convert to SciPy CSC for eigen-decomp
        L = to_scipy_sparse_matrix((val, (row, col)),
                                   num_nodes, num_nodes).tocsc()

        # Compute k+1 smallest eigenvalues (smallest is ~0)
        eigs, _ = scipy.sparse.linalg.eigsh(L,
                                            k=self.k+1,
                                            sigma=0.0,
                                            which='LM')
        # Drop the first zero λ and keep the next k
        spec = torch.from_numpy(eigs[1:self.k+1]).float().to(edge_index.device)

        # If batch >1 graph, do this per-graph (batch.to_data_list()), but
        # for now assume one graph per forward.
        return self.mlp(spec.unsqueeze(0))  # → [1, hidden]
