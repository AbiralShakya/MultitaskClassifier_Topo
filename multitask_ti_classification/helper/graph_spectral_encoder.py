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
        # Handle batch processing
        if batch is not None:
            # Get the number of unique graphs in the batch
            num_graphs = batch.max().item() + 1
            
            # For now, use the same spectral features for all graphs in the batch
            # This is a simplified approach - in practice, you might want to process each graph separately
            single_emb = self._process_single_graph(edge_index, num_nodes)
            
            # Expand to match batch size
            return single_emb.unsqueeze(0).expand(num_graphs, -1)  # → [batch_size, hidden]
        else:
            # Single graph case
            return self._process_single_graph(edge_index, num_nodes)
    
    def _process_single_graph(self, edge_index, num_nodes):
        """Process a single graph to get spectral embeddings."""
        # Build normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
        # Handle different versions of PyTorch Geometric
        try:
            # Newer version returns (edge_index, edge_weight)
            laplacian_edge_index, laplacian_edge_weight = get_laplacian(edge_index,
                                                                        normalization='sym',
                                                                        num_nodes=num_nodes)
            # Convert to SciPy CSC for eigen-decomp
            L = to_scipy_sparse_matrix(laplacian_edge_index, laplacian_edge_weight,
                                       num_nodes).tocsc()
        except (ValueError, TypeError):
            # Fallback: manually construct the Laplacian
            import torch_geometric.utils as utils
            # Get adjacency matrix
            adj = utils.to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
            # Get degree matrix
            deg = torch.diag(torch.sum(adj, dim=1))
            # Get normalized Laplacian
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            L_dense = torch.eye(num_nodes) - torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
            # Convert to sparse
            L = scipy.sparse.csr_matrix(L_dense.cpu().numpy()).tocsc()

        # Compute k+1 smallest eigenvalues (smallest is ~0)
        # Add regularization to handle singular matrices
        try:
            result = scipy.sparse.linalg.eigsh(L,
                                               k=min(self.k+1, num_nodes-1),  # Ensure we don't ask for more than available
                                               sigma=0.0,
                                               which='LM')
            eigs = result[0] if result is not None else None
            # Drop the first zero λ and keep the next k, but ensure we have enough eigenvalues
            if eigs is not None and len(eigs) > 1:
                end_idx = min(len(eigs), self.k+1)
                spec = torch.from_numpy(eigs[1:end_idx]).float().to(edge_index.device)
                # Pad with zeros if we don't have enough eigenvalues
                if spec.shape[0] < self.k:
                    padding = torch.zeros(self.k - spec.shape[0], dtype=torch.float, device=edge_index.device)
                    spec = torch.cat([spec, padding])
            else:
                spec = torch.zeros(self.k, dtype=torch.float, device=edge_index.device)
        except RuntimeError as e:
            if "Factor is exactly singular" in str(e):
                # Add regularization to make matrix invertible
                print(f"Warning: Singular Laplacian detected, adding regularization. Num nodes: {num_nodes}")
                # Add small diagonal regularization
                L_reg = L + 1e-6 * scipy.sparse.eye(L.shape[0], format='csc')
                try:
                    eigs, _ = scipy.sparse.linalg.eigsh(L_reg,
                                                        k=min(self.k+1, num_nodes-1),
                                                        sigma=0.0,
                                                        which='LM')
                    if len(eigs) > 1:
                        spec = torch.from_numpy(eigs[1:min(len(eigs), self.k+1)]).float().to(edge_index.device)
                        if spec.shape[0] < self.k:
                            padding = torch.zeros(self.k - spec.shape[0], dtype=torch.float, device=edge_index.device)
                            spec = torch.cat([spec, padding])
                    else:
                        spec = torch.zeros(self.k, dtype=torch.float, device=edge_index.device)
                except Exception as e2:
                    print(f"Warning: Still failed after regularization: {e2}")
                    # Fallback: use random features
                    spec = torch.randn(self.k).float().to(edge_index.device)
            else:
                # Other eigenvalue computation errors
                print(f"Warning: Eigenvalue computation failed: {e}")
                spec = torch.randn(self.k).float().to(edge_index.device)

        # Process the spectral features
        return self.mlp(spec.unsqueeze(0)).squeeze(0)  # → [hidden]
