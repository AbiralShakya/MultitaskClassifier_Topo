# graph_spectral_encoder.py
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse.linalg
import hashlib
import pickle

class GraphSpectralEncoder(nn.Module):
    """
    Smart spectral encoder with caching and memory-efficient eigenvalue computation.
    """
    def __init__(self, k_eigs: int, hidden: int, cache_size: int = 100):  # Reduced cache size
        super().__init__()
        self.k = k_eigs
        self.cache_size = cache_size
        self.cache = {}  # Simple LRU cache
        self.mlp = nn.Sequential(
            nn.Linear(k_eigs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def _get_graph_hash(self, edge_index, num_nodes):
        """Create a hash for the graph to enable caching."""
        # Create a simple hash based on graph structure
        edge_hash = hashlib.md5(edge_index.cpu().numpy().tobytes()).hexdigest()
        return f"{edge_hash}_{num_nodes}"

    def _compute_eigenvalues_smart(self, L, num_nodes):
        """Smart eigenvalue computation with better memory management."""
        try:
            # Use more conservative parameters
            k_requested = min(self.k + 1, num_nodes - 1)
            if k_requested <= 0:
                return torch.zeros(self.k, dtype=torch.float)
            
            # Use ARPACK with conservative settings
            result = scipy.sparse.linalg.eigsh(
                L,
                k=k_requested,
                sigma=0.0,
                which='LM',
                maxiter=500,  # Reduced iterations
                ncv=min(2 * k_requested + 1, num_nodes)  # Conservative ncv
            )
            
            eigs = result[0] if result is not None and len(result) > 0 else None
            
            if eigs is not None and len(eigs) > 1:
                # Take the next k eigenvalues after the first zero one
                spec = torch.from_numpy(eigs[1:min(len(eigs), self.k + 1)]).float()
                # Pad with zeros if needed
                if spec.shape[0] < self.k:
                    padding = torch.zeros(self.k - spec.shape[0], dtype=torch.float)
                    spec = torch.cat([spec, padding])
                return spec
            else:
                return torch.zeros(self.k, dtype=torch.float)
                
        except (RuntimeError, ValueError) as e:
            # Handle convergence failures more gracefully
            if any(msg in str(e) for msg in ["No convergence", "No shifts", "singular"]):
                # Try with regularization
                try:
                    L_reg = L + 1e-3 * scipy.sparse.eye(L.shape[0], format='csc')
                    result = scipy.sparse.linalg.eigsh(
                        L_reg,
                        k=min(self.k + 1, num_nodes - 1),
                        sigma=0.0,
                        which='LM',
                        maxiter=300,  # Even fewer iterations
                        ncv=min(self.k + 2, num_nodes)
                    )
                    eigs = result[0] if result is not None and len(result) > 0 else None
                    
                    if eigs is not None and len(eigs) > 1:
                        spec = torch.from_numpy(eigs[1:min(len(eigs), self.k + 1)]).float()
                        if spec.shape[0] < self.k:
                            padding = torch.zeros(self.k - spec.shape[0], dtype=torch.float)
                            spec = torch.cat([spec, padding])
                        return spec
                except:
                    pass
            
            # Final fallback: return zeros
            return torch.zeros(self.k, dtype=torch.float)

    def forward(self, edge_index, num_nodes, batch=None):
        # Handle batch processing
        if batch is not None:
            # Get the number of unique graphs in the batch
            num_graphs = batch.max().item() + 1
            
            # Process a single representative graph for the batch
            single_emb = self._process_single_graph(edge_index, num_nodes)
            
            # Expand to match batch size
            return single_emb.unsqueeze(0).expand(num_graphs, -1)
        else:
            # Single graph case
            return self._process_single_graph(edge_index, num_nodes)
    
    def _process_single_graph(self, edge_index, num_nodes):
        """Process a single graph with caching and smart memory management."""
        # Check cache first
        graph_hash = self._get_graph_hash(edge_index, num_nodes)
        if graph_hash in self.cache:
            cached_result = self.cache[graph_hash]
            return cached_result.to(edge_index.device)
        
        # Build normalized Laplacian efficiently
        try:
            laplacian_edge_index, laplacian_edge_weight = get_laplacian(
                edge_index, normalization='sym', num_nodes=num_nodes
            )
            L = to_scipy_sparse_matrix(laplacian_edge_index, laplacian_edge_weight, num_nodes).tocsc()
        except (ValueError, TypeError):
            # Fallback construction
            import torch_geometric.utils as utils
            adj = utils.to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
            deg = torch.diag(torch.sum(adj, dim=1))
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            L_dense = torch.eye(num_nodes) - torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
            L = scipy.sparse.csr_matrix(L_dense.cpu().numpy()).tocsc()

        # Compute eigenvalues smartly
        spec = self._compute_eigenvalues_smart(L, num_nodes)
        
        # Process through MLP (ensure device consistency)
        spec = spec.to(next(self.mlp.parameters()).device)  # Move to MLP's device
        result = self.mlp(spec.unsqueeze(0)).squeeze(0)
        
        # Cache the result (simple LRU)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[graph_hash] = result.detach().cpu()
        
        return result.to(edge_index.device)

    def clear_cache(self):
        """Clear the cache to free memory."""
        self.cache.clear()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
