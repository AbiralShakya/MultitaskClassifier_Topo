# gpu_spectral_encoder.py
# ──────────────────────────────────────────────────────────────────────────────
"""
GPU-accelerated spectral encoder using PyTorch instead of CPU-intensive SciPy.
This version computes eigenvalues on GPU for much faster training.
"""

import torch
import torch.nn as nn
import hashlib

class GPUSpectralEncoder(nn.Module):
    """
    GPU-accelerated spectral encoder with caching and PyTorch-based eigenvalue computation.
    """
    def __init__(self, k_eigs: int, hidden: int, cache_size: int = 100):
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

    def _build_adjacency_matrix(self, edge_index, num_nodes, device):
        """Manually build adjacency matrix to avoid PyTorch Geometric dtype issues."""
        # Ensure edge_index is in the correct format
        edge_index = edge_index.long().to(device)
        
        # Initialize adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float, device=device)
        
        # Fill adjacency matrix
        row, col = edge_index
        adj[row, col] = 1.0
        adj[col, row] = 1.0  # Make it symmetric (undirected)
        
        return adj

    def _build_laplacian_matrix(self, edge_index, num_nodes, device):
        """Manually build normalized Laplacian matrix."""
        # Build adjacency matrix
        adj = self._build_adjacency_matrix(edge_index, num_nodes, device)
        
        # Compute degree matrix
        deg = torch.sum(adj, dim=1, keepdim=True)
        
        # Handle isolated nodes (degree = 0)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt[torch.isnan(deg_inv_sqrt)] = 0
        
        # Build normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        I = torch.eye(num_nodes, dtype=torch.float, device=device)
        D_inv_sqrt = torch.diag(deg_inv_sqrt.squeeze())
        
        # L = I - D^(-1/2) A D^(-1/2)
        L = I - torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        
        return L

    def _compute_eigenvalues_gpu(self, edge_index, num_nodes, device):
        """GPU-accelerated eigenvalue computation using PyTorch."""
        try:
            # Build normalized Laplacian matrix manually
            L = self._build_laplacian_matrix(edge_index, num_nodes, device)
            
            # Add small regularization to ensure numerical stability
            L = L + 1e-6 * torch.eye(num_nodes, dtype=torch.float, device=device)
            
            # Compute eigenvalues using PyTorch's GPU-accelerated eigendecomposition
            eigenvals, _ = torch.linalg.eigh(L)
            
            # Sort eigenvalues and take the k smallest non-zero ones
            eigenvals = torch.sort(eigenvals)[0]  # Sort in ascending order
            
            # Skip the first eigenvalue (should be ~0) and take next k
            k_requested = min(self.k, num_nodes - 1)
            if k_requested > 0:
                spec = eigenvals[1:k_requested + 1]
                # Pad with zeros if we don't have enough eigenvalues
                if spec.shape[0] < self.k:
                    padding = torch.zeros(self.k - spec.shape[0], dtype=torch.float, device=device)
                    spec = torch.cat([spec, padding])
            else:
                spec = torch.zeros(self.k, dtype=torch.float, device=device)
            
            return spec
            
        except Exception as e:
            # Fallback: return zeros if computation fails
            print(f"Warning: GPU eigenvalue computation failed: {e}")
            return torch.zeros(self.k, dtype=torch.float, device=device)

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
        """Process a single graph with GPU-accelerated computation."""
        device = edge_index.device
        
        # Check cache first
        graph_hash = self._get_graph_hash(edge_index, num_nodes)
        if graph_hash in self.cache:
            cached_result = self.cache[graph_hash]
            return cached_result.to(device)
        
        # Compute eigenvalues on GPU
        spec = self._compute_eigenvalues_gpu(edge_index, num_nodes, device)
        
        # --- DEBUG: Print eigenvalues for first few calls ---
        if not hasattr(self, '_print_count'):
            self._print_count = 0
        if self._print_count < 5:
            print(f"[GPUSpectralEncoder] Eigenvalues (spec) for graph with {num_nodes} nodes:", spec.detach().cpu().numpy())
            self._print_count += 1
        # ---------------------------------------------------
        
        # Process through MLP (already on correct device)
        result = self.mlp(spec.unsqueeze(0)).squeeze(0)
        
        # Cache the result (simple LRU)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[graph_hash] = result.detach().cpu()
        
        return result

    def clear_cache(self):
        """Clear the cache to free memory."""
        self.cache.clear()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()


class FastSpectralEncoder(nn.Module):
    """
    Ultra-fast spectral encoder using approximate methods for very large graphs.
    """
    def __init__(self, k_eigs: int, hidden: int, cache_size: int = 100):
        super().__init__()
        self.k = k_eigs
        self.cache_size = cache_size
        self.cache = {}
        self.mlp = nn.Sequential(
            nn.Linear(k_eigs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def _get_graph_hash(self, edge_index, num_nodes):
        """Create a hash for the graph to enable caching."""
        edge_hash = hashlib.md5(edge_index.cpu().numpy().tobytes()).hexdigest()
        return f"{edge_hash}_{num_nodes}"

    def _build_adjacency_matrix(self, edge_index, num_nodes, device):
        """Manually build adjacency matrix to avoid PyTorch Geometric dtype issues."""
        # Ensure edge_index is in the correct format
        edge_index = edge_index.long().to(device)
        
        # Initialize adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float, device=device)
        
        # Fill adjacency matrix
        row, col = edge_index
        adj[row, col] = 1.0
        adj[col, row] = 1.0  # Make it symmetric (undirected)
        
        return adj

    def _compute_approximate_eigenvalues(self, edge_index, num_nodes, device):
        """Fast approximate eigenvalue computation using power iteration."""
        try:
            # Build adjacency matrix manually
            adj = self._build_adjacency_matrix(edge_index, num_nodes, device)
            
            # Compute degree matrix
            deg = torch.sum(adj, dim=1, keepdim=True)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            deg_inv_sqrt[torch.isnan(deg_inv_sqrt)] = 0
            
            # Normalized adjacency matrix: A_norm = D^(-1/2) A D^(-1/2)
            A_norm = torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
            
            # Use power iteration to approximate largest eigenvalues
            # This is much faster than full eigendecomposition
            n = A_norm.shape[0]
            k_requested = min(self.k, n - 1)
            
            if k_requested <= 0:
                return torch.zeros(self.k, dtype=torch.float, device=device)
            
            # Initialize random vectors
            V = torch.randn(n, k_requested, device=device)
            V = torch.nn.functional.normalize(V, dim=0)
            
            # Power iteration
            for _ in range(10):  # 10 iterations usually sufficient
                V_new = torch.mm(A_norm, V)
                V_new = torch.nn.functional.normalize(V_new, dim=0)
                V = V_new
            
            # Compute approximate eigenvalues
            eigenvals = torch.diag(torch.mm(torch.mm(V.T, A_norm), V))
            
            # Convert to Laplacian eigenvalues: λ_L = 1 - λ_A
            laplacian_eigenvals = 1 - eigenvals
            
            # Sort and take the smallest k
            laplacian_eigenvals = torch.sort(laplacian_eigenvals)[0]
            
            spec = laplacian_eigenvals[:k_requested]
            
            # Pad with zeros if needed
            if spec.shape[0] < self.k:
                padding = torch.zeros(self.k - spec.shape[0], dtype=torch.float, device=device)
                spec = torch.cat([spec, padding])
            
            return spec
            
        except Exception as e:
            print(f"Warning: Fast eigenvalue computation failed: {e}")
            return torch.zeros(self.k, dtype=torch.float, device=device)

    def forward(self, edge_index, num_nodes, batch=None):
        device = edge_index.device
        
        # Check cache first
        graph_hash = self._get_graph_hash(edge_index, num_nodes)
        if graph_hash in self.cache:
            cached_result = self.cache[graph_hash]
            return cached_result.to(device)
        
        # Compute approximate eigenvalues
        spec = self._compute_approximate_eigenvalues(edge_index, num_nodes, device)
        
        # Process through MLP
        result = self.mlp(spec.unsqueeze(0)).squeeze(0)
        
        # Cache the result
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[graph_hash] = result.detach().cpu()
        
        return result

    def clear_cache(self):
        """Clear the cache to free memory."""
        self.cache.clear()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()