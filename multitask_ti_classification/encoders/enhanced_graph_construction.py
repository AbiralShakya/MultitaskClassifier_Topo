import torch
import numpy as np
try:
    from scipy.spatial import Voronoi, distance_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available, using fallback graph construction")

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import VoronoiNN
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("Warning: PyMatGen not available, using distance-based graph construction")

import torch_geometric
from torch_geometric.data import Data as PyGData

class EnhancedGraphConstructor:
    """
    Enhanced crystal graph construction using Voronoi-Dirichlet polyhedra
    and improved edge features as described in the Nature paper.
    """
    
    def __init__(self, max_neighbors=12, cutoff_radius=15.0):
        self.max_neighbors = max_neighbors
        self.cutoff_radius = cutoff_radius
        if PYMATGEN_AVAILABLE:
            self.voronoi_nn = VoronoiNN()
        else:
            self.voronoi_nn = None
        
        # Covalent radii for bond validation (Cordero et al.)
        self.covalent_radii = {
            1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66,
            9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
            16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60,
            23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32,
            30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16,
            # Add more as needed...
        }
    
    def construct_enhanced_graph(self, structure) -> PyGData:
        """
        Construct enhanced crystal graph using Voronoi tessellation
        """
        # Get atomic positions and numbers
        positions = structure.cart_coords
        atomic_numbers = [site.specie.Z for site in structure]
        
        # Build adjacency using Voronoi tessellation
        edge_indices, edge_features = self._build_voronoi_edges(structure)
        
        # Create node features (will be enhanced by EnhancedAtomicFeatures)
        node_features = torch.tensor(atomic_numbers, dtype=torch.float).unsqueeze(1)
        
        # Create PyTorch Geometric data object
        graph = PyGData(
            x=node_features,
            edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            pos=torch.tensor(positions, dtype=torch.float),
            atomic_numbers=atomic_numbers
        )
        
        return graph
    
    def _build_voronoi_edges(self, structure):
        """Build edges using Voronoi-Dirichlet polyhedra or fallback to distance"""
        edge_indices = []
        edge_features = []
        
        if not PYMATGEN_AVAILABLE or self.voronoi_nn is None:
            # Fallback to distance-based construction
            return self._build_distance_edges(structure)
        
        for i, site in enumerate(structure):
            try:
                # Get Voronoi neighbors
                neighbors = self.voronoi_nn.get_nn_info(structure, i)
                
                # Filter and sort neighbors
                valid_neighbors = []
                for neighbor in neighbors:
                    j = neighbor['site_index']
                    distance = neighbor['weight']  # Voronoi weight
                    
                    # Validate bond using covalent radii
                    if self._is_valid_bond(structure[i], structure[j], distance):
                        valid_neighbors.append((j, distance, neighbor))
                
                # Sort by distance and take top max_neighbors
                valid_neighbors.sort(key=lambda x: x[1])
                valid_neighbors = valid_neighbors[:self.max_neighbors]
                
                # Create edges
                for j, distance, neighbor_info in valid_neighbors:
                    edge_indices.append([i, j])
                    
                    # Enhanced edge features
                    edge_feat = self._compute_edge_features(
                        structure[i], structure[j], distance, neighbor_info
                    )
                    edge_features.append(edge_feat)
                    
            except Exception as e:
                # Fallback to distance-based neighbors
                print(f"Voronoi failed for site {i}, using distance fallback: {e}")
                self._add_distance_neighbors(structure, i, edge_indices, edge_features)
        
        return edge_indices, edge_features
    
    def _build_distance_edges(self, structure):
        """Fallback distance-based edge construction"""
        edge_indices = []
        edge_features = []
        
        # Simple distance-based neighbor finding
        positions = structure.cart_coords if hasattr(structure, 'cart_coords') else structure
        
        for i in range(len(positions)):
            distances = []
            for j in range(len(positions)):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.cutoff_radius:
                        distances.append((j, dist))
            
            # Sort by distance and take top neighbors
            distances.sort(key=lambda x: x[1])
            distances = distances[:self.max_neighbors]
            
            for j, distance in distances:
                edge_indices.append([i, j])
                
                # Simple edge features
                features = [
                    distance / self.cutoff_radius,  # Normalized distance
                    1.0,  # Placeholder for same element
                    0.0, 0.0, 0.0  # Padding
                ]
                # Pad to match enhanced feature length
                features.extend([0.0] * 10)  # Distance bins
                edge_features.append(features)
        
        return edge_indices, edge_features
    
    def _is_valid_bond(self, site1, site2, distance):
        """Validate bond using covalent radii criteria"""
        z1, z2 = site1.specie.Z, site2.specie.Z
        r1 = self.covalent_radii.get(z1, 1.5)
        r2 = self.covalent_radii.get(z2, 1.5)
        
        # Bond is valid if distance is within reasonable range
        min_dist = 0.5  # Minimum reasonable distance
        max_dist = 1.5 * (r1 + r2)  # 1.5x sum of covalent radii
        
        return min_dist < distance < min(max_dist, self.cutoff_radius)
    
    def _compute_edge_features(self, site1, site2, distance, neighbor_info):
        """Compute enhanced edge features"""
        features = []
        
        # 1. Distance (normalized)
        normalized_dist = distance / self.cutoff_radius
        features.append(normalized_dist)
        
        # 2. Binned distance (10 bins)
        dist_bins = torch.zeros(10)
        bin_idx = min(9, int(normalized_dist * 10))
        dist_bins[bin_idx] = 1.0
        features.extend(dist_bins.tolist())
        
        # 3. Bond type features
        z1, z2 = site1.specie.Z, site2.specie.Z
        
        # Same element bond
        features.append(1.0 if z1 == z2 else 0.0)
        
        # Electronegativity difference
        en1 = self._get_electronegativity(z1)
        en2 = self._get_electronegativity(z2)
        en_diff = abs(en1 - en2) / 4.0  # Normalize by max EN difference
        features.append(en_diff)
        
        # 4. Coordination features from Voronoi
        if 'weight' in neighbor_info:
            voronoi_weight = neighbor_info['weight']
            features.append(voronoi_weight)
        else:
            features.append(0.0)
        
        # 5. Angular features (if available)
        if 'face_dist' in neighbor_info:
            face_dist = neighbor_info['face_dist']
            features.append(face_dist)
        else:
            features.append(0.0)
        
        return features
    
    def _get_electronegativity(self, atomic_number):
        """Get electronegativity for an element"""
        en_dict = {
            1: 2.20, 2: 4.16, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
            9: 3.98, 10: 4.79, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19,
            16: 2.58, 17: 3.16, 18: 3.24, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54,
            # Add more...
        }
        return en_dict.get(atomic_number, 2.0)  # Default to 2.0
    
    def _add_distance_neighbors(self, structure, i, edge_indices, edge_features):
        """Fallback distance-based neighbor finding"""
        site = structure[i]
        neighbors = structure.get_neighbors(site, self.cutoff_radius)
        
        # Sort by distance and take top neighbors
        neighbors.sort(key=lambda x: x[1])
        neighbors = neighbors[:self.max_neighbors]
        
        for neighbor_site, distance in neighbors:
            j = structure.index(neighbor_site)
            edge_indices.append([i, j])
            
            # Simple edge features for fallback
            features = [
                distance / self.cutoff_radius,  # Normalized distance
                1.0 if site.specie.Z == neighbor_site.specie.Z else 0.0,  # Same element
                0.0, 0.0, 0.0  # Padding for consistency
            ]
            # Pad to match enhanced feature length
            features.extend([0.0] * 10)  # Distance bins
            edge_features.append(features)


class ImprovedCrystalGraphLoader:
    """
    Improved crystal graph loader that uses enhanced graph construction
    """
    
    def __init__(self, max_neighbors=12, cutoff_radius=15.0):
        self.graph_constructor = EnhancedGraphConstructor(max_neighbors, cutoff_radius)
    
    def load_from_structure(self, structure) -> PyGData:
        """Load enhanced graph from structure (pymatgen Structure or array)"""
        return self.graph_constructor.construct_enhanced_graph(structure)
    
    def load_from_poscar(self, poscar_path: str) -> PyGData:
        """Load enhanced graph from POSCAR file"""
        if PYMATGEN_AVAILABLE:
            from pymatgen.core import Structure
            structure = Structure.from_file(poscar_path)
            return self.load_from_structure(structure)
        else:
            raise ImportError("PyMatGen required for POSCAR loading")
    
    def load_from_cif(self, cif_path: str) -> PyGData:
        """Load enhanced graph from CIF file"""
        if PYMATGEN_AVAILABLE:
            from pymatgen.core import Structure
            structure = Structure.from_file(cif_path)
            return self.load_from_structure(structure)
        else:
            raise ImportError("PyMatGen required for CIF loading")