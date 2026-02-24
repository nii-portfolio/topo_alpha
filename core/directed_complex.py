import numpy as np
from typing import List, Set, Tuple, Dict, Optional, Any
from collections import defaultdict
import itertools
from dataclasses import dataclass


@dataclass
class DirectedSimplex:
    """Represents a directed simplex with orientation."""
    vertices: Tuple[int, ...]
    orientation: int  # +1 or -1
    filtration_value: float
    birth_time: float
    
    def __hash__(self):
        return hash((self.vertices, self.orientation))
    
    def dimension(self) -> int:
        return len(self.vertices) - 1


class DirectedFlagComplex:
    """
    Construction of directed flag complex from directed graph.
    A directed clique (totally ordered subset) forms a simplex.
    """
    
    def __init__(self, adjacency_matrix: np.ndarray, 
                 filtration_values: Optional[np.ndarray] = None) -> None:
        """
        Args:
            adjacency_matrix: Directed adjacency matrix (asymmetric)
            filtration_values: Edge weights for filtration [N, N]
        """
        self.n_vertices = adjacency_matrix.shape[0]
        self.adj = adjacency_matrix
        self.filtration = filtration_values if filtration_values is not None else adjacency_matrix
        
        # Cache for computed simplices
        self._simplices_cache: Dict[int, List[DirectedSimplex]] = {}
        self._adjacency_list = self._build_adjacency_list()
    
    def _build_adjacency_list(self) -> Dict[int, Set[int]]:
        """Convert matrix to adjacency list for faster traversal."""
        adj_list = defaultdict(set)
        for i in range(self.n_vertices):
            for j in range(self.n_vertices):
                if self.adj[i, j] > 0 and i != j:
                    adj_list[i].add(j)
        return adj_list
    
    def _is_directed_clique(self, vertices: List[int]) -> bool:
        """
        Check if vertices form a directed clique (total ordering).
        For vertices [v0, v1, ..., vk], we need edges vi -> vj for all i < j.
        """
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if self.adj[vertices[i], vertices[j]] == 0:
                    return False
        return True
    
    def _get_filtration_value(self, simplex: Tuple[int, ...]) -> float:
        """Compute filtration value as max edge weight in simplex."""
        max_val = 0.0
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                max_val = max(max_val, self.filtration[simplex[i], simplex[j]])
        return max_val
    
    def _compute_orientation(self, vertices: Tuple[int, ...]) -> int:
        """
        Compute orientation of simplex based on vertex ordering.
        +1 if even permutation of natural order, -1 if odd.
        """
        # Sort vertices to get natural order
        sorted_verts = sorted(vertices)
        # Count inversions
        inversions = 0
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if vertices[i] > vertices[j]:
                    inversions += 1
        return 1 if inversions % 2 == 0 else -1
    
    def construct_simplices(self, max_dimension: int = 3) -> Dict[int, List[DirectedSimplex]]:
        """
        Construct all directed simplices up to max_dimension.
        
        Returns:
            Dictionary mapping dimension to list of directed simplices
        """
        if self._simplices_cache:
            return self._simplices_cache
        
        simplices = {d: [] for d in range(max_dimension + 1)}
        
        # 0-simplices (vertices)
        for v in range(self.n_vertices):
            simplices[0].append(DirectedSimplex(
                vertices=(v,),
                orientation=1,
                filtration_value=0.0,
                birth_time=0.0
            ))
        
        # Higher dimensions via breadth-first search
        for dim in range(1, max_dimension + 1):
            print(f"Constructing {dim}-simplices...")
            
            # Find all directed cliques of size dim+1
            cliques = self._find_directed_cliques(dim + 1)
            
            for clique in cliques:
                # Determine orientation based on natural ordering
                orientation = self._compute_orientation(tuple(clique))
                filt_val = self._get_filtration_value(tuple(clique))
                
                simplices[dim].append(DirectedSimplex(
                    vertices=tuple(clique),
                    orientation=orientation,
                    filtration_value=filt_val,
                    birth_time=filt_val
                ))
        
        self._simplices_cache = simplices
        return simplices
    
    def _find_directed_cliques(self, size: int) -> List[List[int]]:
        """Find all directed cliques of given size using DFS."""
        cliques = []
        
        def dfs(current_clique: List[int], start_vertex: int):
            if len(current_clique) == size:
                cliques.append(current_clique.copy())
                return
            
            # Try to extend current clique
            candidates = self._adjacency_list[current_clique[-1]] if current_clique else set(range(self.n_vertices))
            
            for v in candidates:
                if v in current_clique:
                    continue
                
                # Check if v can be added (must have edges from all in current_clique)
                can_add = True
                for u in current_clique:
                    if self.adj[u, v] == 0:
                        can_add = False
                        break
                
                if can_add:
                    current_clique.append(v)
                    dfs(current_clique, v + 1)
                    current_clique.pop()
        
        dfs([], 0)
        return cliques
    
    def get_boundary_matrix(self, dimension: int) -> np.ndarray:
        """
        Compute boundary matrix for persistent homology.
        
        Args:
            dimension: Simplex dimension
        
        Returns:
            Boundary matrix as numpy array
        """
        if dimension not in self._simplices_cache:
            self.construct_simplices(dimension)
        
        simplices_d = self._simplices_cache[dimension]
        simplices_d_minus_1 = self._simplices_cache.get(dimension - 1, [])
        
        if not simplices_d_minus_1:
            return np.array([]).reshape(len(simplices_d), 0)
        
        # Map simplices to indices
        simplex_to_idx = {s.vertices: i for i, s in enumerate(simplices_d_minus_1)}
        
        # Build boundary matrix
        n_rows = len(simplices_d_minus_1)
        n_cols = len(simplices_d)
        boundary = np.zeros((n_rows, n_cols))
        
        for j, simplex in enumerate(simplices_d):
            # Boundary is alternating sum of faces
            for i, v in enumerate(simplex.vertices):
                # Face obtained by removing vertex i
                face = simplex.vertices[:i] + simplex.vertices[i+1:]
                if face in simplex_to_idx:
                    row = simplex_to_idx[face]
                    # Orientation sign: (-1)^i
                    boundary[row, j] = ((-1) ** i) * simplex.orientation
        
        return boundary
    
    def to_undirected(self) -> np.ndarray:
        """Convert to undirected adjacency for standard persistence."""
        return np.maximum(self.adj, self.adj.T)


class HierarchicalRegimeDetector:
    """
    Detect market regimes using multi-scale directed complexes.
    """
    
    def __init__(self, thresholds: List[float] = [0.3, 0.5, 0.7, 0.9]):
        self.thresholds = thresholds
        self.regime_history = []
    
    def detect_regime(self, returns: np.ndarray, 
                     adjacency_matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect current market regime using topological features at multiple scales.
        
        Args:
            returns: Asset returns [T, N]
            adjacency_matrices: Dict of adjacency matrices at different thresholds
        
        Returns:
            Regime classification and confidence
        """
        regime_features = {}
        
        for scale_name, adj in adjacency_matrices.items():
            # Build directed complex
            dfc = DirectedFlagComplex(adj)
            simplices = dfc.construct_simplices(max_dimension=2)
            
            # Compute simplex counts as features
            n_edges = len(simplices[1])
            n_triangles = len(simplices[2])
            
            # Density metrics
            n = adj.shape[0]
            possible_edges = n * (n - 1) / 2
            edge_density = n_edges / possible_edges if possible_edges > 0 else 0
            
            # Clustering coefficient approximation
            clustering = 0.0
            if n_edges > 0:
                clustering = n_triangles / (n_edges * (n - 2) / 3) if n_edges > 0 else 0
            
            regime_features[scale_name] = {
                "edge_density": edge_density,
                "triangle_density": n_triangles / (n ** 3) if n > 0 else 0,
                "clustering": clustering,
                "n_simplices": {d: len(simplices[d]) for d in simplices}
            }
        
        # Classify regime based on connectivity patterns
        avg_density = np.mean([f["edge_density"] for f in regime_features.values()])
        
        if avg_density < 0.1:
            regime = "fragmented"
        elif avg_density < 0.3:
            regime = "weakly_connected"
        elif avg_density < 0.6:
            regime = "clustered"
        else:
            regime = "highly_connected"
        
        # Compute regime stability (persistence across scales)
        densities = [f["edge_density"] for f in regime_features.values()]
        stability = 1.0 - np.std(densities)
        
        return {
            "regime": regime,
            "confidence": stability,
            "features": regime_features,
            "scale_consistency": stability > 0.7
        }