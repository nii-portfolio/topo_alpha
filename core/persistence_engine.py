import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
from functools import lru_cache
import hashlib

try:
    import ripser # pyright: ignore[reportMissingImports]
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    warnings.warn("Ripser not available. Install with: pip install ripser")

try:
    import gudhi # pyright: ignore[reportMissingImports]
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    warnings.warn("GUDHI not available. Install with: pip install gudhi")


@dataclass
class PersistenceDiagram:
    """Container for persistence diagram with metadata."""
    dimension: int
    birth_death_pairs: np.ndarray  # [n_points, 2]
    persistence_points: np.ndarray  # [n_points, 2] (birth, death)
    
    def __post_init__(self):
        if len(self.persistence_points) > 0:
            self.persistence = self.persistence_points[:, 1] - self.persistence_points[:, 0]
        else:
            self.persistence = np.array([])
    
    def total_persistence(self) -> float:
        return np.sum(self.persistence)
    
    def persistent_entropy(self) -> float:
        """Compute persistence entropy."""
        if len(self.persistence) == 0:
            return 0.0
        p = self.persistence / self.persistence.sum()
        return -np.sum(p * np.log(p + 1e-10))


class FastPersistenceComputer:
    """
    High-performance persistence computation with caching and 
    optimizations for financial time series.
    """
    
    def __init__(self, use_ripser: bool = True, max_dim: int = 2):
        self.use_ripser = use_ripser and RIPSER_AVAILABLE
        self.max_dim = max_dim
        self._cache = {}
        self.computation_stats = {
            "ripser_calls": 0,
            "gudhi_calls": 0,
            "cache_hits": 0
        }
    
    def _get_cache_key(self, distance_matrix: np.ndarray, 
                       max_dim: int, 
                       threshold: float) -> str:
        """Generate cache key from matrix content."""
        # Use matrix hash for caching
        matrix_bytes = distance_matrix.tobytes()
        return hashlib.md5(matrix_bytes).hexdigest() + f"_{max_dim}_{threshold}"
    
    def compute_ripser(self, distance_matrix: np.ndarray, 
                       max_dim: int = 2) -> Dict[int, PersistenceDiagram]:
        """
        Compute persistent homology using Ripser (optimized for speed).
        
        Args:
            distance_matrix: Distance matrix [N, N]
            max_dim: Maximum homology dimension
        
        Returns:
            Dictionary of persistence diagrams by dimension
        """
        if not RIPSER_AVAILABLE:
            raise ImportError("Ripser not installed")
        
        self.computation_stats["ripser_calls"] += 1
        
        # Ripser expects condensed distance matrix or point cloud
        # For distance matrix, we use the squareform
        from scipy.spatial.distance import squareform
        
        # Ensure matrix is symmetric and zero-diagonal
        dist_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        
        # Convert to condensed form for efficiency
        try:
            condensed = squareform(dist_matrix, checks=False)
        except ValueError:
            # If not perfect distance matrix, use as-is
            condensed = dist_matrix
        
        # Compute persistence
        diagrams = ripser.ripser(
            condensed,
            maxdim=max_dim,
            thresh=np.max(dist_matrix) * 1.1,
            coeff=2,
            do_cocycles=False
        )['dgms']
        
        result = {}
        for dim, dgm in enumerate(diagrams):
            # Filter out infinity points
            finite = dgm[dgm[:, 1] < np.inf]
            result[dim] = PersistenceDiagram(
                dimension=dim,
                birth_death_pairs=finite,
                persistence_points=finite
            )
        
        return result
    
    def compute_gudhi(self, distance_matrix: np.ndarray,
                      max_dim: int = 2) -> Dict[int, PersistenceDiagram]:
        """
        Compute persistent homology using GUDHI (more features).
        
        Args:
            distance_matrix: Distance matrix [N, N]
            max_dim: Maximum homology dimension
        
        Returns:
            Dictionary of persistence diagrams by dimension
        """
        if not GUDHI_AVAILABLE:
            raise ImportError("GUDHI not installed")
        
        self.computation_stats["gudhi_calls"] += 1
        
        # Build Rips complex
        rips_complex = gudhi.RipsComplex(
            distance_matrix=distance_matrix,
            max_edge_length=np.max(distance_matrix) * 1.1
        )
        
        # Create simplex tree
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        
        # Compute persistence
        persistence = simplex_tree.persistence()
        
        # Organize by dimension
        result = {d: [] for d in range(max_dim + 1)}
        for dim, (birth, death) in persistence:
            if dim <= max_dim:
                result[dim].append([birth, death])
        
        # Convert to PersistenceDiagram objects
        diagrams = {}
        for dim, pairs in result.items():
            arr = np.array(pairs) if pairs else np.array([]).reshape(0, 2)
            # Filter infinity
            finite = arr[arr[:, 1] < np.inf] if len(arr) > 0 else arr
            diagrams[dim] = PersistenceDiagram(
                dimension=dim,
                birth_death_pairs=finite,
                persistence_points=finite
            )
        
        return diagrams
    
    def compute(self, distance_matrix: np.ndarray,
                max_dim: Optional[int] = None,
                use_cache: bool = True) -> Dict[int, PersistenceDiagram]:
        """
        Main computation method with caching.
        
        Args:
            distance_matrix: Distance matrix
            max_dim: Maximum dimension (default: self.max_dim)
            use_cache: Whether to use caching
        
        Returns:
            Persistence diagrams
        """
        max_dim = max_dim or self.max_dim
        
        if use_cache:
            cache_key = self._get_cache_key(distance_matrix, max_dim, 0.0)
            if cache_key in self._cache:
                self.computation_stats["cache_hits"] += 1
                return self._cache[cache_key]
        
        # Choose backend
        if self.use_ripser:
            result = self.compute_ripser(distance_matrix, max_dim)
        else:
            result = self.compute_gudhi(distance_matrix, max_dim)
        
        if use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def compute_directed_persistence(self, 
                                     directed_adj: np.ndarray,
                                     undirected_fallback: bool = True) -> Dict[int, PersistenceDiagram]:
        """
        Compute persistence for directed graph.
        Uses directed flag complex or falls back to undirected.
        
        Args:
            directed_adj: Directed adjacency matrix
            undirected_fallback: If True, convert to undirected
        
        Returns:
            Persistence diagrams
        """
        if undirected_fallback:
            # Symmetrize for standard persistence
            dist_matrix = np.maximum(directed_adj, directed_adj.T)
            # Convert to distance (inverse of similarity)
            dist_matrix = 1.0 / (dist_matrix + 1e-5)
            np.fill_diagonal(dist_matrix, 0)
            return self.compute(dist_matrix)
        else:
            # TODO: Implement directed persistence (requires ordered complex)
            raise NotImplementedError("True directed persistence not yet implemented")
    
    def sliding_window_persistence(self, 
                                   time_series: np.ndarray,
                                   window_size: int = 50,
                                   stride: int = 10) -> List[Dict[int, PersistenceDiagram]]:
        """
        Compute persistence diagrams over sliding windows.
        Optimized for streaming financial data.
        
        Args:
            time_series: [T, N] time series
            window_size: Size of sliding window
            stride: Step size
        
        Returns:
            List of persistence diagrams for each window
        """
        T, N = time_series.shape
        results = []
        
        for start in range(0, T - window_size + 1, stride):
            window = time_series[start:start + window_size]
            
            # Compute correlation distance
            corr = np.corrcoef(window.T)
            # Handle NaN
            corr = np.nan_to_num(corr, nan=0.0)
            dist = 1 - np.abs(corr)
            np.fill_diagonal(dist, 0)
            
            # Compute persistence
            dgm = self.compute(dist, use_cache=False)  # Don't cache sliding windows
            results.append(dgm)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get computation statistics."""
        return self.computation_stats.copy()
    
    def clear_cache(self):
        """Clear computation cache."""
        self._cache.clear()


class RealTimePersistenceTracker:
    """
    Optimized tracker for real-time persistence computation.
    Uses incremental updates and approximation.
    """
    
    def __init__(self, n_assets: int, max_dim: int = 1):
        self.n_assets = n_assets
        self.max_dim = max_dim
        self.computer = FastPersistenceComputer()
        self.history = []
        self.approximate = True  # Use approximation for speed
    
    def update(self, new_returns: np.ndarray, 
               correlation_buffer: List[np.ndarray]) -> Optional[Dict[int, PersistenceDiagram]]:
        """
        Update persistence with new data point.
        
        Args:
            new_returns: New returns vector [N]
            correlation_buffer: List of recent correlation matrices
        
        Returns:
            Updated persistence diagram or None if not computed
        """
        self.history.append(new_returns)
        
        # Compute persistence every N steps or on significant change
        if len(self.history) % 20 == 0:  # Every 20 ticks
            if len(correlation_buffer) > 0:
                # Use most recent correlation
                corr = correlation_buffer[-1]
                dist = 1 - np.abs(corr)
                np.fill_diagonal(dist, 0)
                
                if self.approximate and self.n_assets > 50:
                    # Subsample for large universes
                    indices = np.random.choice(
                        self.n_assets, 
                        size=min(50, self.n_assets), 
                        replace=False
                    )
                    dist = dist[np.ix_(indices, indices)]
                
                return self.computer.compute(dist, use_cache=True)
        
        return None