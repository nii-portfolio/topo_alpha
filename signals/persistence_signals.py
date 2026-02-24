import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


@dataclass
class WassersteinTracker:
    """Tracks Wasserstein distance between persistence diagrams over time."""
    history: List[Dict[str, Any]]
    barycenter: Dict[int, np.ndarray]
    distances: List[float]
    warning_threshold: float = 2.0
    
    def __init__(self, max_dim: int = 1, threshold: float = 2.0):
        self.history = []
        self.max_dim = max_dim
        self.barycenter = {d: np.array([]).reshape(0, 2) for d in range(max_dim + 1)}
        self.distances = []
        self.warning_threshold = threshold
        self.alerts_triggered = 0


def wasserstein_distance(dgm1: np.ndarray, 
                         dgm2: np.ndarray,
                         p: int = 2) -> float:
    """
    Compute p-Wasserstein distance between two persistence diagrams.
    Uses optimal matching with death-on-diagonal projections.
    
    Args:
        dgm1: First diagram [n1, 2]
        dgm2: Second diagram [n2, 2]
        p: Order of Wasserstein distance
    
    Returns:
        Wasserstein distance
    """
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    if len(dgm1) == 0 or len(dgm2) == 0:
        # Distance to empty diagram
        non_empty = dgm1 if len(dgm1) > 0 else dgm2
        # Project to diagonal (birth = death)
        diagonal_dists = np.abs(non_empty[:, 0] - non_empty[:, 1])
        return np.sum(diagonal_dists ** p) ** (1/p)
    
    # Create cost matrix
    n1, n2 = len(dgm1), len(dgm2)
    
    # Cost between real points
    cost_matrix = np.zeros((n1 + n2, n1 + n2))
    
    # Real to real
    for i in range(n1):
        for j in range(n2):
            cost_matrix[i, j] = np.sum(np.abs(dgm1[i] - dgm2[j]) ** p)
    
    # Real to diagonal projection (for dgm1 points)
    for i in range(n1):
        # Project dgm1[i] to diagonal
        diag_proj = (dgm1[i, 0] + dgm1[i, 1]) / 2
        dist_to_diag = np.abs(dgm1[i, 0] - dgm1[i, 1]) / np.sqrt(2)
        cost_matrix[i, n2 + i] = dist_to_diag ** p
    
    # Diagonal projection to real (for dgm2 points)
    for j in range(n2):
        diag_proj = (dgm2[j, 0] + dgm2[j, 1]) / 2
        dist_to_diag = np.abs(dgm2[j, 0] - dgm2[j, 1]) / np.sqrt(2)
        cost_matrix[n1 + j, j] = dist_to_diag ** p
    
    # Diagonal to diagonal is zero (already initialized)
    
    # Solve optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Compute total cost
    total_cost = np.sum(cost_matrix[row_ind, col_ind])
    
    return total_cost ** (1/p)


def compute_barycenter(diagrams: List[np.ndarray],
                       weights: Optional[np.ndarray] = None,
                       n_iterations: int = 10) -> np.ndarray:
    """
    Compute Wasserstein barycenter of multiple persistence diagrams.
    Uses iterative refinement.
    
    Args:
        diagrams: List of persistence diagrams
        weights: Optional weights for each diagram
        n_iterations: Number of refinement iterations
    
    Returns:
        Barycenter diagram
    """
    if not diagrams:
        return np.array([]).reshape(0, 2)
    
    if weights is None:
        weights = np.ones(len(diagrams)) / len(diagrams)
    
    # Initialize barycenter as weighted average of means
    all_points = np.vstack([d for d in diagrams if len(d) > 0])
    if len(all_points) == 0:
        return np.array([]).reshape(0, 2)
    
    # Start with average number of points
    avg_n_points = int(np.mean([len(d) for d in diagrams]))
    
    # Initialize barycenter points randomly from data
    indices = np.random.choice(len(all_points), size=min(avg_n_points, len(all_points)), replace=False)
    barycenter = all_points[indices].copy()
    
    for _ in range(n_iterations):
        # For each barycenter point, find optimal matches in all diagrams
        new_points = []
        
        for b_point in barycenter:
            matched_points = []
            for dgm, w in zip(diagrams, weights):
                if len(dgm) == 0:
                    continue
                
                # Find closest point in diagram
                dists = np.sum(np.abs(dgm - b_point) ** 2, axis=1)
                closest_idx = np.argmin(dists)
                closest_point = dgm[closest_idx]
                
                # Check if projection to diagonal is closer
                diag_proj = np.array([(b_point[0] + b_point[1]) / 2] * 2)
                dist_to_diag = np.sum(np.abs(b_point - diag_proj) ** 2)
                
                if dists[closest_idx] < dist_to_diag:
                    matched_points.append((closest_point, w))
            
            if matched_points:
                # Weighted average of matched points
                pts, ws = zip(*matched_points)
                new_point = np.average(pts, axis=0, weights=ws)
                new_points.append(new_point)
        
        if new_points:
            barycenter = np.array(new_points)
    
    return barycenter


class PersistenceEarlyWarningSystem:
    """
    Early warning system based on Wasserstein tracking of persistence diagrams.
    Detects topological regime shifts before they manifest in prices.
    """
    
    def __init__(self, max_dim: int = 1, window_size: int = 20):
        self.max_dim = max_dim
        self.window_size = window_size
        self.trackers = {d: WassersteinTracker(max_dim=0) for d in range(max_dim + 1)}
        self.diagram_history = {d: [] for d in range(max_dim + 1)}
        self.baseline_established = False
    
    def update(self, 
               persistence_diagrams: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """
        Update tracking with new persistence diagrams.
        
        Args:
            persistence_diagrams: Dict[dim, diagram_array]
        
        Returns:
            Alert status and metrics
        """
        alerts = []
        metrics = {}
        
        for dim in range(self.max_dim + 1):
            dgm = persistence_diagrams.get(dim, np.array([]).reshape(0, 2))
            self.diagram_history[dim].append(dgm)
            
            # Keep only recent history
            if len(self.diagram_history[dim]) > self.window_size:
                self.diagram_history[dim].pop(0)
            
            tracker = self.trackers[dim]
            tracker.history.append({"diagram": dgm, "timestamp": len(tracker.history)})
            
            if len(self.diagram_history[dim]) < 5:
                continue
            
            # Compute barycenter of recent history
            recent_diagrams = self.diagram_history[dim][-10:]
            barycenter = compute_barycenter(recent_diagrams[:-1])  # Exclude current
            
            # Compute distance from current to barycenter
            if len(barycenter) > 0:
                dist = wasserstein_distance(dgm, barycenter, p=2)
                tracker.distances.append(dist)
                
                # Check for anomaly
                if len(tracker.distances) > 10:
                    mean_dist = np.mean(tracker.distances[-20:])
                    std_dist = np.std(tracker.distances[-20:])
                    
                    if std_dist > 0:
                        z_score = (dist - mean_dist) / std_dist
                        metrics[f"dim{dim}_zscore"] = z_score
                        
                        if z_score > tracker.warning_threshold:
                            alerts.append({
                                "dimension": dim,
                                "severity": "high" if z_score > 3 else "medium",
                                "z_score": z_score,
                                "distance": dist,
                                "type": "topological_regime_shift"
                            })
                            tracker.alerts_triggered += 1
        
        return {
            "alerts": alerts,
            "metrics": metrics,
            "status": "critical" if any(a["severity"] == "high" for a in alerts) else \
                     "warning" if alerts else "normal",
            "n_alerts": len(alerts)
        }
    
    def get_stability_score(self) -> float:
        """
        Compute overall topological stability score.
        Higher = more stable, Lower = regime change likely.
        """
        scores = []
        for dim in range(self.max_dim + 1):
            tracker = self.trackers[dim]
            if len(tracker.distances) > 5:
                recent_volatility = np.std(tracker.distances[-10:])
                scores.append(1.0 / (1.0 + recent_volatility))
        
        return float(np.mean(scores)) if scores else 0.5


class TopologicalMomentumSignal:
    """
    Generate trading signals from persistence diagram dynamics.
    """
    
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.persistence_history = []
    
    def compute_persistence_landscape(self, 
                                       diagram: np.ndarray,
                                       resolution: int = 100) -> np.ndarray:
        """
        Convert diagram to persistence landscape (stable representation).
        """
        if len(diagram) == 0:
            return np.zeros(resolution)
        
        # Create landscape functions
        x = np.linspace(0, np.max(diagram[:, 1]), resolution)
        landscape = np.zeros(resolution)
        
        for birth, death in diagram:
            persistence = death - birth
            mid = (birth + death) / 2
            
            # Triangle function for each point
            triangle = np.maximum(0, persistence - np.abs(x - mid))
            landscape = np.maximum(landscape, triangle)
        
        return landscape
    
    def generate_signal(self, 
                       current_diagram: np.ndarray,
                       reference_diagram: np.ndarray) -> float:
        """
        Generate signal based on landscape changes.
        
        Returns:
            Signal in [-1, 1] where positive = increasing topological activity
        """
        current_land = self.compute_persistence_landscape(current_diagram)
        ref_land = self.compute_persistence_landscape(reference_diagram)
        
        # Compute correlation between landscapes
        if np.std(current_land) > 0 and np.std(ref_land) > 0:
            correlation = np.corrcoef(current_land, ref_land)[0, 1]
            # Signal based on deviation
            signal = 1.0 - correlation
        else:
            signal = 0.0
        
        return np.clip(signal, -1, 1)