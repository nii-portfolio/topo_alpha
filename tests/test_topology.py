import numpy as np
import torch
import unittest
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

from core.directed_complex import DirectedFlagComplex
from core.dynamic_graph import DynamicGraphNetwork, GraphConfig
from core.persistence_engine import FastPersistenceComputer
from signals.persistence_signals import compute_barycenter, wasserstein_distance

@dataclass
class PersistenceEntry:
    diagram: np.ndarray
    timestamp: int

class TopologicalMomentumSignal:
    """
    Generate trading signals from persistence diagram dynamics.
    """
    
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.persistence_history: List[PersistenceEntry] = []
    
    def compute_persistence_landscape(self, 
                                      diagram: np.ndarray,
                                      resolution: int = 100) -> np.ndarray:
        """
        Convert diagram to persistence landscape (stable representation).
        """
        diagram = np.asarray(diagram)
        if diagram.size == 0:
            return np.zeros(resolution, dtype=float)
        
        # ensure shape (n,2)
        if diagram.ndim != 2 or diagram.shape[1] < 2:
            raise ValueError("diagram must be shape (n,2) with birth,death pairs")
        
        # Create landscape functions
        x = np.linspace(0.0, float(np.max(diagram[:, 1])), resolution)
        landscape = np.zeros(resolution, dtype=float)
        
        for birth, death in diagram:
            persistence = float(death) - float(birth)
            mid = (float(birth) + float(death)) / 2.0
            
            # Triangle function for each point
            triangle = np.maximum(0.0, persistence - np.abs(x - mid))
            landscape = np.maximum(landscape, triangle)
        
        return landscape


class TestDynamicGraph(unittest.TestCase):
    
    def setUp(self):
        self.config = GraphConfig(n_heads=4, n_layers=2, hidden_dim=32)
        self.n_assets = 10
        self.graph_net = DynamicGraphNetwork(
            n_assets=self.n_assets,
            feature_dim=5,
            config=self.config
        )
    
    def test_graph_construction(self):
        features = torch.randn(self.n_assets, 5)
        returns = torch.randn(50, self.n_assets)
        
        output = self.graph_net(features, returns)
        
        self.assertIn("adjacencies", output)
        self.assertIn("scale_0.5", output["adjacencies"])
        
        # Check adjacency properties
        adj = output["adjacencies"]["scale_0.5"]
        self.assertEqual(adj.shape, (self.n_assets, self.n_assets))
        self.assertTrue(torch.all(adj >= 0))
    
    def test_correlation_structure(self):
        returns = torch.randn(100, self.n_assets)
        corr = self.graph_net.compute_correlation_structure(returns)
        
        self.assertEqual(corr.shape, (self.n_assets, self.n_assets))
        self.assertTrue(torch.allclose(corr, corr.T, atol=1e-6))  # Symmetric
        self.assertTrue(torch.all(torch.diag(corr) > 0.99))  # Unit diagonal


class TestDirectedComplex(unittest.TestCase):
    
    def test_clique_construction(self):
        # Create simple directed graph
        adj = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=float)
        
        dfc = DirectedFlagComplex(adj)
        simplices = dfc.construct_simplices(max_dimension=2)
        
        # Should have 3 vertices, 3 edges, 1 triangle
        self.assertEqual(len(simplices[0]), 3)
        self.assertEqual(len(simplices[1]), 3)
        self.assertEqual(len(simplices[2]), 1)
        
        # Check triangle orientation
        triangle = simplices[2][0]
        self.assertEqual(triangle.vertices, (0, 1, 2))


class TestPersistence(unittest.TestCase):
    
    def setUp(self):
        self.computer = FastPersistenceComputer(use_ripser=True, max_dim=1)
    
    def test_circle_topology(self):
        # Create points on circle (should have 1-dim homology)
        n_points = 20
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Distance matrix
        from scipy.spatial.distance import pdist, squareform
        dist = squareform(pdist(points))
        
        diagrams = self.computer.compute(dist)
        
        # Should have significant 1-dimensional persistence
        self.assertIn(1, diagrams)
        if len(diagrams[1].persistence_points) > 0:
            pers_1 = diagrams[1].persistence_points[:, 1] - diagrams[1].persistence_points[:, 0]
            self.assertTrue(np.max(pers_1) > 0.5)


class TestWasserstein(unittest.TestCase):
    
    def test_distance_symmetry(self):
        dgm1 = np.array([[0, 1], [0.5, 1.5]])
        dgm2 = np.array([[0.1, 1.1], [0.4, 1.4]])
        
        d12 = wasserstein_distance(dgm1, dgm2, p=2)
        d21 = wasserstein_distance(dgm2, dgm1, p=2)
        
        self.assertAlmostEqual(d12, d21, places=5)
    
    def test_barycenter(self):
        diagrams = [
            np.array([[0, 1], [0.5, 2]]),
            np.array([[0.1, 1.1], [0.6, 2.1]]),
            np.array([[-0.1, 0.9], [0.4, 1.9]])
        ]
        
        bary = compute_barycenter(diagrams, n_iterations=5)
        
        self.assertTrue(len(bary) > 0)
        self.assertEqual(bary.shape[1], 2)


if __name__ == "__main__":
    unittest.main()