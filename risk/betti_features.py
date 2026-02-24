import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class TopologicalRiskFeatures:
    """Container for topological risk metrics."""
    betti_0: int  # Number of connected components
    betti_1: int  # Number of loops
    euler_characteristic: int
    persistence_entropy: float
    total_persistence: float
    max_persistence_ratio: float
    feature_vector: np.ndarray


class BettiRiskEncoder(nn.Module):
    """
    Encode Betti numbers and topological features for risk model.
    """
    
    def __init__(self, max_betti: int = 100):
        super().__init__()
        self.max_betti = max_betti
        
        # Embedding for Betti numbers
        self.betti_embed = nn.Embedding(max_betti, 16)
        
        # MLP for continuous features
        self.continuous_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Combined processing
        self.combiner = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, risk_features: TopologicalRiskFeatures) -> torch.Tensor:
        """
        Encode topological risk features.
        
        Args:
            risk_features: Computed risk features
        
        Returns:
            Encoded vector [32]
        """
        # Embed discrete Betti numbers
        b0_emb = self.betti_embed(
            torch.clamp(torch.tensor(risk_features.betti_0), 0, self.max_betti - 1)
        )
        b1_emb = self.betti_embed(
            torch.clamp(torch.tensor(risk_features.betti_1), 0, self.max_betti - 1)
        )
        
        # Encode continuous features
        continuous = torch.tensor([
            risk_features.persistence_entropy,
            risk_features.total_persistence,
            risk_features.max_persistence_ratio,
            float(risk_features.euler_characteristic)
        ], dtype=torch.float32)
        
        cont_emb = self.continuous_encoder(continuous)
        
        # Combine
        combined = torch.cat([b0_emb + b1_emb, cont_emb], dim=-1)
        output = self.combiner(combined)
        
        return output


class TopologicalRiskModel:
    """
    Risk model using Betti numbers and persistence features.
    """
    
    def __init__(self):
        self.feature_history = []
        self.risk_regime_map = {}
    
    def compute_features(self, 
                        persistence_diagrams: Dict[int, np.ndarray],
                        n_assets: int) -> TopologicalRiskFeatures:
        """
        Compute risk features from persistence diagrams.
        
        Args:
            persistence_diagrams: Dict[dim, diagram]
            n_assets: Number of assets
        
        Returns:
            TopologicalRiskFeatures
        """
        # Betti numbers
        betti_0 = len(persistence_diagrams.get(0, []))
        betti_1 = len(persistence_diagrams.get(1, []))
        
        # Euler characteristic
        euler = betti_0 - betti_1
        
        # Persistence statistics
        all_persistence = []
        for dim, dgm in persistence_diagrams.items():
            if len(dgm) > 0:
                pers = dgm[:, 1] - dgm[:, 0]
                all_persistence.extend(pers.tolist())
        
        if all_persistence:
            total_pers = sum(all_persistence)
            max_pers = max(all_persistence)
            pers_array = np.array(all_persistence)
            pers_array = pers_array[pers_array > 0]
            
            if len(pers_array) > 0:
                p = pers_array / pers_array.sum()
                entropy = -np.sum(p * np.log(p + 1e-10))
            else:
                entropy = 0.0
            
            max_ratio = max_pers / (total_pers + 1e-8)
        else:
            total_pers = 0.0
            entropy = 0.0
            max_ratio = 0.0
        
        # Feature vector
        feature_vector = np.array([
            betti_0 / n_assets,  # Normalized
            betti_1 / n_assets,
            euler,
            entropy,
            np.log(1 + total_pers),
            max_ratio
        ])
        
        features = TopologicalRiskFeatures(
            betti_0=betti_0,
            betti_1=betti_1,
            euler_characteristic=euler,
            persistence_entropy=entropy,
            total_persistence=total_pers,
            max_persistence_ratio=max_ratio,
            feature_vector=feature_vector
        )
        
        self.feature_history.append(features)
        
        return features
    
    def estimate_tail_risk(self, features: TopologicalRiskFeatures) -> float:
        """
        Estimate tail risk from topological features.
        High Betti_1 (loops) indicates complex interconnected risk.
        """
        # Heuristic: more loops = more systemic risk
        loop_risk = 1 - np.exp(-features.betti_1 / 10)
        
        # Low entropy = concentrated risk
        concentration_risk = 1 / (1 + features.persistence_entropy)
        
        # Combined
        tail_risk = 0.6 * loop_risk + 0.4 * concentration_risk
        
        return tail_risk
    
    def get_risk_budget(self, 
                        features: TopologicalRiskFeatures,
                        base_risk: float = 0.1) -> float:
        """
        Adjust risk budget based on topological conditions.
        """
        tail_risk = self.estimate_tail_risk(features)
        
        # Reduce exposure in high loop regimes
        adjustment = 1.0 - 0.5 * tail_risk
        
        return base_risk * adjustment