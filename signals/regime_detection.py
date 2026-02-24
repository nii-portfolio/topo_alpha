import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import torch
import torch.nn as nn


@dataclass
class RegimeState:
    """Current market regime characterization."""
    regime_id: int
    confidence: float
    betti_features: np.ndarray
    persistence_entropy: float
    duration: int
    stability_score: float


class HierarchicalRegimeDetector:
    """
    Detect market regimes across multiple time scales and correlation thresholds.
    Uses Betti numbers and persistence features for regime classification.
    """
    
    def __init__(self, 
                 correlation_thresholds: List[float] = [0.3, 0.5, 0.7, 0.9],
                 lookback_windows: List[int] = [20, 60, 120]):
        self.thresholds = correlation_thresholds
        self.windows = lookback_windows
        self.regime_history = deque(maxlen=1000)
        self.current_regime = None
        self.regime_counts = {i: 0 for i in range(8)}  # 8 possible regimes
        
        # Regime classifier (simple MLP)
        self.feature_dim = len(correlation_thresholds) * 4  # 4 features per scale
        self.classifier = self._build_classifier()
    
    def _build_classifier(self) -> nn.Module:
        """Build simple regime classifier."""
        return nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 8),  # 8 regime classes
            nn.Softmax(dim=-1)
        )
    
    def extract_betti_features(self, 
                               persistence_diagrams: Dict[str, Dict[int, np.ndarray]]) -> np.ndarray:
        """
        Extract Betti number features from multi-scale persistence.
        
        Args:
            persistence_diagrams: Dict[scale, Dict[dim, diagram]]
        
        Returns:
            Feature vector [n_features]
        """
        features = []
        
        for scale in sorted(persistence_diagrams.keys()):
            diagrams = persistence_diagrams[scale]
            
            # Betti numbers (count of persistent features)
            betti_0 = len(diagrams.get(0, []))
            betti_1 = len(diagrams.get(1, []))
            
            # Persistence statistics
            if len(diagrams.get(0, [])) > 0:
                dgm_0 = diagrams[0]
                pers_0 = dgm_0[:, 1] - dgm_0[:, 0]
                avg_pers_0 = np.mean(pers_0)
                max_pers_0 = np.max(pers_0)
            else:
                avg_pers_0 = 0
                max_pers_0 = 0
            
            features.extend([betti_0, betti_1, avg_pers_0, max_pers_0])
        
        return np.array(features, dtype=np.float32)
    
    def detect_regime(self,
                      returns: np.ndarray,
                      multi_scale_diagrams: Dict[str, Dict[int, np.ndarray]]) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            returns: Recent returns [T, N]
            multi_scale_diagrams: Persistence diagrams at multiple scales
        
        Returns:
            RegimeState object
        """
        # Extract features
        features = self.extract_betti_features(multi_scale_diagrams)
        
        # Classify regime
        with torch.no_grad():
            logits = self.classifier(torch.from_numpy(features).unsqueeze(0))
            probs = logits.squeeze().numpy()
            regime_id = int(np.argmax(probs))
            confidence = float(probs[regime_id])
        
        # Compute persistence entropy as stability measure
        entropies = []
        for scale, diagrams in multi_scale_diagrams.items():
            for dim, dgm in diagrams.items():
                if len(dgm) > 0:
                    pers = dgm[:, 1] - dgm[:, 0]
                    pers = pers[pers > 0]
                    if len(pers) > 0:
                        p = pers / pers.sum()
                        entropy = -np.sum(p * np.log(p + 1e-10))
                        entropies.append(entropy)
        
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        
        # Update regime tracking
        if self.current_regime is None or self.current_regime.regime_id != regime_id:
            duration = 1
        else:
            duration = self.current_regime.duration + 1
        
        # Stability based on confidence and entropy
        stability = confidence * np.exp(-avg_entropy / 10)
        
        regime_state = RegimeState(
            regime_id=regime_id,
            confidence=confidence,
            betti_features=features,
            persistence_entropy=avg_entropy,
            duration=duration,
            stability_score=stability
        )
        
        self.current_regime = regime_state
        self.regime_history.append(regime_state)
        self.regime_counts[regime_id] += 1
        
        return regime_state
    
    def get_regime_transition_matrix(self) -> np.ndarray:
        """Compute empirical regime transition probabilities."""
        if len(self.regime_history) < 2:
            return np.eye(8)
        
        transitions = np.zeros((8, 8))
        for i in range(len(self.regime_history) - 1):
            from_regime = self.regime_history[i].regime_id
            to_regime = self.regime_history[i + 1].regime_id
            transitions[from_regime, to_regime] += 1
        
        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions = np.divide(transitions, row_sums, 
                               where=row_sums!=0, 
                               out=np.zeros_like(transitions))
        
        return transitions
    
    def predict_regime_duration(self) -> float:
        """Predict expected remaining duration of current regime."""
        if self.current_regime is None:
            return 0.0
        
        regime_id = self.current_regime.regime_id
        
        # Historical average duration for this regime
        durations = [r.duration for r in self.regime_history 
                    if r.regime_id == regime_id]
        
        if not durations:
            return 10.0  # Default
        
        avg_duration = np.mean(durations)
        current_duration = self.current_regime.duration
        
        # Expected remaining time
        return float(max(0.0, avg_duration - current_duration))