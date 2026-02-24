import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ResidualSignal:
    """Trading signal based on equilibrium deviation."""
    asset_id: int
    expected_return: float
    confidence: float
    half_life: float
    regime: str


class ResidualAlphaGenerator:
    """
    Generate alpha from deviation between market price and graph equilibrium.
    """
    
    def __init__(self, 
                 mean_reversion_speed: float = 0.1,
                 signal_threshold: float = 1.0):
        self.mean_reversion = mean_reversion_speed
        self.threshold = signal_threshold
        self.equilibrium_history = []
        self.residual_history = []
    
    def compute_residual(self,
                         current_price: float,
                         equilibrium_price: float,
                         uncertainty: float) -> Tuple[float, float]:
        """
        Compute standardized residual.
        
        Returns:
            (raw_residual, z_score)
        """
        raw_residual = current_price - equilibrium_price
        
        # Standardize by uncertainty
        z_score = raw_residual / (uncertainty + 1e-6)
        
        return raw_residual, z_score
    
    def generate_signal(self,
                        asset_id: int,
                        residual: float,
                        z_score: float,
                        regime_stability: float) -> Optional[ResidualSignal]:
        """
        Generate trading signal if residual is significant.
        """
        # Only trade if confidence is sufficient
        if abs(z_score) < self.threshold:
            return None
        
        # Expected return from mean reversion
        expected_return = -self.mean_reversion * residual
        
        # Confidence based on z-score and regime
        confidence = min(abs(z_score) / 3, 1.0) * regime_stability
        
        # Half-life of signal
        half_life = np.log(2) / self.mean_reversion
        
        regime = "stable" if regime_stability > 0.7 else "unstable"
        
        return ResidualSignal(
            asset_id=asset_id,
            expected_return=expected_return,
            confidence=confidence,
            half_life=half_life,
            regime=regime
        )
    
    def batch_generate(self,
                       prices: np.ndarray,
                       equilibrium_prices: np.ndarray,
                       uncertainties: np.ndarray,
                       regime_stability: float) -> List[ResidualSignal]:
        """
        Generate signals for all assets.
        """
        signals = []
        
        for i in range(len(prices)):
            res, z = self.compute_residual(
                prices[i], 
                equilibrium_prices[i], 
                uncertainties[i]
            )
            
            self.residual_history.append(res)
            
            signal = self.generate_signal(i, res, z, regime_stability)
            if signal:
                signals.append(signal)
        
        return signals
    
    def get_signal_decay(self, 
                         signal_age: int,
                         half_life: float) -> float:
        """
        Compute decay factor for aged signals.
        """
        return np.exp(-signal_age / half_life)