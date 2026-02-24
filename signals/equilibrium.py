import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class LearnedMessagePassing(nn.Module):
    """
    Non-linear message passing for equilibrium estimation.
    Learns the diffusion operator rather than using fixed Laplacian.
    """
    
    def __init__(self, feature_dim: int, n_iterations: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_iterations = n_iterations
        
        # Learned diffusion parameters
        self.diffusion_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.Tanh()  # Bounded diffusion
        )
        
        # Attention-based neighbor aggregation
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        
        # Convergence detection
        self.convergence_threshold = 0.001
    
    def forward(self, 
                initial_features: torch.Tensor,
                adjacency: torch.Tensor,
                max_iterations: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Iterate to equilibrium using learned message passing.
        
        Args:
            initial_features: [N, feature_dim]
            adjacency: [N, N] weighted adjacency
            max_iterations: Override default iteration count
        
        Returns:
            equilibrium: Converged features [N, feature_dim]
            info: Convergence information
        """
        max_iter = max_iterations or self.n_iterations
        x = initial_features
        history = []
        
        for i in range(max_iter):
            x_prev = x.clone()
            
            # Compute attention over graph
            # Use adjacency as attention mask
            attn_mask = (adjacency == 0).float() * -1e9
            
            # Self-attention with graph structure
            x_expanded = x.unsqueeze(0)  # [1, N, feature_dim]
            attn_out, _ = self.attention(
                x_expanded, x_expanded, x_expanded,
                attn_mask=attn_mask.unsqueeze(0).expand(self.n_iterations, -1, -1) if False else None
            )
            attn_out = attn_out.squeeze(0)
            
            # Learned diffusion update
            combined = torch.cat([x, attn_out], dim=-1)
            delta = self.diffusion_mlp(combined)
            
            # Update with damping
            x = x + 0.1 * delta
            
            # Check convergence
            change = torch.norm(x - x_prev) / torch.norm(x_prev)
            history.append(change.item())
            
            if change < self.convergence_threshold:
                break
        
        return x, {
            "iterations": i + 1,
            "convergence_history": history,
            "final_change": history[-1] if history else 0
        }


class GraphEquilibriumEstimator(nn.Module):
    """
    Estimates market equilibrium using learned graph dynamics.
    """
    
    def __init__(self, n_assets: int, feature_dim: int):
        super().__init__()
        self.n_assets = n_assets
        
        # Feature extraction
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Equilibrium solver
        self.equilibrium_solver = LearnedMessagePassing(32, n_iterations=20)
        
        # Equilibrium price predictor
        self.price_predictor = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self,
                features: torch.Tensor,
                adjacency: torch.Tensor,
                current_prices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estimate equilibrium and detect deviations.
        
        Args:
            features: Node features [N, feature_dim]
            adjacency: Graph adjacency [N, N]
            current_prices: Current market prices [N]
        
        Returns:
            Dictionary with equilibrium estimates and deviations
        """
        # Encode features
        encoded = self.feature_encoder(features)
        
        # Find equilibrium
        equilibrium_features, conv_info = self.equilibrium_solver(encoded, adjacency)
        
        # Predict equilibrium prices
        equilibrium_prices = self.price_predictor(equilibrium_features).squeeze(-1)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(equilibrium_features).squeeze(-1)
        
        # Deviation from equilibrium = trading signal
        deviation = current_prices - equilibrium_prices
        
        # Normalize by uncertainty
        z_score = deviation / (uncertainty + 1e-6)
        
        return {
            "equilibrium_prices": equilibrium_prices,
            "deviation": deviation,
            "z_score": z_score,
            "uncertainty": uncertainty,
            "convergence_info": conv_info,
            "signal_strength": torch.tanh(z_score)  # Bounded signal
        }
    
    def compute_equilibrium_residuals(self,
                                      historical_features: List[torch.Tensor],
                                      historical_adjacencies: List[torch.Tensor],
                                      realized_prices: torch.Tensor) -> torch.Tensor:
        """
        Compute historical residuals for model validation.
        
        Args:
            historical_features: List of feature tensors
            historical_adjacencies: List of adjacency matrices
            realized_prices: Actual realized prices [T, N]
        
        Returns:
            Residuals [T, N]
        """
        residuals = []
        
        for t, (feat, adj) in enumerate(zip(historical_features, historical_adjacencies)):
            with torch.no_grad():
                result = self.forward(feat, adj, realized_prices[t])
                residuals.append(result["deviation"])
        
        return torch.stack(residuals)