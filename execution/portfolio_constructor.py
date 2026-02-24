import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
import torch
import torch.nn as nn
from scipy.optimize import minimize

if TYPE_CHECKING:
    from topo_alpha.features import TopologicalRiskFeatures  # pyright: ignore[reportMissingImports]


@dataclass
class ExecutionCosts:
    """Transaction cost model."""
    base_bps: float = 5.0  # Base cost in basis points
    market_impact_coef: float = 0.1  # Impact coefficient
    spread_bps: float = 2.0  # Half-spread
    
    def estimate_cost(self, 
                      trade_size: float,
                      avg_daily_volume: float,
                      volatility: float) -> float:
        """
        Estimate total transaction cost in basis points.
        
        Args:
            trade_size: Dollar trade size
            avg_daily_volume: ADV in dollars
            volatility: Annualized volatility
        
        Returns:
            Cost in basis points
        """
        # Market impact (square root model)
        participation = trade_size / (avg_daily_volume + 1e-8)
        impact = self.market_impact_coef * volatility * np.sqrt(participation) * 10000
        
        # Spread cost
        spread_cost = self.spread_bps
        
        # Total
        total = self.base_bps + impact + spread_cost
        
        return total


class PortfolioConstructor:
    """
    Constructs dollar-neutral, beta-neutral portfolios.
    Models constraint slippage and execution costs.
    """
    
    def __init__(self,
                 target_portfolio_beta: float = 0.0,
                 max_position_size: float = 0.1,
                 max_turnover: float = 0.5):
        self.target_beta = target_portfolio_beta
        self.max_position = max_position_size
        self.max_turnover = max_turnover
        self.cost_model = ExecutionCosts()
        self.previous_weights = None
    
    def construct_portfolio(self,
                            signals: np.ndarray,
                            betas: np.ndarray,
                            beta_uncertainties: np.ndarray,
                            prices: np.ndarray,
                            volumes: np.ndarray,
                            current_weights: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Construct optimal portfolio with neutrality constraints.
        
        Args:
            signals: Alpha signals [N]
            betas: Asset betas [N]
            beta_uncertainties: Beta std [N]
            prices: Current prices [N]
            volumes: Average volumes [N]
            current_weights: Current portfolio weights [N]
        
        Returns:
            Dictionary with optimal weights and diagnostics
        """
        n_assets = len(signals)
        
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        
        # Objective: maximize signal exposure minus costs
        def objective(w):
            # Expected return from signals
            alpha_return = -np.dot(w, signals)  # Negative for minimization
            
            # Transaction costs
            trades = w - current_weights
            costs = np.array([
                self.cost_model.estimate_cost(
                    abs(trades[i]) * prices[i] * 1e6,  # Assume $1M portfolio
                    volumes[i],
                    0.2  # Assume 20% vol
                ) / 10000  # Convert to decimal
                for i in range(n_assets)
            ])
            total_cost = np.sum(np.abs(trades) * costs)
            
            # Beta neutrality penalty (soft constraint)
            port_beta = np.dot(w, betas)
            beta_penalty = 100 * (port_beta - self.target_beta) ** 2
            
            # Position size penalty
            size_penalty = np.sum(np.maximum(0, np.abs(w) - self.max_position) ** 2)
            
            return alpha_return + total_cost + beta_penalty + 10 * size_penalty
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w)},  # Dollar neutral
        ]
        
        # Probabilistic beta constraint
        def beta_constraint(w):
            port_beta = np.dot(w, betas)
            port_beta_var = np.dot(w ** 2, beta_uncertainties ** 2)
            # P(|beta| < 0.1) > 0.95 approximation
            return 0.1 - abs(port_beta) - 2 * np.sqrt(port_beta_var)
        
        constraints.append({'type': 'ineq', 'fun': beta_constraint})
        
        # Bounds
        bounds = [(-self.max_position, self.max_position) for _ in range(n_assets)]
        
        # Initial guess
        w0 = current_weights.copy()
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        optimal_weights = result.x
        
        # Compute diagnostics
        trades = optimal_weights - current_weights
        turnover = np.sum(np.abs(trades)) / 2
        
        diagnostics = {
            "optimal_weights": optimal_weights,
            "trades": trades,
            "turnover": turnover,
            "portfolio_beta": np.dot(optimal_weights, betas),
            "portfolio_beta_risk": np.dot(optimal_weights ** 2, beta_uncertainties ** 2),
            "expected_alpha": np.dot(optimal_weights, signals),
            "execution_costs": objective(optimal_weights) - (-np.dot(optimal_weights, signals)),
            "constraint_violation": result.maxcv if hasattr(result, 'maxcv') else 0
        }
        
        self.previous_weights = optimal_weights
        
        return diagnostics
    
    def build_risk_parity_overlay(self,
                                   base_weights: np.ndarray,
                                   risk_features: Optional['TopologicalRiskFeatures'] = None) -> np.ndarray:
        """
        Adjust weights based on topological risk features.
        """
        # Return base weights if risk features are not available
        if risk_features is None:
            return base_weights
        
        # Reduce exposure when Betti_1 is high (many loops)
        loop_factor = 1.0 / (1.0 + risk_features.betti_1 / 10)
        
        # Adjust positions
        adjusted = base_weights * loop_factor
        
        # Re-normalize to dollar neutral
        adjusted = adjusted - adjusted.mean()
        
        return adjusted