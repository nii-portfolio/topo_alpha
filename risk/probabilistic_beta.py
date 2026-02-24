import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
from scipy import stats


@dataclass
class BetaDistribution:
    """Beta as a random variable with uncertainty."""
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    sample_path: Optional[np.ndarray] = None


class StochasticBetaModel:
    """
    Models beta as a stochastic process rather than point estimate.
    Uses Bayesian updates and regime-dependent volatility.
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 max_beta_deviation: float = 0.1):
        self.confidence = confidence_level
        self.max_deviation = max_beta_deviation
        self.beta_history = []
        self.uncertainty_history = []
        self.regime_specific_params = {}
    
    def estimate_beta_distribution(self,
                                   asset_returns: np.ndarray,
                                   market_returns: np.ndarray,
                                   regime: Optional[int] = None) -> BetaDistribution:
        """
        Estimate beta with uncertainty using Bayesian rolling regression.
        
        Args:
            asset_returns: [T] asset returns
            market_returns: [T] market returns
            regime: Optional regime identifier for regime-specific estimation
        
        Returns:
            BetaDistribution with uncertainty quantification
        """
        # Rolling window regression with uncertainty
        window = min(60, len(asset_returns))
        
        if len(asset_returns) < window:
            # Insufficient data
            return BetaDistribution(
                mean=1.0,
                std=0.5,
                confidence_interval=(0.0, 2.0)
            )
        
        # Use last window
        y = asset_returns[-window:]
        x = market_returns[-window:]
        
        # Add intercept
        X = np.vstack([np.ones(window), x]).T
        
        # OLS estimation
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        beta_market = beta_hat[1]
        
        # Residuals and standard error
        residuals = y - X @ beta_hat
        mse = np.sum(residuals**2) / (window - 2)
        
        # Standard error of beta
        x_var = np.var(x)
        if x_var > 0:
            beta_std = np.sqrt(mse / (window * x_var))
        else:
            beta_std = 0.5
        
        # Confidence interval
        alpha = 1 - self.confidence
        t_crit = stats.t.ppf(1 - alpha/2, window - 2)
        ci_lower = beta_market - t_crit * beta_std
        ci_upper = beta_market + t_crit * beta_std
        
        # Regime adjustment
        if regime is not None and regime in self.regime_specific_params:
            regime_adj = self.regime_specific_params[regime]
            beta_std *= regime_adj["uncertainty_multiplier"]
        
        dist = BetaDistribution(
            mean=beta_market,
            std=beta_std,
            confidence_interval=(ci_lower, ci_upper)
        )
        
        self.beta_history.append(beta_market)
        self.uncertainty_history.append(beta_std)
        
        return dist
    
    def check_neutrality_constraint(self,
                                    portfolio_beta_dist: BetaDistribution) -> Tuple[bool, float]:
        """
        Check if portfolio satisfies probabilistic neutrality constraint.
        
        Args:
            portfolio_beta_dist: Distribution of portfolio beta
        
        Returns:
            (satisfied, penalty)
        """
        # Check if confidence interval contains 0
        ci_lower, ci_upper = portfolio_beta_dist.confidence_interval
        
        # Probabilistic constraint: P(|beta| < epsilon) > confidence
        # Approximate using normal distribution
        prob_within_bounds = stats.norm.cdf(
            self.max_deviation,
            loc=abs(portfolio_beta_dist.mean),
            scale=portfolio_beta_dist.std
        ) - stats.norm.cdf(
            -self.max_deviation,
            loc=abs(portfolio_beta_dist.mean),
            scale=portfolio_beta_dist.std
        )
        
        satisfied = (prob_within_bounds >= self.confidence).item()
        
        # Compute constraint violation penalty
        if satisfied:
            penalty = 0.0
        else:
            # Distance from satisfaction
            excess = abs(portfolio_beta_dist.mean) - self.max_deviation
            penalty = excess ** 2 + portfolio_beta_dist.std ** 2
        
        return satisfied, penalty
    
    def optimize_neutral_weights(self,
                                  betas: List[BetaDistribution],
                                  target_return: Optional[float] = None) -> np.ndarray:
        """
        Optimize portfolio weights for stochastic neutrality.
        
        Args:
            betas: List of beta distributions for each asset
            target_return: Optional target return
        
        Returns:
            Optimal weights [N]
        """
        n_assets = len(betas)
        
        # Mean-variance optimization with probabilistic constraints
        means = np.array([b.mean for b in betas])
        stds = np.array([b.std for b in betas])
        
        # Covariance of betas (simplified diagonal)
        beta_cov = np.diag(stds ** 2)
        
        # Objective: minimize portfolio beta variance + deviation from zero
        # min w' * beta_cov * w + lambda * (w' * means)^2
        # s.t. sum(w) = 0 (dollar neutral)
        
        # Use quadratic programming approximation
        # For simplicity, use iterative approach
        
        weights = np.zeros(n_assets)
        
        # Sort by beta magnitude
        beta_order = np.argsort(np.abs(means))
        
        # Pair long and short to cancel beta
        long_idx = beta_order[means[beta_order] < 0]
        short_idx = beta_order[means[beta_order] > 0]
        
        # Allocate to most negative beta (long) and most positive beta (short)
        allocation = 1.0 / min(len(long_idx), len(short_idx), n_assets // 2)
        
        for i in range(min(len(long_idx), len(short_idx))):
            weights[long_idx[i]] = allocation
            weights[short_idx[i]] = -allocation
        
        # Normalize to dollar neutral
        weights = weights - weights.mean()
        
        # Verify constraint
        port_beta_mean = np.dot(weights, means)
        port_beta_var = weights @ beta_cov @ weights
        
        return weights


class BetaNeutralConstraintLayer(nn.Module):
    """
    Neural network layer enforcing soft beta neutrality.
    """
    
    def __init__(self, n_assets: int, max_beta: float = 0.1):
        super().__init__()
        self.n_assets = n_assets
        self.max_beta = max_beta
        
        # Learnable risk budget allocation
        self.risk_budget = nn.Parameter(torch.ones(n_assets) / n_assets)
    
    def forward(self, 
                raw_weights: torch.Tensor,
                betas: torch.Tensor,
                beta_uncertainties: torch.Tensor) -> torch.Tensor:
        """
        Project weights onto beta-neutral manifold.
        
        Args:
            raw_weights: Unconstrained weights [N]
            betas: Expected betas [N]
            beta_uncertainties: Beta standard deviations [N]
        
        Returns:
            Constrained weights [N]
        """
        # Soft constraint using Lagrangian relaxation
        # min ||w - w_raw||^2 + lambda * (w' * beta)^2
        
        # Normalize input
        w = raw_weights - raw_weights.mean()  # Dollar neutral
        
        # Compute current portfolio beta
        port_beta = torch.dot(w, betas)
        
        # Correction to reduce beta exposure
        if abs(port_beta) > self.max_beta:
            # Find adjustment
            beta_sensitivity = betas / (beta_uncertainties + 1e-6)
            adjustment = -port_beta * beta_sensitivity / torch.sum(beta_sensitivity ** 2)
            w = w + adjustment
        
        # Final normalization
        w = w / (torch.abs(w).sum() + 1e-8)
        
        return w