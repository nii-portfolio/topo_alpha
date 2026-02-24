import numpy as np
import torch
import yaml
from typing import Dict, List
import argparse
import logging

# Core imports
from core.dynamic_graph import DynamicGraphNetwork, GraphConfig
from core.directed_complex import DirectedFlagComplex, HierarchicalRegimeDetector
from core.persistence_engine import FastPersistenceComputer, RealTimePersistenceTracker
from core.topological_gnn import TopologicalFusionNetwork

# Signal imports
from signals.equilibrium import GraphEquilibriumEstimator
from signals.persistence_signals import PersistenceEarlyWarningSystem, TopologicalMomentumSignal
from signals.regime_detection import HierarchicalRegimeDetector as RegimeDetector

# Risk imports
from risk.probabilistic_beta import StochasticBetaModel, BetaNeutralConstraintLayer
from risk.betti_features import TopologicalRiskModel, BettiRiskEncoder

# Execution imports
from execution.portfolio_constructor import PortfolioConstructor
from execution.residual_signals import ResidualAlphaGenerator, ResidualSignal

# Utils
from utils.caching import PersistenceCache, IncrementalTopologyUpdater


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopologicalAlphaSystem:
    """
    Main system integrating all components.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.n_assets = 100  # Will be set dynamically
        
        # Initialize components
        self._init_graph_constructor()
        self._init_persistence_engine()
        self._init_gnn_fusion()
        self._init_signals()
        self._init_risk_model()
        self._init_execution()
        
        # State
        self.current_regime = None
        self.persistence_cache = PersistenceCache()
        self.incremental_updater = IncrementalTopologyUpdater()
        
        logger.info("TopologicalAlphaSystem initialized")
    
    def _init_graph_constructor(self):
        """Initialize dynamic graph construction."""
        graph_cfg = GraphConfig(
            n_heads=self.config['graph']['gat_heads'],
            n_layers=self.config['graph']['gat_layers'],
            hidden_dim=self.config['graph']['hidden_dim'],
            dropout=self.config['graph']['dropout']
        )
        self.graph_net = DynamicGraphNetwork(
            n_assets=self.n_assets,
            feature_dim=10,  # OHLCV + features
            config=graph_cfg
        )
    
    def _init_persistence_engine(self):
        """Initialize persistence computation."""
        self.persistence_computer = FastPersistenceComputer(
            use_ripser=True,
            max_dim=max(self.config['topology']['persistence_dimensions'])
        )
        self.persistence_tracker = RealTimePersistenceTracker(
            n_assets=self.n_assets,
            max_dim=1
        )
    
    def _init_gnn_fusion(self):
        """Initialize topological GNN."""
        self.topo_gnn = TopologicalFusionNetwork(
            n_assets=self.n_assets,
            node_feature_dim=10,
            max_homology_dim=1
        )
    
    def _init_signals(self):
        """Initialize signal generators."""
        self.equilibrium_estimator = GraphEquilibriumEstimator(
            n_assets=self.n_assets,
            feature_dim=10
        )
        self.early_warning = PersistenceEarlyWarningSystem(max_dim=1)
        self.momentum_signal = TopologicalMomentumSignal(lookback=10)
        self.residual_generator = ResidualAlphaGenerator()
        self.regime_detector = RegimeDetector(
            correlation_thresholds=self.config['topology']['correlation_thresholds'],
            lookback_windows=self.config['regime']['lookback_windows']
        )
    
    def _init_risk_model(self):
        """Initialize risk management."""
        self.beta_model = StochasticBetaModel(
            confidence_level=self.config['risk']['beta_confidence'],
            max_beta_deviation=self.config['risk']['max_beta_deviation']
        )
        self.risk_model = TopologicalRiskModel()
        self.betti_encoder = BettiRiskEncoder()
    
    def _init_execution(self):
        """Initialize execution engine."""
        self.portfolio_constructor = PortfolioConstructor(
            target_portfolio_beta=self.config['risk']['target_portfolio_beta'],
            max_position_size=self.config['execution']['max_position_size']
        )
    
    def process_market_data(self, 
                           returns: np.ndarray,
                           features: np.ndarray,
                           prices: np.ndarray,
                           volumes: np.ndarray) -> Dict:
        """
        Main processing pipeline.
        
        Args:
            returns: [T, N] return series
            features: [N, F] current features
            prices: [N] current prices
            volumes: [N] average volumes
        
        Returns:
            Trading decisions and diagnostics
        """
        # 1. Dynamic Graph Construction
        graph_output = self.graph_net(
            torch.from_numpy(features).float(),
            torch.from_numpy(returns).float()
        )
        
        adjacencies = graph_output["adjacencies"]
        embeddings = graph_output["embeddings"]
        
        # 2. Persistence Computation (multi-scale)
        persistence_diagrams = {}
        for scale_name, adj in adjacencies.items():
            # Check cache
            cached = self.persistence_cache.get(adj.numpy())
            
            if cached is not None:
                persistence_diagrams[scale_name] = cached
            else:
                # Compute persistence
                dist_matrix = 1.0 - np.abs(adj.numpy())
                np.fill_diagonal(dist_matrix, 0)
                
                diagrams = self.persistence_computer.compute(dist_matrix)
                
                # Convert to numpy
                diagrams_np = {
                    dim: dgm.persistence_points 
                    for dim, dgm in diagrams.items()
                }
                
                persistence_diagrams[scale_name] = diagrams_np
                self.persistence_cache.put(adj.numpy(), diagrams_np)
        
        # 3. Topological GNN Fusion
        gnn_output = self.topo_gnn(
            torch.from_numpy(features).float(),
            adjacencies,
            persistence_diagrams
        )
        
        # 4. Equilibrium Estimation
        equilibrium_output = self.equilibrium_estimator(
            torch.from_numpy(features).float(),
            adjacencies["scale_0.5"],
            torch.from_numpy(prices).float()
        )
        
        # 5. Regime Detection
        regime_state = self.regime_detector.detect_regime(
            returns,
            persistence_diagrams
        )
        self.current_regime = regime_state
        
        # 6. Early Warning System
        warnings = self.early_warning.update(
            persistence_diagrams.get("scale_0.5", {})
        )
        
        # 7. Residual Signals
        signals = self.residual_generator.batch_generate(
            prices,
            equilibrium_output["equilibrium_prices"].numpy(),
            equilibrium_output["uncertainty"].numpy(),
            regime_state.stability_score
        )
        
        # Convert signals to array
        signal_array = np.zeros(self.n_assets)
        confidence_array = np.zeros(self.n_assets)
        for sig in signals:
            signal_array[sig.asset_id] = sig.expected_return
            confidence_array[sig.asset_id] = sig.confidence
        
        # 8. Risk Features
        risk_features = self.risk_model.compute_features(
            persistence_diagrams.get("scale_0.5", {}),
            self.n_assets
        )
        
        # 9. Beta Estimation
        # Assume market is first asset or external index
        market_returns = returns[:, 0]  # Simplified
        beta_dists = []
        for i in range(self.n_assets):
            beta_dist = self.beta_model.estimate_beta_distribution(
                returns[:, i],
                market_returns,
                regime=regime_state.regime_id
            )
            beta_dists.append(beta_dist)
        
        betas = np.array([b.mean for b in beta_dists])
        beta_uncs = np.array([b.std for b in beta_dists])
        
        # 10. Portfolio Construction
        portfolio = self.portfolio_constructor.construct_portfolio(
            signals=signal_array,
            betas=betas,
            beta_uncertainties=beta_uncs,
            prices=prices,
            volumes=volumes,
            current_weights=None  # Or previous weights
        )
        
        # Apply topological risk overlay
        adjusted_weights = self.portfolio_constructor.build_risk_parity_overlay(
            portfolio["optimal_weights"],
            risk_features
        )
        
        return {
            "weights": adjusted_weights,
            "raw_weights": portfolio["optimal_weights"],
            "trades": portfolio["trades"],
            "turnover": portfolio["turnover"],
            "portfolio_beta": portfolio["portfolio_beta"],
            "expected_alpha": portfolio["expected_alpha"],
            "regime": regime_state,
            "warnings": warnings,
            "risk_features": risk_features,
            "equilibrium_convergence": equilibrium_output["convergence_info"],
            "topological_stability": self.early_warning.get_stability_score()
        }
    
    def run_backtest(self, 
                     data: Dict[str, np.ndarray],
                     rebalance_freq: int = 1) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            data: Dictionary with 'returns', 'features', 'prices', 'volumes'
            rebalance_freq: Rebalance every N periods
        
        Returns:
            Backtest results
        """
        T = len(data['returns'])
        results = []
        
        for t in range(100, T, rebalance_freq):  # Start after warmup
            # Use last 100 observations
            ret_window = data['returns'][t-100:t]
            feat = data['features'][t]
            price = data['prices'][t]
            vol = data['volumes'][t]
            
            decision = self.process_market_data(ret_window, feat, price, vol)
            results.append(decision)
            
            if t % 100 == 0:
                logger.info(f"Processed {t}/{T} time steps")
        
        return {
            "decisions": results,
            "final_regime": self.current_regime,
            "computation_stats": self.persistence_computer.get_stats()
        }


def generate_synthetic_data(n_assets: int = 50, 
                           n_periods: int = 1000) -> Dict[str, np.ndarray]:
    """Generate synthetic market data for testing."""
    np.random.seed(42)
    
    # Generate correlated returns
    factor_loadings = np.random.randn(n_assets, 3)
    factors = np.random.randn(n_periods, 3) * 0.02
    
    idiosyncratic = np.random.randn(n_periods, n_assets) * 0.01
    returns = factors @ factor_loadings.T + idiosyncratic
    
    # Features (momentum, volatility, etc.)
    features = np.random.randn(n_periods, n_assets, 10)
    
    # Prices (random walk)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    # Volumes
    volumes = np.random.lognormal(15, 0.5, (n_periods, n_assets))
    
    return {
        "returns": returns,
        "features": features,
        "prices": prices,
        "volumes": volumes
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topological Market-Neutral Strategy")
    parser.add_argument("--mode", choices=["live", "backtest"], default="backtest")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    
    # Initialize system
    system = TopologicalAlphaSystem(args.config)
    
    if args.mode == "backtest":
        logger.info("Running backtest...")
        data = generate_synthetic_data(n_assets=50, n_periods=500)
        
        # Adjust system to actual data size
        system.n_assets = 50
        
        results = system.run_backtest(data)
        
        logger.info("Backtest complete")
        logger.info(f"Final portfolio beta: {results['decisions'][-1]['portfolio_beta']:.4f}")
        logger.info(f"Computation stats: {results['computation_stats']}")
    else:
        logger.info("Live mode not implemented in this demo")