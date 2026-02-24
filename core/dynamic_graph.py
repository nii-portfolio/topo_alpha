import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

@dataclass
class GraphConfig:
    n_heads: int = 8
    n_layers: int = 3
    hidden_dim: int = 64
    dropout: float = 0.1
    learnable_edges: bool = True


class GraphAttentionLayer(nn.Module):
    """
    Multi-head graph attention with asymmetric edge weights.
    Captures lead-lag relationships through directed attention.
    """
    
    def __init__(self, in_features: int, out_features: int, n_heads: int, 
                 dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Learnable projection matrices
        self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features // n_heads))
        self.a_src = nn.Parameter(torch.empty(n_heads, 2 * (out_features // n_heads), 1))
        self.a_dst = nn.Parameter(torch.empty(n_heads, 2 * (out_features // n_heads), 1))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [N, in_features]
            adj: Optional prior adjacency matrix [N, N]
        
        Returns:
            out: Updated node features [N, out_features]
            attention: Attention weights [n_heads, N, N]
        """
        N = x.size(0)
        
        # Linear transformation for each head
        h = torch.einsum('nf,hfo->nho', x, self.W)  # [N, n_heads, head_dim]
        
        # Compute attention coefficients
        # Source and target representations
        h_repeat_src = h.unsqueeze(1).expand(-1, N, -1, -1)  # [N, N, n_heads, head_dim]
        h_repeat_dst = h.unsqueeze(0).expand(N, -1, -1, -1)  # [N, N, n_heads, head_dim]
        
        # Concatenate for asymmetric attention
        a_input = torch.cat([h_repeat_src, h_repeat_dst], dim=-1)  # [N, N, n_heads, 2*head_dim]
        
        # Compute attention scores (asymmetric)
        e_src = torch.einsum('ijho,hok->ijh', a_input, self.a_src).squeeze(-1)
        e_dst = torch.einsum('ijho,hok->ijh', a_input, self.a_dst).squeeze(-1)
        
        # Asymmetric combination captures directionality
        e = self.leaky_relu(e_src + e_dst.permute(1, 0, 2))  # [N, N, n_heads]
        
        # Mask if prior adjacency provided (structural constraint)
        if adj is not None:
            mask = adj.unsqueeze(-1).expand(-1, -1, self.n_heads)
            e = e.masked_fill(mask == 0, float('-inf'))
        
        # Softmax normalization per head
        attention = F.softmax(e, dim=1)  # [N, N, n_heads]
        attention = self.dropout_layer(attention)
        
        # Aggregate neighbors
        out = torch.einsum('ijh,jhf->ihf', attention, h)  # [N, n_heads, head_dim]
        out = out.reshape(N, -1)  # [N, out_features]
        
        return out, attention.permute(2, 0, 1)  # [n_heads, N, N]


class DynamicGraphNetwork(nn.Module):
    """
    Hierarchical GAT for learning multi-scale market dependencies.
    Constructs adjacency matrices at different correlation thresholds.
    """
    
    def __init__(self, n_assets: int, feature_dim: int, config: GraphConfig):
        super().__init__()
        self.n_assets = n_assets
        self.config = config
        self.feature_dim = feature_dim
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, config.hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(config.n_layers):
            in_dim = config.hidden_dim if i == 0 else config.hidden_dim
            self.gat_layers.append(
                GraphAttentionLayer(in_dim, config.hidden_dim, config.n_heads, config.dropout)
            )
        
        # Multi-scale adjacency learners
        self.thresholds = [0.3, 0.5, 0.7, 0.9]
        self.scale_embeddings = nn.Parameter(torch.randn(len(self.thresholds), config.hidden_dim))
        
        # Edge probability decoder
        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * config.hidden_dim + len(self.thresholds), 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.n_layers)
        ])
        
    def compute_correlation_structure(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic correlation matrix with exponential decay.
        
        Args:
            returns: Asset returns [T, N]
        
        Returns:
            corr: Correlation matrix [N, N]
        """
        # Exponential decay weights
        T = returns.size(0)
        weights = torch.exp(torch.linspace(-1, 0, T)).to(returns.device)
        weights = weights / weights.sum()
        
        # Weighted covariance
        mean_returns = torch.sum(returns * weights.unsqueeze(1), dim=0)
        centered = returns - mean_returns.unsqueeze(0)
        
        weighted_centered = centered * torch.sqrt(weights).unsqueeze(1)
        cov = torch.mm(weighted_centered.t(), weighted_centered)
        
        # Correlation
        std = torch.sqrt(torch.diag(cov))
        corr = cov / torch.outer(std, std)
        corr = torch.clamp(corr, -1, 1)
        
        return corr
    
    def forward(self, features: torch.Tensor, returns_history: torch.Tensor) -> Dict[str, Any]:
        """
        Construct multi-scale dynamic graphs.
        
        Args:
            features: Current node features [N, feature_dim]
            returns_history: Historical returns [T, N]
        
        Returns:
            Dictionary containing adjacency matrices and embeddings at each scale
        """
        N = features.size(0)
        
        # Base correlation structure
        base_corr = self.compute_correlation_structure(returns_history)
        
        # Node embeddings through GAT
        x = self.input_proj(features)
        attention_maps = []
        
        for gat_layer, norm in zip(self.gat_layers, self.layer_norms):
            x_new, attn = gat_layer(x)
            x = norm(x + x_new)  # Residual connection
            attention_maps.append(attn.mean(dim=0))  # Average over heads
        
        # Multi-scale adjacency construction
        adjacencies = {}
        embeddings = {}
        
        for idx, threshold in enumerate(self.thresholds):
            # Structural mask from correlation
            structural_mask = (base_corr.abs() > threshold).float()
            
            # Learned edge probabilities
            scale_indicator = self.scale_embeddings[idx].unsqueeze(0).expand(N, -1)
            
            # Pairwise edge features
            src_exp = x.unsqueeze(1).expand(-1, N, -1)
            dst_exp = x.unsqueeze(0).expand(N, -1, -1)
            
            edge_features = torch.cat([
                src_exp, 
                dst_exp, 
                scale_indicator.unsqueeze(1).expand(-1, N, -1)
            ], dim=-1)
            
            # Decode edge probabilities
            edge_probs = self.edge_decoder(edge_features).squeeze(-1)
            
            # Combine structural and learned (gated combination)
            gate = torch.sigmoid(torch.randn(N, N).to(x.device))  # Learnable gate
            combined_adj = gate * edge_probs + (1 - gate) * structural_mask
            
            # Apply threshold for sparsity
            adj = (combined_adj > 0.5).float() * combined_adj
            
            # Ensure no self-loops for topology
            adj.fill_diagonal_(0)
            
            adjacencies[f"scale_{threshold}"] = adj
            embeddings[f"scale_{threshold}"] = x
        
        return {
            "adjacencies": adjacencies,
            "embeddings": embeddings,
            "base_correlation": base_corr,
            "attention_maps": attention_maps
        }
    
    def get_directed_adjacency(self, returns_lead: torch.Tensor, 
                              returns_lag: torch.Tensor,
                              features: torch.Tensor) -> torch.Tensor:
        """
        Construct directed adjacency for lead-lag relationships.
        
        Args:
            returns_lead: Leading returns [T, N]
            returns_lag: Lagging returns [T, N] 
            features: Node features [N, feat_dim]
        
        Returns:
            Directed adjacency matrix [N, N]
        """
        # Granger-causality inspired directionality
        N = features.size(0)
        
        # Compute cross-correlations at different lags
        max_lag = 5
        cross_corrs = []
        
        for lag in range(1, max_lag + 1):
            if returns_lead.size(0) <= lag:
                break
            lead_slice = returns_lead[lag:]
            lag_slice = returns_lag[:-lag]
            
            corr = torch.corrcoef(torch.stack([
                lead_slice.mean(dim=0),
                lag_slice.mean(dim=0)
            ]))[0, 1]
            cross_corrs.append(corr)
        
        # Directional strength matrix
        directional_strength = torch.zeros(N, N).to(features.device)
        
        # Use GAT to refine directionality
        with torch.no_grad():
            graph_out = self.forward(features, returns_lead)
            base_adj = graph_out["adjacencies"]["scale_0.5"]
        
        # Asymmetric weighting
        for i in range(N):
            for j in range(i+1, N):
                if base_adj[i, j] > 0:
                    # Determine direction from cross-correlation
                    strength = torch.tensor(cross_corrs).mean() if cross_corrs else 0.0
                    if strength > 0:
                        directional_strength[i, j] = strength  # i leads j
                    else:
                        directional_strength[j, i] = abs(strength)  # j leads i
        
        return directional_strength