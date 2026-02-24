import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TopologicalFeatures:
    """Container for topological features from persistence."""
    betti_numbers: torch.Tensor  # [max_dim + 1]
    persistence_images: torch.Tensor  # [max_dim, resolution, resolution]
    persistence_statistics: torch.Tensor  # [n_stats]
    diagram_embeddings: torch.Tensor  # [embedding_dim]


class PersistenceImageEncoder(nn.Module):
    """
    Encode persistence diagrams as images for CNN processing.
    """
    
    def __init__(self, resolution: int = 20, 
                 sigma: float = 0.1,
                 max_persistence: float = 2.0):
        super().__init__()
        self.resolution = resolution
        self.sigma = sigma
        self.max_persistence = max_persistence
        
        # CNN for processing persistence images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Calculate flattened dimension
        self.flat_dim = 32 * (resolution // 4) * (resolution // 4)
        self.fc = nn.Linear(self.flat_dim, 64)
    
    def diagram_to_image(self, 
                         birth_death_pairs: np.ndarray,
                         weight_fn: Optional[Callable] = None) -> torch.Tensor:
        """
        Convert persistence diagram to persistence image.
        
        Args:
            birth_death_pairs: [n_points, 2] array of (birth, death)
            weight_fn: Optional weighting function
        
        Returns:
            Persistence image tensor [resolution, resolution]
        """
        if len(birth_death_pairs) == 0:
            return torch.zeros(self.resolution, self.resolution)
        
        # Transform to birth-persistence coordinates
        births = birth_death_pairs[:, 0]
        deaths = birth_death_pairs[:, 1]
        persistences = deaths - births
        
        # Filter by max persistence
        mask = persistences < self.max_persistence
        births = births[mask]
        persistences = persistences[mask]
        
        if len(births) == 0:
            return torch.zeros(self.resolution, self.resolution)
        
        # Create grid
        img = np.zeros((self.resolution, self.resolution))
        
        # Gaussian kernel for each point
        x_coords = np.linspace(0, self.max_persistence, self.resolution)
        y_coords = np.linspace(0, self.max_persistence, self.resolution)
        XX, YY = np.meshgrid(x_coords, y_coords)
        
        for b, p in zip(births, persistences):
            # Weight by persistence
            weight = p if weight_fn is None else weight_fn(p)
            
            # Gaussian centered at (b, p)
            gaussian = np.exp(-((XX - b)**2 + (YY - p)**2) / (2 * self.sigma**2))
            img += weight * gaussian
        
        return torch.from_numpy(img).float()
    
    def forward(self, persistence_diagrams: Dict[int, np.ndarray]) -> torch.Tensor:
        """
        Encode multiple persistence diagrams.
        
        Args:
            persistence_diagrams: Dict mapping dimension to birth_death arrays
        
        Returns:
            Encoded topological features [n_dims * 64]
        """
        embeddings = []
        
        for dim in sorted(persistence_diagrams.keys()):
            dgm = persistence_diagrams[dim]
            img = self.diagram_to_image(dgm)
            
            # Add batch and channel dims
            img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # CNN encoding
            conv_out = self.conv_layers(img)
            emb = self.fc(conv_out)
            embeddings.append(emb.squeeze())
        
        return torch.cat(embeddings) if embeddings else torch.zeros(64)


class TopologicalGNNLayer(nn.Module):
    """
    GNN layer that incorporates topological features into message passing.
    """
    
    def __init__(self, node_dim: int, topo_dim: int, out_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.topo_dim = topo_dim
        
        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + topo_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Topological attention
        self.topo_gate = nn.Linear(topo_dim, 1)
    
    def forward(self, 
                node_features: torch.Tensor,
                adjacency: torch.Tensor,
                topological_context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with topological modulation.
        
        Args:
            node_features: [N, node_dim]
            adjacency: [N, N] adjacency matrix
            topological_context: [topo_dim] global topological features
        
        Returns:
            Updated node features [N, out_dim]
        """
        N = node_features.size(0)
        
        # Expand topological context to all nodes
        topo_expanded = topological_context.unsqueeze(0).expand(N, -1)  # [N, topo_dim]
        
        # Compute messages for each edge
        src_exp = node_features.unsqueeze(1).expand(-1, N, -1)  # [N, N, node_dim]
        dst_exp = node_features.unsqueeze(0).expand(N, -1, -1)  # [N, N, node_dim]
        
        # Concatenate features
        edge_features = torch.cat([
            src_exp, 
            dst_exp,
            topo_expanded.unsqueeze(1).expand(-1, N, -1)
        ], dim=-1)  # [N, N, 2*node_dim + topo_dim]
        
        # Compute messages
        messages = self.message_mlp(edge_features)  # [N, N, out_dim]
        
        # Weight by adjacency and topological gate
        topo_weight = torch.sigmoid(self.topo_gate(topo_expanded))  # [N, 1]
        weights = adjacency.unsqueeze(-1) * topo_weight.unsqueeze(1)
        
        # Aggregate messages
        aggregated = (messages * weights).sum(dim=1)  # [N, out_dim]
        
        # Update
        combined = torch.cat([node_features, aggregated], dim=-1)
        updated = self.update_mlp(combined)
        
        return updated


class TopologicalFusionNetwork(nn.Module):
    """
    Main network fusing GNN with topological features.
    """
    
    def __init__(self, 
                 n_assets: int,
                 node_feature_dim: int,
                 max_homology_dim: int = 1):
        super().__init__()
        self.n_assets = n_assets
        self.max_homology_dim = max_homology_dim
        
        # Persistence encoder
        self.persistence_encoder = PersistenceImageEncoder(resolution=20)
        
        # Calculate topological feature dimension
        self.topo_dim = (max_homology_dim + 1) * 64
        
        # Node embedding
        self.node_embed = nn.Linear(node_feature_dim, 64)
        
        # Topological GNN layers
        self.gnn_layers = nn.ModuleList([
            TopologicalGNNLayer(64 if i == 0 else 128, self.topo_dim, 128)
            for i in range(3)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(128) for _ in range(3)
        ])
        
        # Output heads
        self.equilibrium_head = nn.Linear(128, 1)  # For equilibrium estimation
        self.regime_head = nn.Linear(128 + self.topo_dim, 4)  # Regime classification
    
    def encode_topology(self, 
                        persistence_diagrams: Dict[int, Dict[int, np.ndarray]]) -> torch.Tensor:
        """
        Encode persistence diagrams from multiple scales.
        
        Args:
            persistence_diagrams: Dict[scale, Dict[dim, diagram]]
        
        Returns:
            Encoded topological features [topo_dim]
        """
        scale_embeddings = []
        
        for scale_name, diagrams in persistence_diagrams.items():
            emb = self.persistence_encoder(diagrams)
            scale_embeddings.append(emb)
        
        # Average over scales
        if scale_embeddings:
            return torch.stack(scale_embeddings).mean(dim=0)
        else:
            return torch.zeros(self.topo_dim)
    
    def forward(self,
                node_features: torch.Tensor,
                adjacency_matrices: Dict[str, torch.Tensor],
                persistence_diagrams: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass fusing graph and topological information.
        
        Args:
            node_features: [N, node_feature_dim]
            adjacency_matrices: Dict of adjacency matrices per scale
            persistence_diagrams: Dict of persistence diagrams per scale
        
        Returns:
            Dictionary with equilibrium estimates and regime predictions
        """
        # Encode topology
        topo_features = self.encode_topology(persistence_diagrams) # type: ignore
        
        # Initial node embedding
        x = self.node_embed(node_features)
        
        # Use middle scale adjacency (0.5 threshold) for message passing
        adj = adjacency_matrices.get("scale_0.5", list(adjacency_matrices.values())[0])
        
        # GNN layers with topological context
        for gnn_layer, norm in zip(self.gnn_layers, self.layer_norms):
            x_new = gnn_layer(x, adj, topo_features)
            x = norm(x + x_new)  # Residual
        
        # Equilibrium estimation (deviation from this = signal)
        equilibrium = self.equilibrium_head(x).squeeze(-1)  # [N]
        
        # Regime classification (global features)
        global_features = torch.cat([
            x.mean(dim=0),  # Average node features
            topo_features
        ])
        regime_logits = self.regime_head(global_features)
        
        return {
            "equilibrium": equilibrium,
            "regime_logits": regime_logits,
            "node_embeddings": x,
            "topological_features": topo_features
        }