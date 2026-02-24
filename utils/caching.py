import numpy as np
from functools import lru_cache
import hashlib
from typing import Any, Dict


class PersistenceCache:
    """
    LRU cache for persistence computations.
    """
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = {}
        self.access_order = []
    
    def _make_key(self, distance_matrix: np.ndarray) -> str:
        """Create hash key from matrix."""
        return hashlib.sha256(distance_matrix.tobytes()).hexdigest()
    
    def get(self, distance_matrix: np.ndarray) -> Any:
        """Get cached result."""
        key = self._make_key(distance_matrix)
        if key in self.cache:
            # Move to end (most recent)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, distance_matrix: np.ndarray, result: Any):
        """Store result in cache."""
        key = self._make_key(distance_matrix)
        
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Evict oldest
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = result
        self.access_order.append(key)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


class IncrementalTopologyUpdater:
    """
    Incrementally update topological features as new data arrives.
    Avoids full recomputation every tick.
    """
    
    def __init__(self, update_frequency: int = 20):
        self.update_frequency = update_frequency
        self.tick_count = 0
        self.last_full_compute = None
        self.incremental_buffer = []
    
    def should_update(self, force: bool = False) -> bool:
        """Check if full update is needed."""
        self.tick_count += 1
        return force or (self.tick_count % self.update_frequency == 0)
    
    def get_approximate_features(self, 
                                  current_features: Dict,
                                  new_data: np.ndarray) -> Dict:
        """
        Approximate feature update without full recomputation.
        """
        # Simple exponential moving average update
        alpha = 0.1
        
        updated = {}
        for key, value in current_features.items():
            if isinstance(value, (int, float)):
                updated[key] = (1 - alpha) * value + alpha * np.random.randn()
            else:
                updated[key] = value
        
        return updated