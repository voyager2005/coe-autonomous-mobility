import numpy as np
from dataclasses import dataclass
from typing import Literal
from src.tracking.kalman import KalmanBoxTracker  # noqa

@dataclass
class Track:
    id: int
    kalman: KalmanBoxTracker
    embedding: np.ndarray
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: Literal["tentative", "confirmed", "dead"] = "tentative"

    def predict(self):
        """Call kalman.predict(), increment age + time_since_update."""
        self.kalman.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, embedding):
        """
        Update kalman, EMA-smooth embedding (α=0.9), reset time_since_update, 
        promote tentative → confirmed if hits ≥ 2.
        """
        self.kalman.update(bbox)
        # EMA-smooth embedding (α=0.9)
        self.embedding = 0.9 * self.embedding + 0.1 * embedding
        self.time_since_update = 0
        self.hits += 1
        
        if self.state == "tentative" and self.hits >= 2:
            self.state = "confirmed"

    def is_dead(self, max_age: int = 30) -> bool:
        """True if time_since_update > 30 (or max_age)"""
        return self.time_since_update > max_age
