# src/individual/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import random
from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseIndividual(ABC):
    _id_counter = 0

    def __init__(self, body: np.ndarray, connections=None):
        self.body = body
        self.connections = connections 
        self.fitness: float | None = None
        self.metadata = {}
        self.id = BaseIndividual._id_counter
        BaseIndividual._id_counter += 1
        # log.debug(f"Spawned {self.__class__.__name__}-{self.id} with {np.sum(body > 0)} voxels")

    @abstractmethod
    def copy(self) -> "BaseIndividual":
        """Create a deep copy of this individual"""
        pass

    @abstractmethod
    def mutate(self, mutation_rate: float, mutation_amount: float, **controller_kwargs) -> "BaseIndividual":
        """Return a mutated copy of this individual"""
        pass

    @abstractmethod
    def controller(self, obs):
        """The control policy - maps observations to actions"""
        pass

    def mutate_morphology(self, mutation_rate, mutation_amount):
        """Mutate robot morphology maintaining connectivity"""
        body_copy = self.body.copy()
        
        # Try mutation up to 10 times
        for attempt in range(10):
            temp_body = body_copy.copy()
            
            # Mutate some voxels
            for i in range(temp_body.shape[0]):
                for j in range(temp_body.shape[1]):
                    if random.random() < mutation_rate:
                        current = temp_body[i, j]
                        if current == 0:  # Empty
                            if random.random() < mutation_amount:
                                temp_body[i, j] = random.choice([1, 2, 3, 4])
                        else:  # Non-empty
                            if random.random() < mutation_amount:
                                temp_body[i, j] = 0
                            else:
                                temp_body[i, j] = random.choice([1, 2, 3, 4])
            
            # Check if result is valid
            if self.is_connected(temp_body):
                old_voxels = np.sum(self.body > 0)
                new_voxels = np.sum(temp_body > 0)
                log.debug(f"{self.__class__.__name__}-{self.id} morphology: {old_voxels}→{new_voxels} voxels")
                return temp_body, None
        
        # If not valid, return original
        log.debug(f"{self.__class__.__name__}-{self.id} morphology mutation failed, keeping original")
        return body_copy, self.connections

    @staticmethod
    def is_connected(body):
        """Check if all non-empty voxels in body are connected"""
        if np.sum(body > 0) == 0:
            return False
        
        # Find all non-empty positions
        positions = np.argwhere(body > 0)
        if len(positions) == 0:
            return False
        
        # Flood fill from first non-empty position
        visited = set()
        to_visit = [tuple(positions[0])]
        
        while to_visit:
            pos = to_visit.pop()
            if pos in visited:
                continue
            visited.add(pos)
            
            # Check 4-connected neighbors
            i, j = pos
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < body.shape[0] and 0 <= nj < body.shape[1]:
                    if body[ni, nj] > 0 and (ni, nj) not in visited:
                        to_visit.append((ni, nj))
        
        # Check if all non-empty positions were visited
        return len(visited) == len(positions)

# Helper function to perform uniform crossover on two bodies
def crossover_bodies(body1: np.ndarray, body2: np.ndarray) -> np.ndarray:
    """
    Performs uniform crossover on two bodies, ensuring the result is valid.
    """
    for _ in range(10): # Try a few times to get a valid crossover
        child_body = np.zeros_like(body1)
        for r in range(body1.shape[0]):
            for c in range(body1.shape[1]):
                child_body[r, c] = random.choice([body1[r, c], body2[r, c]])
        
        if BaseIndividual.is_connected(child_body):
            return child_body
    
    # If crossover fails repeatedly, return a copy of the first parent
    return body1.copy()