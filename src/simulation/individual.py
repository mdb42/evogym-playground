# evogym-playground/src/simulation/individual.py
"""
Individual class representing a robot in EvolutionGym.
"""

import numpy as np
import random

class Individual:
    def __init__(self, body, connections=None, controller=None):
        self.body = body
        self.connections = connections
        self.controller = controller
        self.fitness = None
        self.metadata = {}
    
    def is_valid(self):
        """Check if morphology is valid (connected)"""
        return Individual.is_connected(self.body)
    
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
    
    def mutate(self, mutation_rate=0.1, mutation_amount=0.3):
        """Mutate this individual, return new Individual"""
        new_body, new_connections = self._mutate_morphology(
            self.body, self.connections, mutation_rate, mutation_amount
        )
        # TODO: also mutate controller
        return Individual(new_body, new_connections, self.controller)
    
    def _mutate_morphology(self, body, connections, mutation_rate, mutation_amount):
        """Simple mutation of robot morphology"""
        body_copy = body.copy()
        
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
                                temp_body[i, j] = random.choice([1, 2, 3, 4]) # Add new voxel
                        else:  # Non-empty
                            if random.random() < mutation_amount:
                                temp_body[i, j] = 0
                            else:
                                temp_body[i, j] = random.choice([1, 2, 3, 4]) # Change voxel type
            
            # Check if result is valid
            if Individual.is_connected(temp_body):
                return temp_body, None  # Let EvoGym handle connections
        
        # If not valid, return original
        return body.copy(), connections
    
    def copy(self):
        """Deep copy of individual"""
        return Individual(
            self.body.copy(), 
            self.connections,
            self.controller  # TODO: Will need deep copy
        )