# src/simulation/individual.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import random
import copy

from src.utils import get_logger
from src.neat.neat import NEATGenome, crossover as crossover_genomes
from src.neat.network import NEATNetwork

log = get_logger(__name__)

def crossover_bodies(body1: np.ndarray, body2: np.ndarray) -> np.ndarray:
    """
    Performs uniform crossover on two bodies, ensuring the result is valid.
    """
    for _ in range(10):
        child_body = np.zeros_like(body1)
        for r in range(body1.shape[0]):
            for c in range(body1.shape[1]):
                child_body[r, c] = random.choice([body1[r, c], body2[r, c]])
        
        if BaseIndividual.is_connected(child_body):
            return child_body
    
    return body1.copy()

class BaseIndividual(ABC):
    _id_counter = 0

    def __init__(self, body: np.ndarray, connections=None):
        self.body = body
        self.connections = connections 
        self.fitness: float | None = None
        self.metadata = {}
        self.id = BaseIndividual._id_counter
        BaseIndividual._id_counter += 1

    @abstractmethod
    def copy(self) -> "BaseIndividual":
        pass

    @abstractmethod
    def mutate(self, mutation_rate: float, mutation_amount: float, **controller_kwargs) -> "BaseIndividual":
        pass

    @abstractmethod
    def controller(self, obs):
        pass
    
    def mutate_morphology(self, mutation_rate, mutation_amount):
        """
        Gently mutates robot morphology, allowing for addition, removal, or transformation.
        """
        body_copy = self.body.copy()
        
        for _ in range(10): # Try up to 10 times to find a valid mutation
            temp_body = body_copy.copy()

            # Determine how many voxels to change (usually 1, sometimes 0 or 2)
            # This controls the magnitude of the mutation.
            num_mutations = np.random.poisson(mutation_rate) + 1 
            
            for _ in range(num_mutations):
                ix, iy = random.randint(0, temp_body.shape[0]-1), random.randint(0, temp_body.shape[1]-1)
                current_type = temp_body[ix, iy]

                if current_type == 0:
                    # If empty, always ADD a new voxel.
                    temp_body[ix, iy] = random.choice([1, 2, 3, 4])
                else:
                    # If not empty, 50% chance to REMOVE, 50% chance to TRANSFORM.
                    if random.random() < 0.5:
                        temp_body[ix, iy] = 0 # Removal
                    else:
                        # Transformation to a different, non-empty type.
                        possible_types = [t for t in [1, 2, 3, 4] if t != current_type]
                        if possible_types:
                           temp_body[ix, iy] = random.choice(possible_types)
            
            # Only accept the change if the body remains connected.
            if self.is_connected(temp_body):
                return temp_body, None
        
        return body_copy, self.connections

    @staticmethod
    def is_connected(body):
        if np.sum(body > 0) == 0:
            return False
        
        positions = np.argwhere(body > 0)
        if len(positions) == 0: return False
        
        visited, to_visit = set(), [tuple(positions[0])]
        
        while to_visit:
            pos = to_visit.pop()
            if pos in visited: continue
            visited.add(pos)
            
            i, j = pos
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < body.shape[0] and 0 <= nj < body.shape[1]:
                    if body[ni, nj] > 0 and (ni, nj) not in visited:
                        to_visit.append((ni, nj))
        
        return len(visited) == len(positions)

class RandomIndividual(BaseIndividual):
    def __init__(self, body: np.ndarray, connections=None, env_name: str = "Walker-v0"):
        super().__init__(body, connections)
        self.env_name = env_name
        self._action_space = None

    def controller(self, obs):
        if self._action_space is None:
            raise RuntimeError(f"RandomIndividual-{self.id}: action_space not set")
        return self._action_space.sample()

    def copy(self) -> "RandomIndividual":
        return RandomIndividual(self.body.copy(), self.connections, self.env_name)

    def mutate(self, mutation_rate: float = 0.10, mutation_amount: float = 0.30, **__) -> "RandomIndividual":
        new_body, new_conn = self.mutate_morphology(mutation_rate, mutation_amount)
        return RandomIndividual(new_body, new_conn, self.env_name)

    def set_action_space(self, space):
        self._action_space = space

class NEATIndividual(BaseIndividual):
    def __init__(self, body: np.ndarray, connections=None, genome: NEATGenome | None = None, neat_config: dict | None = None, **__):
        super().__init__(body, connections)
        self.neat_config = neat_config or {}

        if genome is not None:
            self.genome = genome
        else:
            self.genome = NEATGenome.create_for_morphology(body)

        self._network = NEATNetwork(self.genome)

    def copy(self) -> "NEATIndividual":
        return NEATIndividual(self.body.copy(), self.connections, self.genome.copy(), neat_config=copy.deepcopy(self.neat_config))

    def mutate(self, mutation_rate: float, mutation_amount: float, neat_config: dict | None = None, **__) -> "NEATIndividual":
        cfg = neat_config or self.neat_config
        new_body, new_conn = self.mutate_morphology(mutation_rate, mutation_amount)
        new_genome = self.genome.copy()

        num_new_sensory_inputs = np.sum(new_body > 0) * 2
        num_new_outputs = np.sum((new_body == 3) | (new_body == 4))
        new_genome.adapt_io(num_new_sensory_inputs, num_new_outputs)

        if cfg:
            new_genome.mutate(cfg)
        
        return NEATIndividual(new_body, new_conn, new_genome, neat_config=cfg)

    def crossover(self, other_parent: "NEATIndividual") -> "NEATIndividual":
        if self.fitness > other_parent.fitness:
            fitter_parent = self
        else:
            fitter_parent = other_parent
        child_body = fitter_parent.body.copy()

        child_genome = crossover_genomes(self.genome, other_parent.genome)
        
        num_child_sensory_inputs = np.sum(child_body > 0) * 2
        num_child_outputs = np.sum((child_body == 3) | (child_body == 4))
        child_genome.adapt_io(num_child_sensory_inputs, num_child_outputs)
        
        return NEATIndividual(child_body, connections=None, genome=child_genome, neat_config=self.neat_config)

    def controller(self, obs):
        features = np.asarray(obs, dtype=float).flatten()
        
        if len(features) < self.genome.num_sensory_inputs:
            features = np.pad(features, (0, self.genome.num_sensory_inputs - len(features)))
        elif len(features) > self.genome.num_sensory_inputs:
            features = features[:self.genome.num_sensory_inputs]
        
        output = self._network.activate(features)
        
        output = output * 3.0
        output = np.clip(output, -1.0, 1.0)
        
        expected_actions = np.sum((self.body == 3) | (self.body == 4))
        if len(output) != expected_actions:
            final_output = np.zeros(expected_actions)
            limit = min(len(output), expected_actions)
            final_output[:limit] = output[:limit]
            return final_output
        
        return output