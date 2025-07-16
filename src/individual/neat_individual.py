# src/individual/neat_individual.py
from __future__ import annotations
import numpy as np
import copy
from src.individual.base import BaseIndividual, log, crossover_bodies
from src.neat.neat import NEATGenome
from src.neat.network import NEATNetwork
from src.neat.crossover import crossover as crossover_genomes

class NEATIndividual(BaseIndividual):
    def __init__(
        self,
        body: np.ndarray,
        connections=None,
        genome: NEATGenome | None = None,
        neat_config: dict | None = None,
        env_name: str | None = None,
        **__,
    ):
        super().__init__(body, connections)
        self.neat_config = neat_config or {}

        # Build / store genome
        if genome is not None:
            self.genome = genome
        else:
            self.genome = NEATGenome.create_for_morphology(body)

        # Compile genome to network once
        self._network = NEATNetwork(self.genome)

    def copy(self) -> "NEATIndividual":
        return NEATIndividual(
            self.body.copy(),
            self.connections,
            copy.deepcopy(self.genome),
            neat_config=copy.deepcopy(self.neat_config),
        )

    def mutate(self, mutation_rate: float = 0.1, mutation_amount: float = 0.3,
               neat_config: dict | None = None, **__) -> "NEATIndividual":
        cfg = neat_config or self.neat_config

        new_body, new_conn = self.mutate_morphology(mutation_rate, mutation_amount)
        new_genome = self.genome.copy()

        # Adapt genome to new body structure
        new_actuators = np.sum((new_body == 3) | (new_body == 4))
        new_genome.adapt_io(new_genome.num_inputs, new_actuators)

        # And mutate
        if cfg:
            new_genome.mutate(cfg)
        
        return NEATIndividual(new_body, new_conn, new_genome, neat_config=cfg)

    def crossover(self, other_parent: "NEATIndividual") -> "NEATIndividual":
        # Crossover bodies
        child_body = crossover_bodies(self.body, other_parent.body)

        # Crossover genomes, preserving fitter parent's topology
        child_genome = crossover_genomes(self.genome, other_parent.genome)
        
        # Adapt new genome to new body
        num_new_actuators = np.sum((child_body == 3) | (child_body == 4))
        child_genome.adapt_io(child_genome.num_inputs, num_new_actuators)
        
        return NEATIndividual(
            child_body,
            connections=None,
            genome=child_genome,
            neat_config=self.neat_config
        )

    def controller(self, obs):
        x = np.asarray(obs, dtype=float).flatten()
        
        # Use first N values
        features = x[:self.genome.num_inputs]
        if len(features) < self.genome.num_inputs:
            features = np.pad(features, (0, self.genome.num_inputs - len(features)))
        
        output = self._network.activate(features)
        
        # Scale up - tanh outputs are too small
        output = output * 3.0
        
        # Clip to valid action range
        output = np.clip(output, -1.0, 1.0)
        
        # Verify output size matches what env expects
        expected_actions = np.sum((self.body == 3) | (self.body == 4))
        if len(output) != expected_actions:
            log.error(f"NEAT-{self.id}: Network output size {len(output)} doesn't match "
                    f"expected {expected_actions} actuators!")
            if len(output) < expected_actions:
                output = np.pad(output, (0, expected_actions - len(output)))
            else:
                output = output[:expected_actions]
        
        return output