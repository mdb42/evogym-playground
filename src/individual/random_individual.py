# src/individual/random_individual.py
from __future__ import annotations
import numpy as np
from src.individual.base import BaseIndividual, log


class RandomIndividual(BaseIndividual):
    def __init__(
        self,
        body: np.ndarray,
        connections=None,
        env_name: str = "Walker-v0",
    ):
        super().__init__(body, connections)
        self.env_name = env_name
        self._action_space = None

    def controller(self, obs):
        if self._action_space is None:
            raise RuntimeError(
                f"RandomIndividual-{self.id}: action_space not set before first call"
            )
        return self._action_space.sample()

    def copy(self) -> "RandomIndividual":
        return RandomIndividual(self.body.copy(), self.connections, self.env_name)

    def mutate(
        self,
        mutation_rate: float = 0.10,
        mutation_amount: float = 0.30,
        **__,
    ) -> "RandomIndividual":
        new_body, new_conn = self.mutate_morphology(mutation_rate, mutation_amount)
        return RandomIndividual(new_body, new_conn, self.env_name)

    def set_action_space(self, space):
        self._action_space = space
        log.debug("RandomIndividual-%d received action-space %s", self.id, space)
