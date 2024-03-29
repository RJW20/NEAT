from abc import ABC
from __future__ import annotations

from NEAT.genome import Genome


class BasePlayer(ABC):
    """Base class for a Player.
    
    Designed to be extended when using this package.
    """

    def __init__(self) -> None:
        self.fitness: float
        self.adjusted_fitness: float
        self.genome: Genome

    def clone(self, player_args: dict) -> BasePlayer:
        """Return a BasePlayer with this BasePlayer's Genome."""

        clone = self.__class__(player_args)
        clone.genome = self.genome.clone()
        return clone