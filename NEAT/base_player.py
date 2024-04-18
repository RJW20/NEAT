from abc import ABC, abstractmethod
from typing import Any
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

    @abstractmethod
    def look(self) -> None:
        """Update the attributes used as input to the Genome."""
        pass

    @abstractmethod
    def think(self) -> Any:
        """Feed the input into the Genome and return the output as a valid move."""
        pass

    @abstractmethod
    def move(self, move: Any) -> None:
        """Advance to the state achieved by carrying out move."""
        pass