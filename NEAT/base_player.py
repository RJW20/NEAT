from abc import ABC

from NEAT.genome import Genome


class BasePlayer(ABC):
    """Base class for a Player.
    
    Designed to be extended when using this package.
    """

    def __init__(self) -> None:
        self.fitness: float
        self.adjusted_fitness: float
        self.genome: Genome