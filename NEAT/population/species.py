from NEAT.baseplayer import BasePlayer
from NEAT.genome import Genome


class Species:
    """Contains a large group of Players that have similar Genomes."""

    def __init__(self, player: BasePlayer) -> None:
        self.rep: Genome
        self.players: list[BasePlayer]