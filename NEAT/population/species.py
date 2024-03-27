from NEAT.baseplayer import BasePlayer
from NEAT.genome import Genome


class Species:
    """Contains a large group of Players that have similar Genomes."""

    def __init__(self, player: BasePlayer, **settings: dict) -> None:
        self.rep: Genome
        self.players: list[BasePlayer]

        # Unload the settings
        self._c1 = settings['c1']
        self._c2 = settings['c2']
        self._c3 = settings['c3']
        self._delta = settings['delta']

    def same_species(self, genome: Genome) -> bool:
        """Return True if the given Genome is considered to be a part of this Species.
        
        This is determined by calculating the compatibility distance from this Species' rep 
        and comparing it to the chosen compatibility threshold.
        """

