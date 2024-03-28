from NEAT.baseplayer import BasePlayer
from NEAT.genome import Genome


class Species:
    """Contains a large group of Players that have similar Genomes.
    
    A Species' rep will always be the Genome of the Player in the Species that obtained the 
    best fitness in the previous generation.
    The staleness of a species counts how many generations have gone without any improvement in the 
    best fitness.
    """

    def __init__(self, player: BasePlayer, **settings: dict) -> None:

        # When creating a new Species the Player will always be the best of its Species 
        # as we speciate in descending fitness order
        self.rep: Genome = player.genome.clone()
        self.players: list[BasePlayer] = list(player)

        self.staleness: int = 0

        # Unload the settings
        self._c1 = settings['c1']
        self._c2 = settings['c2']
        self._c3 = settings['c3']
        self._delta = settings['delta']

    def excess_and_disjoint(self, genome: Genome) -> tuple[int, int]:
        """Return the number of excess and disjoint genes the given Genome has with this 
        Species' rep."""

        # Get self.rep's innovation numbers
        rep_innovation_numbers = self.rep.innovation_numbers
        rep_max_innovation_number = max(rep_innovation_numbers)

        # Count the excess and disjoint connections in the Genome we are testing
        excess, disjoint = 0, 0
        for connection in genome.connections:
            if connection.innovation_number not in rep_innovation_numbers:
                if connection.innovation_number > rep_max_innovation_number:
                    excess += 1
                else:
                    disjoint += 1

        return excess, disjoint

    def average_weight_difference(self, genome: Genome) -> float:
        """Return the average weight difference between this Species' rep and the given 
        Genome."""

        matching, total_difference = 0, .0
        for connection1 in self.rep.connections:
            for connection2 in genome.connections:
                if connection1.innovation_number == connection2.innovation_number:
                    matching += 1
                    total_difference += abs(connection1.weight - connection2.weight)

        return total_difference/matching if matching else 100

    def is_same_species(self, player: BasePlayer) -> bool:
        """Return True if the given player's Genome is considered to be a part of this Species.
        
        This is determined by calculating the compatibility distance from this Species' rep 
        and comparing it to the chosen compatibility threshold.
        """

        excess, disjoint = self.excess_and_disjoint(player.genome)
        average_weight_difference = self.average_weight_difference(player.genome)
        genome_normalizer = max(len(self.rep.connections) - 20, 1)

        compatibility = (self._c1 * excess + self._c2 * disjoint) / genome_normalizer + \
                        self._c3 * average_weight_difference
        
        return compatibility < self._delta
