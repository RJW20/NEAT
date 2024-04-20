from __future__ import annotations
from pathlib import Path
import pickle

from neat.base_player import BasePlayer
from neat.genome import Genome


class Species:
    """Contains a large group of Players that have similar Genomes.
    
    A Species' rep will always be the Genome of the Player in the Species that obtained the 
    best fitness in the previous generations.
    The staleness of a species counts how many generations have gone without any improvement in the 
    best fitness.
    """

    def __init__(self, player: BasePlayer, settings: dict) -> None:

        # When creating a new Species the given Player will always be the only option
        # for a rep 
        self.rep: Genome = player.genome.clone()
        self.players: list[BasePlayer] = list(player)

        self.staleness: int = 0
        self.best_fitness: int = 0

        # Unload the settings
        try: 
            self._c1 = settings['excess_coefficient']
            self._c2 = settings['disjoint_coefficient']
            self._c3 = settings['weight_difference_coefficient']
            self._delta = settings['compatibility_threshold']
            self._max_staleness = settings['max_staleness']
        except KeyError as e:
            raise Exception(f'Setting \'{e.args[0]}\' not found in species_settings.')

    @property
    def champ(self) -> BasePlayer:
        """Return the Player in this Species with the highest fitness.
        
        Should only be called after the Players have been ranked.
        """
        return self.players[0]
    
    @property
    def size(self) -> int:
        """Return the number of Players in this Species."""
        return len(self.players)
    
    @property
    def total_adjusted_fitness(self) -> float:
        """Return the total adjusted fitness of Players in this Species."""
        return sum([player.adjusted_fitness for player in self.players]) if self.size else .0
    
    @property
    def gone_stale(self) -> bool:
        """Return True if no improvements have been made for too many generations."""
        return self.staleness >= self._max_staleness

    def excess_and_disjoint(self, genome: Genome) -> tuple[int, int]:
        """Return the number of excess and disjoint Connections the given Genome has with this 
        Species' rep."""

        excess, disjoint = 0, 0
        rep_max_innovation_number = max(self.rep.innovation_numbers)
        for connection in genome.connections:
            if connection.innovation_number not in self.rep.innovation_numbers:
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
                    break

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
    
    def rank_players(self) -> None:
        """Sort the Players in the Species by fitness in descending order."""
        self.players.sort(key = lambda player: player.fitness, reverse=True)

    def check_progress(self) -> None:
        """Check if this Species is improving."""

        if self.champ.fitness > self.best_fitness:
            self.staleness = 0
            self.best_fitness = self.champ.fitness
            self.rep = self.champ.genome.clone()
        else:
            self.staleness += 1

    def fitness_share(self) -> None:
        """Compute the adjusted fitness for each Player in this Species."""
        
        specie_size = self.size
        for player in self.players:
            player.adjusted_fitness = player.fitness / specie_size

    def cull(self, cull_percentage: float) -> None:
        """Remove the bottom cull_percentage of Players in this Species."""

        remaining = max(int((1 - cull_percentage) * self.size), 1)
        self.players = self.players[:remaining]

    def save(self, destination: Path) -> None:
        """Save this Species in the given destination with pickle.
        
        If a Species save already exists in the same destination it will be overwritten.
        Fails if this Species has a non-empty players list.
        """

        if self.players:
            raise Exception('Do not attempt to save a Species between speciation and creating \
                            the next generation (when the Species\' player list is non-empty).')

        with destination.open('wb') as dest:
            pickle.dump(self, dest)

    @classmethod
    def load(cls, source: Path) -> Species:
        """Return the Species saved in the given source.
        
        Throws an OSError if fails to open the save.
        """

        with source.open('rb') as src:
            return pickle.load(src)