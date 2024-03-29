import random

from NEAT.base_player import BasePlayer
from NEAT.genome import Genome
from NEAT.history import History
from NEAT.evolution import fitness_weighted_selection, crossover, mutate


class PlayerFactory:
    """Object that creates new instances of the Population's Players' class and 
    creates/assigns them a Genome through different methods."""

    def __init__(
        self,
        PlayerClass: type,
        player_args: dict,
        genome_args: dict,
        reproduction_settings: dict,
    ) -> None:
        self.history: list[History]
        self.PlayerClass: type = PlayerClass
        self.player_args: dict = player_args
        self.genome_args: dict = genome_args
        
        try:
            self.crossover_rate = reproduction_settings['reproduction_settings']
        except KeyError as e:
            raise Exception(f'Setting {e.args[0]} not found in reproduction_settings.')
        
    def new_players(self, total: int) -> list[BasePlayer]:
        """Return a list of length total consisting of Players which have random Genomes."""
        return [
            Genome.new(
                input_count = self.genome_args['input_count'],
                output_count = self.genome_args['output_count'],
                history = self.history,
            )
            for _ in range(total)
        ]

    def clone(self, player: BasePlayer):
        """Return a new Player with the given Player's Genome."""

        clone = self.PlayerClass(self.player_args)
        clone.genome = player.genome.clone()
        return clone
    
    def generate_offspring(self, parents: list[BasePlayer], total: int) -> list[BasePlayer]:
        """Return a list of length total consisting of Players that are the offspring of 
        the given parents."""

        offspring = []

        # Generate the offspring
        while len(offspring) < total:

            # Get a child as either a clone or a crossover
            if random.uniform(0,1) < self.crossover_rate:
                [parent1, parent2] = fitness_weighted_selection(parents, 2)
                child = crossover(parent1, parent2)
            else:
                [parent] = fitness_weighted_selection(parents, 1)
                child = parent.clone(self.player_args)

            # Mutate the child
            
        # The method for adding children to offspring (in 1's and 2's) can cause there to be 1 too many
        offspring = offspring[:total]

        return offspring