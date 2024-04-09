import random

from NEAT.base_player import BasePlayer
from NEAT.genome import Genome
from NEAT.history import History
from NEAT.genome.activation_functions import activation_by_name
from NEAT.evolution import fitness_weighted_selection, crossover, mutate


class PlayerFactory:
    """Object that creates new instances of the Population's Players' class and 
    creates/assigns them a Genome through different methods."""

    def __init__(
        self,
        PlayerClass: type,
        player_args: dict,
        genome_settings: dict,
        reproduction_settings: dict,
    ) -> None:
        self._PlayerClass: type = PlayerClass
        self._player_args: dict = player_args
        self._genome_settings: dict = genome_settings

        try:
            self._hidden_activation = activation_by_name(genome_settings['hidden_activation'])
        except KeyError as e:
            raise Exception(f'Setting {e.args[0]} not found in genome_settings.')
        
        try:
            self._crossover_rate = reproduction_settings['reproduction_settings']
            self._disabled_rate = reproduction_settings['disabled_rate']
            self._weights_rate = reproduction_settings['weights_rate']
            self._weight_replacement_rate = reproduction_settings['weight_replacement_rate']
            self._connection_rate = reproduction_settings['connection_rate']
            self._node_rate = reproduction_settings['node_rate']
        except KeyError as e:
            raise Exception(f'Setting {e.args[0]} not found in reproduction_settings.')
        
    def empty_player(self) -> BasePlayer:
        """Return a new Player without a Genome."""
        return self._PlayerClass(self._player_args)
        
    def new_players(self, total: int, history: History) -> list[BasePlayer]:
        """Return a list of length total consisting of Players which have random Genomes."""

        players = [self.empty_player() for _ in range(total)]
        for player in players:
            player.genome = Genome.new(
                input_count = self._genome_settings['input_count'],
                output_count = self._genome_settings['output_count'],
                history = history,
            )

        return players

    def clone(self, player: BasePlayer):
        """Return a new Player with the given Player's Genome."""

        clone = self.empty_player()
        clone.genome = player.genome.clone()
        return clone
    
    def generate_offspring(self, parents: list[BasePlayer], total: int, history: History) -> list[BasePlayer]:
        """Return a list of length total consisting of Players that are the offspring of 
        the given parents."""

        offspring = []

        # Generate the offspring
        while len(offspring) < total:

            # Get a child as either a crossover or a clone
            if random.uniform(0,1) < self._crossover_rate:
                [parent1, parent2] = fitness_weighted_selection(parents, 2)
                if parent1.fitness < parent2.fitness:
                    parent1, parent2 = parent2, parent1
                child = self.empty_player()
                child.genome = crossover(parent1.genome, parent2.genome, self._disabled_rate)
            else:
                [parent] = fitness_weighted_selection(parents, 1)
                child = self.clone(parent)

            # Mutate the child
            mutate(
                genome = child.genome,
                weights_rate = self._weights_rate,
                weight_replacement_rate = self._weight_replacement_rate,
                connection_rate = self._connection_rate,
                node_rate = self._node_rate,
                node_activation = self._hidden_activation,
                history = history
            )
            
        # The method for adding children to offspring (in 1's and 2's) can cause there to be 1 too many
        offspring = offspring[:total]

        return offspring