import random

from neat.base_player import BasePlayer
from neat.genome import Genome
from neat.history import History
from neat.genome.activation_functions import activation_by_name
from neat.evolution import fitness_weighted_selection, crossover, mutate


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
        self.player_args: dict = player_args

        try:
            self._genome_input_count = genome_settings['input_count']
            self._genome_output_count = genome_settings['output_count']
            self._hidden_activation = activation_by_name(genome_settings['hidden_activation'])
        except KeyError as e:
            raise Exception(f'Setting \'{e.args[0]}\' not found in genome_settings.')
        
        try:
            self._crossover_rate = reproduction_settings['crossover_rate']
            self._disabled_rate = reproduction_settings['disabled_rate']
            self._weights_rate = reproduction_settings['weights_rate']
            self._weight_replacement_rate = reproduction_settings['weight_replacement_rate']
            self._connection_rate = reproduction_settings['connection_rate']
            self._node_rate = reproduction_settings['node_rate']
        except KeyError as e:
            raise Exception(f'Setting \'{e.args[0]}\' not found in reproduction_settings.')
        
    @property
    def genome_settings(self) -> dict:
        """Recollect the attributes that make up the genome_settings dictionary."""

        genome_settings = {
            'input_count': self._genome_input_count,
            'output_count': self._genome_output_count,
            'hidden_activation': self._hidden_activation.__name__,
        }
        return genome_settings
    
    @property
    def reproduction_settings(self) -> dict:
        """Recollect the attributes that make up the reproduction_settings dictionary."""

        reproduction_settings = {
            'crossover_rate': self._crossover_rate,
            'disabled_rate': self._disabled_rate,
            'weights_rate': self._weights_rate,
            'weight_replacement_rate': self._weight_replacement_rate,
            'connection_rate': self._connection_rate,
            'node_rate': self._node_rate,
        }
        return reproduction_settings
        
    def empty_player(self) -> BasePlayer:
        """Return a new Player without a Genome."""
        return self._PlayerClass(self.player_args)
        
    def new_players(self, total: int, history: History) -> list[BasePlayer]:
        """Return a list of length total consisting of Players which have random Genomes."""

        players = [self.empty_player() for _ in range(total)]
        for player in players:
            player.genome = Genome.new(
                input_count = self._genome_input_count,
                output_count = self._genome_output_count,
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

            offspring.append(child)

        return offspring