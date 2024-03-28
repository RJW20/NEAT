from NEAT.baseplayer import BasePlayer
from NEAT.population.species import Species
from NEAT.history.history import History


class Population:
    """Contains Species of Players.
    
    Must be initiated with a settings dictionary as found in the documentation.
    """

    def __init__(self, settings: dict) -> None:
        self.generation: int
        self.players: list[BasePlayer]
        self.species: list[Species]
        self.history: list[History]

        # Unload the settings
        try:
            self._excess_coefficient = settings['excess_coefficient']
            self._disjoint_coefficient = settings['disjoint_coefficient']
            self._weight_difference_coefficient = settings['weight_difference_coefficient']
            self._compatibility_threshold = settings['compatibility_threshold']

            self._cull_percentage = settings['cull_percentage']
            self._max_specie_staleness = settings['max_specie_staleness']
        except KeyError as e:
            print(f'Setting {e.args[0]} not found in settings.')

    def speciate(self) -> None:
        """Split the players into Species.
        
        They will be split based on how similar they are to leaders of the previous generation.
        """

        for player in self.players:
            species_found = False
            for specie in self.species:
                if specie.is_same_species(player):
                    specie.players.append(player)
                    species_found = True

            if not species_found:
                new_species = Species(
                    player = player,
                    c1 = self._excess_coefficient,
                    c2 = self._disjoint_coefficient,
                    c3 = self._weight_difference_coefficient,
                    delta = self._compatibility_threshold
                )
                self.species.append(new_species)

    def rank_species(self) -> None:
        """Sort the Species in the Population by their best fitness in descending order."""

        for specie in self.species:
            specie.rank_players()

        self.species.sort(key = lambda specie: specie.best_fitness, reverse=True)

    def fitness_share(self) -> None:
        """Compute the adjusted fitness for each Player in each Species."""

        for specie in self.species:
            specie.fitness_share()

    def cull(self) -> None:
        """Remove all but top percentage of Players in each Species."""

        for specie in self.species:
            specie.cull(self._cull_percentage)

    def remove_stale_species(self) -> None:
        """Remove Species which haven't improved for too many generations."""
        self.species = [specie for specie in self.species if specie.staleness < self._max_specie_staleness]

    def remove_bad_species(self) -> None:
        """Remove Species which are deemed too bad to reproduce.
        
        More specifically remove the Species which will be assigned fewer than one child 
        in the calculation for number of offspring per species.
        """

        total_population_adjusted_fitness = sum([specie.total_adjusted_fitness for specie in self.species])
        self.species =  [specie for specie in self.species if 
                         specie.total_adjusted_fitness / total_population_adjusted_fitness >= 1]

    def repopulate(self) -> None:
        """"""

    def natural_selection(self) -> None:
        """Create the next generation of Players."""

        # Speciate

        # Rank

        # Fitness Share/Calculate adjusted fitnesses

        # Save playback

        # Cull

        # Save parents for reloading

        # Repopulate