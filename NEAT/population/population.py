from NEAT.base_player import BasePlayer
from NEAT.population.species import Species
from NEAT.history.history import History


class Population:
    """Contains Species of Players."""

    def __init__(
        self,
        PlayerClass: type,
        player_args: dict,
        NEAT_settings: dict,
        playback_settings: dict,
    ) -> None:
        self.generation: int
        self.players: list[BasePlayer]
        self.species: list[Species]
        self.history: list[History]

        self.staleness: int = 0
        self.best_fitness: int = 0

        # Unload the settings
        try:
            self._PlayerClass = PlayerClass
            self._player_args = player_args

            self._size = NEAT_settings['population_size']

            self._excess_coefficient = NEAT_settings['excess_coefficient']
            self._disjoint_coefficient = NEAT_settings['disjoint_coefficient']
            self._weight_difference_coefficient = NEAT_settings['weight_difference_coefficient']
            self._compatibility_threshold = NEAT_settings['compatibility_threshold']

            self._cull_percentage = NEAT_settings['cull_percentage']
            self._max_specie_staleness = NEAT_settings['max_specie_staleness']

            self._max_staleness = NEAT_settings['max_population_staleness']

        except KeyError as e:
            print(f'Setting {e.args[0]} not found in NEAT_settings.')

    @property
    def total_adjusted_fitness(self) -> float:
        """Return the total adjusted fitness of Players in all Species."""
        return sum([specie.total_adjusted_fitness for specie in self.species]) if self.species else .0
    
    @property
    def gone_stale(self) -> bool:
        """Return True if no improvements have been made for too many generations."""
        return self.staleness >= self._max_staleness

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

    def check_progress(self) -> None:
        """Check whether the Population is improving on both a Specie and overall level."""

        for specie in self.species:
            specie.check_progress()

        if self.species[0].best_fitness > self.best_fitness:
            self.staleness = 0
            self.best_fitness = self.species[0].best_fitness
        else:
            self.staleness += 1

    def fitness_share(self) -> None:
        """Compute the adjusted fitness for each Player in each Species."""

        for specie in self.species:
            specie.fitness_share()

    def remove_stale_species(self) -> None:
        """Remove Species which haven't improved for too many generations."""
        self.species = [specie for specie in self.species if specie.staleness < self._max_specie_staleness]

    def remove_bad_species(self) -> None:
        """Remove Species which are deemed too bad to reproduce.
        
        More specifically remove the Species which will be assigned fewer than one child 
        in the calculation for number of offspring per species.
        """

        total_population_adjusted_fitness = self.total_adjusted_fitness
        self.species =  [specie for specie in self.species if (specie.total_adjusted_fitness /
                         total_population_adjusted_fitness) * self._size >= 1]
        
    def mass_extinction_event(self) -> None:
        """Remove all but the top two perfoming Species."""
        self.species = self.species[:2]

    def repopulate(self) -> None:
        """Populate self.players with the next generation."""

        self.players = []

        total_population_adjusted_fitness = self.total_adjusted_fitness
        for specie in self.species:

            # Get the number of offspring this Species is allocated
            offspring_count = int((specie.total_adjusted_fitness / total_population_adjusted_fitness) * self._size)

            # Check if specie is large enough to include a clone of its champ
            clone_champ = True if specie.size > 5 else False

            # Cut Species down to only Players we want to breed from
            self.cull(self._cull_percentage)

            # Get the Species' offspring
            offspring = specie.generate_offspring(offspring_count, clone_champ)

            # Add offspring to self.players
            self.players.extend(offspring)

            # Clear the Species
            specie.players = []

    def natural_selection(self) -> None:
        """Select the best performing Players from this generation and use them to 
        create the next generation."""

        self.speciate()
        self.rank_species()
        self.check_progress()
        self.fitness_share()

        # Save playback

        if not self.gone_stale():
            self.remove_stale_species()
            self.remove_bad_species()
        else:
            self.mass_extinction_event()

        # Save parents for reloading

        self.repopulate()

        self.generation += 1