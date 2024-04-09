from __future__ import annotations
from pathlib import Path
import pickle
import shutil

from NEAT.base_player import BasePlayer
from NEAT.genome import Genome
from NEAT.population.species import Species
from NEAT.history import History
from NEAT.population.player_factory import PlayerFactory


class Population:
    """Contains Species of Players.
    
    Must be initiated with a settings dictionary as found in the documentation.
    """

    def __init__(self, PlayerClass: type, settings: dict) -> None:
        self.generation: int
        self.history: History
        self.players: list[BasePlayer]
        self.species: list[Species]

        self.staleness: int
        self.best_fitness: int

        # Unload the settings
        try:
            self._player_args: dict = settings['player_args']
            self._genome_settings: dict = settings['genome_settings']

            NEAT_settings = settings['NEAT_settings']
            self._size: int = NEAT_settings['population_size']
            self._cull_percentage: float = NEAT_settings['cull_percentage']
            self._max_staleness: int = NEAT_settings['max_population_staleness']
            self._species_settings: dict = NEAT_settings['species_settings']
            self._reproduction_settings: dict = NEAT_settings['reproduction_settings']

            self._playback_settings: dict = settings['playback_settings']   

            self._save_folder = settings['save_folder'] 

        except KeyError as e:
            raise Exception(f'Setting {e.args[0]} not found in settings.')
        
        self.player_factory: PlayerFactory = PlayerFactory(
            PlayerClass = PlayerClass,
            player_args = self._player_args,
            genome_settings = self._genome_settings,
            reproduction_settings = self._reproduction_settings,
        )

    @property
    def total_adjusted_fitness(self) -> float:
        """Return the total adjusted fitness of Players in all Species."""
        return sum([specie.total_adjusted_fitness for specie in self.species]) if self.species else .0
    
    @property
    def gone_stale(self) -> bool:
        """Return True if no improvements have been made for too many generations."""
        return self.staleness >= self._max_staleness
    
    @classmethod
    def new(cls, PlayerClass: type, settings: dict) -> Population:
        """Return a new Population with a full list of Players with randomized Genomes."""

        population = cls(PlayerClass, settings)
        population.generation = 0
        population.history = History()
        population.players = population.player_factory.new_players(population._size, population.history)
        population.species = []
        population.staleness = 0
        population.best_fitness = 0

        return population

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
                new_species = Species(player, self._species_settings)
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
        self.species = [specie for specie in self.species if not specie.gone_stale]

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

    def next_generation(self) -> None:
        """Populate self.players with the next generation."""

        self.players = []
        self.generation += 1

        total_population_adjusted_fitness = self.total_adjusted_fitness
        for specie in self.species:

            # Get the number of offspring this Species is allocated
            offspring_count = int((specie.total_adjusted_fitness / total_population_adjusted_fitness) * self._size)

            # Insert a clone of the Species if applicable
            if specie.size > 5:
                self.players.append(self.player_factory.clone(specie.champ))
                offspring_count -= 1

            # Cut the Species down to only Players we want to breed from
            specie.cull(self._cull_percentage)

            # Get the Species' offspring
            offspring = self.player_factory.generate_offspring(
                parents = specie.players,
                total = offspring_count,
                history = self.history
            )

            # Add offspring to self.players
            self.players.extend(offspring)

            # Clear the Species
            specie.players = []

    def evolve(self) -> None:
        """Select the best performing Players from this generation and use them to 
        create the next generation."""

        self.speciate()
        self.rank_species()
        self.check_progress()
        self.fitness_share()

        # Save playback

        if not self.gone_stale:
            self.remove_stale_species()
            self.remove_bad_species()
        else:
            self.mass_extinction_event()

        self.next_generation()

        self.save()

    def save(self) -> None:
        """Save the Population and its attributes to self._save_folder.
        
        Creates the folder if it doesn't already exist.
        If there is already a Population dump in the folder it will be overwritten.
        """

        save_folder = Path(self._save_folder)

        # Create the folder if it doesn't already exist
        save_folder.mkdir(parents=True, exist_ok=True)

        # Settings
        NEAT_settings = {
            'population_size': self._size,
            'cull_percentage': self._cull_percentage,
            'max_population_staleness': self._max_staleness,
            'species_settings': self._species_settings,
            'reproduction_settings': self._reproduction_settings,
        }
        settings = {
            'player_args': self._player_args,
            'genome_settings': self._genome_settings,
            'NEAT_settings': NEAT_settings,
            'playback_settings': self._playback_settings,
            'save_folder': self._save_folder,
        }
        settings_destination = save_folder / 'settings.pickle'
        with settings_destination.open('wb') as settings_dest:
            pickle.dump(settings, settings_dest)

        # Basic attributes
        attributes = {
            'generation': self.generation,
            'staleness': self.staleness,
            'best_fitness': self.best_fitness,
        }
        attributes_destination = save_folder / 'attributes.pickle'
        with attributes_destination.open('wb') as attributes_dest:
            pickle.dump(attributes, settings_dest)

        # History
        history_dest = save_folder / 'history.pickle'
        self.history.save(history_dest)

        # Players' Genomes
        genomes_destination = save_folder / 'genomes'
        if genomes_destination.exists() and genomes_destination.is_dir():
            shutil.rmtree(genomes_destination)
        genomes_destination.mkdir()
        for i, player in enumerate(self.players):
            player.genome.save(genomes_destination, f'{i}.pickle')

        # Species
        species_destination = save_folder / 'species'
        if species_destination.exists() and species_destination.is_dir():
            shutil.rmtree(species_destination)
        species_destination.mkdir()
        for i, specie in enumerate(self.species):
            specie.save(species_destination / f'{i}.pickle')

    @classmethod
    def load(cls, PlayerClass: type, settings: dict, folder: Path) -> Population:
        """Return the Population saved in the given folder.
         
        The PlayerClass's player_args and the Population's playback_settings will either from the 
        saved settings or the given settings depending on settings['load_all_settings'].
        """

        # Load all aspects of the save
        try:

            # Saved settings
            settings_source = folder / 'settings.pickle'
            with settings_source.open('rb') as settings_src:
                loaded_settings = pickle.load(settings_src)
            try:
                if not settings['load_all_settings']:
                    loaded_settings['player_args'] = settings['player_args']
                    loaded_settings['NEAT_settings']['playback_settings'] = settings['NEAT_settings']['playback_settings']
            except KeyError as e:
                raise Exception(f'Setting {e.args[0]} not found in {e.args[1]}.')

            # Basic attributes
            attributes_source = folder / 'attributes.pickle'
            with attributes_source.open('rb') as attributes_src:
                loaded_attributes = pickle.load(attributes_src)

            # History
            history_source = folder / 'history.pickle'
            loaded_history = History.load(history_source)

            # Players' Genomes
            genomes_source = folder / 'genomes'
            loaded_genomes = [Genome.load(genomes_source, filename) for filename in genomes_source.iterdir()]
                
            # Species
            species_source = folder / 'species'
            loaded_species = [Species.load(species_source / filename) for filename in species_source.iterdir()]

        except OSError as e:
            raise Exception(f'Unable to open part of Population save {e.filename} in {folder}')

        # Create the Population instance with appropriate settings and attributes  
        population = cls(PlayerClass, loaded_settings)
        
        try:
            population.generation = loaded_attributes['generation']
            population.staleness = loaded_attributes['staleness']
            population.best_fitness = loaded_attributes['best_fitness']
        except KeyError as e:
            raise Exception(f'Attribute {e.args[0]} not found in attributes save attributes.pickle in {folder}.')

        population.history = loaded_history

        population.players = []
        for genome in loaded_genomes:
            player = population.player_factory.empty_player()
            player.genome = genome
            population.players.append(player)

        population.species = loaded_species