from pathlib import Path

from neat.base_player import BasePlayer
from neat.genome import Genome


class PlaybackPlayers:
    """Class controlling access to saved playback Genomes."""

    def __init__(
        self, 
        playback_folder: str, 
        PlayerClass: type,
        player_args: dict,
        g: int = 1,
        per_species: bool = True,
    ) -> None:
        self.folder: str = playback_folder
        self._PlayerClass: type = PlayerClass
        self._player_args: dict = player_args

        self.species: list[list[BasePlayer]]

        self.total_generations = len(list(Path(self.folder).iterdir()))
        self.generation: int = g
        self.species_no: int = 0
        self.per_species: bool = per_species
    
    @property
    def generation(self) -> int:
        return self._generation
    
    @generation.setter
    def generation(self, g: int) -> None:
        """Set self._generation and load in the Genomes from the corresponding save folder."""

        self._generation = ((g - 1) % self.total_generations) + 1

        self.species = []
        species_source = Path(self.folder) / str(g)
        for genomes_source in species_source.iterdir():
            
            specie = []
            try:
                genomes = [Genome.load(file_path) for file_path in genomes_source.iterdir()]
                for genome in genomes:
                    player = self._PlayerClass(self._player_args)
                    player.genome = genome
                    specie.append(player)
                self.species.append(specie)
            except OSError:
                raise Exception('Please keep playback saves clean from other files.')

        # Set the number of species in the current generation    
        self.total_species = len(list(species_source.iterdir()))

    @property
    def species_no(self) -> int:
        return self._species_no
    
    @species_no.setter
    def species_no(self, s: int) -> None:
        """Set self._species_no and set self.current_players to the corresponding species."""
        
        self._species_no = s % self.total_species
        self.current_players = self.species[self._species_no]

    @property
    def per_species(self) -> bool:
        return self._per_species
    
    @per_species.setter
    def per_species(self, value: bool) -> None:
        """Set self._per_species and set self.current_players to either the first Species 
        or the entire generation."""

        self._per_species = value

        if value:
            self.current_players = self.species[0]
        else:
            self.current_players = []
            for specie in self.species:
                self.current_players.extend(specie)

    def __getitem__(self, index: int) -> BasePlayer:
        """Return the player at the given index value in self.current_players."""
        return self.current_players[index]