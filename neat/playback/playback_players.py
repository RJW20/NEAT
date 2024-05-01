from pathlib import Path

from neat.base_player import BasePlayer
from neat.genome import Genome


class PlaybackPlayers:
    """Class controlling access to saved playback Genomes."""

    def __init__(
        self, 
        playback_folder: Path, 
        PlayerClass: type,
        player_args: dict,
        g: int = 1,
        per_species: bool = True,
    ) -> None:
        self.folder: Path = playback_folder
        self._PlayerClass: type = PlayerClass
        self._player_args: dict = player_args

        self.species: list[list[BasePlayer]]
        self.species_no: int = 0
        self.generation: int = g
        self.per_species: bool = per_species
    
    @property
    def generation(self) -> int:
        return self._generation
    
    @generation.setter
    def generation(self, g: int) -> None:
        """Set self._generation and load in the Genomes from the corresponding save folder."""

        self._generation = g

        self.species = []
        species_source = self.folder / str(g)
        for folder in species_source.iterdir():
            
            specie = []
            genomes_source = species_source / folder
            try:
                genomes = [Genome.load(genomes_source, filename) for filename in genomes_source.iterdir()]
                for genome in genomes:
                    player = self._PlayerClass(self._player_args)
                    player.genome = genome
                    specie.append(player)
                self.species.append(specie)
            except OSError:
                raise Exception('Please keep playback saves clean from other files.')

    @property
    def per_species(self) -> bool:
        return self._per_species
    
    @per_species.setter
    def per_species(self, value: bool) -> None:
        """Set self._per_species and create self.current_players."""

        self._per_species = value

        if value:
            self.current_players = self.species[self.species_no]
        else:
            self.current_players = []
            for specie in self.species:
                self.current_players.extend(specie)

    def __getitem__(self, index: int) -> BasePlayer:
        """Return the player at the given index value in self.current_players."""
        return self.current_players[index]