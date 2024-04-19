from typing import Iterable, Any

from NEAT import BasePlayer


class Player(BasePlayer):

    def __init__(self, **player_args: dict) -> None:
        super().__init__()
        self.vision: Iterable

    def look(self) -> None:
        """Update the attributes used as input to the Genome."""
        self.vision = []

    def think(self) -> Any:
        """Feed the input into the Genome and return the output as a valid move."""
        return self.genome.propagate(self.vision)

    def move(self, move: Any) -> None:
        """Advance to the state achieved by carrying out move."""
        pass