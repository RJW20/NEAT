from NEAT.baseplayer import BasePlayer


class Species:
    """Contains a large group of Players that have similar Genomes."""

    def __init__(self) -> None:
        self.rep: BasePlayer
        self.players: list[BasePlayer]