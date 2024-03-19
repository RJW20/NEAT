from NEAT.species import Species


class Population:
    """Contains all the different Species."""

    def __init__(self) -> None:
        self.species: list[Species]