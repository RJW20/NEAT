from NEAT.connection import Connection


class Node:
    """Node in the Neural Network of a Genome."""

    def __init__(self) -> None:
        self.number: int
        self.input: float
        self.output: float
        self.output_connections: list[Connection]
        self.layer: int