from NEAT.node import Node
from NEAT.connection import Connection


class Genome:
    """A Neural Network described by a lists of Nodes and the Connections 
    bewtween them."""

    def __init__(self) -> None:
        self.nodes: list[Node]
        self.connections: list[Connection]