from NEAT.node import Node
from NEAT.innovation import Innovation


class Connection:
    """Connection between two Nodes in the Neural Network of a Genome."""

    def __init__(self) -> None:
        self.from_node: Node
        self.to_node: Node
        self.weight: float
        self.enabled: bool
        self.innovation: Innovation