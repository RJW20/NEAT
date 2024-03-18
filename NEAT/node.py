from NEAT.connection import Connection
from NEAT.activation_functions import sigmoid


class Node:
    """Node in the Neural Network of a Genome."""

    def __init__(self, number: int, layer: int) -> None:
        self.number: int = number
        self.input: float = .0
        self.output_connections: list[Connection] = []
        self.layer: int = layer

    @property
    def output(self) -> None:
        """Return sigmoid(self.input) or self.input if self.layer = 0"""

        if self.layer != 0:
            return sigmoid(self.input)
        else:
            return self.input

    def engage(self) -> None:
        """Calculate self.output using activation function and add the value 
        to the input of all connected Nodes."""

        for connection in self.output_connections:
            if connection.enabled:
                connection.to_node += self.output * connection.weight