from NEAT.connection import Connection
from NEAT.activation_functions import sigmoid


class Node:
    """Node in the Neural Network of a Genome."""

    def __init__(self, number: int, layer: int) -> None:
        self.number: int = number
        self.input: float = .0
        self.output: float = .0
        self.output_connections: list[Connection] = []
        self.layer: int = layer

    def engage(self) -> None:
        """Calculate self.output using activation function and add the value 
        to the input of all connected Nodes."""

        # Apply activation if not input or bias
        if self.layer != 0:
            self.output = sigmoid(self.input)

        for connection in self.output_connections:
            if connection.enabled:
                connection.to_node += self.output * connection.weight