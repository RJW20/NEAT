from typing import Callable

from NEAT.connection import Connection
from NEAT.activation_functions import ActivationFunction


class Node:
    """Node in the Neural Network of a Genome."""

    def __init__(self, number: int, layer: int, activation: ActivationFunction) -> None:
        self.number: int = number
        self.layer: int = layer
        self.activation: ActivationFunction = activation
        self.input: float = .0
        self.output_connections: list[Connection] = []

    @property
    def output(self) -> float:
        return self.activation(self.input)

    def engage(self) -> None:
        """Calculate self.output using activation function and add the value 
        to the input of all connected Nodes."""

        for connection in self.output_connections:
            if connection.enabled:
                connection.to_node += self.output * connection.weight