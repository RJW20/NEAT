from __future__ import annotations

from NEAT.genome.connection import Connection
from NEAT.genome.activation_functions import ActivationFunction, linear


class Node:
    """Node in the Neural Network of a Genome."""

    def __init__(self, number: int, layer: int, activation: ActivationFunction = linear) -> None:
        self.number: int = number
        self.layer: int = layer
        self.activation: ActivationFunction = activation
        self.input: float = .0
        self.output_connections: list[Connection] = []

    @property
    def output(self) -> float:
        return self.activation(self.input)
    
    @property
    def connected_to(self, other: Node) -> bool:
        """Return True if the other Node is in this Node's list of outputs."""
        return other in [connection.to_node for connection in self.output_connections]

    def engage(self) -> None:
        """Calculate self.output using activation function and add the value 
        to the input of all connected Nodes."""

        output = self.output
        for connection in self.output_connections:
            if connection.enabled:
                connection.to_node.input += output * connection.weight

    def clone(self) -> Node:
        """Return a copy of this Node."""
        return self.__class__(self.number, self.layer, self.activation)

    def __repr__(self) -> str:
        """Return representation of this Node."""
        return f'<Node: Number = {self.number}, Layer = {self.layer}, Activation = {self.activation.__name__}>'