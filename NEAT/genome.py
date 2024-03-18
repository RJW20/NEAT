from __future__ import annotations

from NEAT.node import Node
from NEAT.connection import Connection


class Genome:
    """A Neural Network described by a lists of Nodes and the Connections 
    bewtween them."""

    def __init__(self, input_count: int, output_count: int) -> None:
        """Initialise the lists of Nodes and Connections.
        
        Populate the list of Nodes with the required amount of (connectionless) 
        Nodes with no activation function.
        """
        
        self.nodes: list[Node] = []
        self.connections: list[Connection] = []

        # Add input Nodes
        for _ in range(input_count):
            node = Node(number=self.next_node, layer=0)
            self.nodes.append(node)

        # Add output Nodes
        for _ in range(output_count):
            node = Node(number=self.next_node, layer=1)
            self.nodes.append(node)

        # Add bias Node
        node = Node(number=self.next_node, layer=0)
        self.nodes.append(node)

    @property
    def next_node(self) -> int:
        """The number to assign to the next Node that is added to this Genome."""

        return len(self.nodes) + 1