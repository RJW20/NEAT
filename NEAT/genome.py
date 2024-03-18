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
        self.layers: int = 2

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

    def prepare_network(self) -> None:
        """Prepare the list of Nodes to be used as a NN."""

        # Assign all Connections to the Nodes themselves so we can engage them properly
        for node in self.nodes:
            node.output_connections.clear()
        for connection in self.connections:
            connection.from_node.output_connections.append(connection)

        # Sort the Nodes by layer (first-key) and by number (second-key) so that they will be 
        # engaged in the correct order and the input and output Nodes don't change position
        self.nodes.sort(key=lambda node: (node.layer, node.number))

    def __repr__(self) -> str:
        """Return representation of this Genome."""
        return f'<Genome: Layers = {self.layers}, Nodes = {len(self.nodes)}, Connections = {len(self.connections)}>'