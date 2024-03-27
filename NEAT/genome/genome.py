from __future__ import annotations
from typing import Iterable

from NEAT.genome.node import Node
from NEAT.genome.connection import Connection


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
        self.output_count: int = output_count

        # Add input Nodes
        for _ in range(input_count):
            node = Node(number=self.next_node, layer=0)
            self.nodes.append(node)

        # Add bias Node
        self.bias_node_idx = self.next_node
        node = Node(number=self.bias_node_idx, layer=0)
        self.nodes.append(node)

        # Add output Nodes
        for _ in range(output_count):
            node = Node(number=self.next_node, layer=1)
            self.nodes.append(node)

    @property
    def next_node(self) -> int:
        """The number to assign to the next Node that is added to this Genome."""
        return len(self.nodes)
    
    @property
    def fully_connected(self) -> bool:
        """Return True if the NN is fully connected."""

        # Create a dictionary containing the number of Nodes in each layer
        nodes_in_layers = {layer: 0 for layer in range(self.layers)}
        for node in self.nodes:
            nodes_in_layers[node.layer] += 1

        # Create a dictionary containing the number of Nodes in front of a layer
        nodes_in_front = {sum([nodes_in_layers[i] for i in range(layer + 1, self.layers)]) for layer in range(self.layers)}

        # Compute the number of connections a fully connected NN would have
        max_connections = 0
        for layer in range(self.layers):
            max_connections += nodes_in_front[layer] * nodes_in_layers[layer]

        return max_connections == len(self.connections)
    
    def add_connection(self) -> None:
        pass

    def add_node(self) -> None:
        pass

    def prepare_network(self) -> None:
        """Prepare the list of Nodes to be used as a NN."""

        # Assign all Connections to the Nodes themselves so we can engage them
        for node in self.nodes:
            node.output_connections.clear()
        for connection in self.connections:
            connection.from_node.output_connections.append(connection)

        # Sort the Nodes by layer (first-key) and by number (second-key) so that they will be 
        # engaged in the correct order and the input and output Nodes don't change relative position
        self.nodes.sort(key=lambda node: (node.layer, node.number))

    def propagate(self, input: Iterable[float]) -> tuple[float, ...]:
        """Feed in input values for the NN and return output.
        
        The input must already be in order and normalised.
        """

        # Clear all Node input values
        for node in self.nodes:
            node.input = 0

        # Set layer zero Nodes input values
        for i, value in enumerate(input):
            self.nodes[i].input = value
        self.nodes[self.bias_node_idx].input = 1

        # Propagate the values
        for node in self.nodes:
            node.engage()

        # Return the output Node output values
        return tuple([node.output for node in self.nodes[len(self.nodes) - self.output_count:]])

    def __repr__(self) -> str:
        """Return representation of this Genome."""
        return f'<Genome: Layers = {self.layers}, Nodes = {len(self.nodes)}, Connections = {len(self.connections)}>'