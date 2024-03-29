from __future__ import annotations
from typing import Iterable

from NEAT.genome.node import Node
from NEAT.genome.connection import Connection
from NEAT.genome.activation_functions import ActivationFunction
from NEAT.history import History


class Genome:
    """A Neural Network described by a lists of Nodes and the Connections 
    bewtween them."""

    def __init__(self, input_count: int, output_count: int) -> None:
        """Initialise the lists of Nodes and Connections."""
        
        self.nodes: list[Node] = []
        self.connections: list[Connection] = []
        self.layers: int
        self.input_count: int = input_count
        self.output_count: int = output_count
        self.bias_node_idx: int

    @classmethod
    def new(cls, input_count: int, output_count: int) -> Genome:
        """Return a Genome with the list of Nodes populated with the right amount of 
        (connectionless) input, bias and output Nodes."""

        genome = cls(input_count, output_count)
        genome.layers = 2

        # Add input Nodes
        for _ in range(input_count):
            node = Node(number=genome.next_node, layer=0)
            genome.nodes.append(node)

        # Add bias Node
        genome.bias_node_idx = genome.next_node
        node = Node(number=genome.bias_node_idx, layer=0)
        genome.nodes.append(node)

        # Add output Nodes
        for _ in range(output_count):
            node = Node(number=genome.next_node, layer=1)
            genome.nodes.append(node)

    @property
    def innovation_numbers(self) -> set:
        """Return the innovation numbers found in the this Genome's Connections."""
        return {connection.innovation_number for connection in self.connections}

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
    
    def add_connection(self, from_node: Node, to_node: Node, history: History, weight: float | None = None) -> None:
        """Add a Connection between the specified Nodes.
        
        If weight is given then the new Connection will have that weight, otherwise it will be 
        assigned a new random weight.
        """

        innovation_number = history.get_innovation_number(self, from_node, to_node)
        if weight:
            new_connection = Connection(from_node, to_node, weight, innovation_number)
        else:
            new_connection = Connection.random_weight(from_node, to_node, innovation_number)
        self.connections.append(new_connection)

    def add_node(self, connection: Connection, activation_function: ActivationFunction, history: History) -> None:
        """Disable the given Connection and then insert a Node inbetween the previous from- and 
        to-Nodes and add new Connections between them.
        
        The new Node will have the given activation function.
        """
        
        connection.enabled = False

        # Increment Node layer numbers if there is no space for a new Node
        if connection.to_node - connection.from_node == 1:
            for node in self.nodes:
                if node.layer > connection.from_node.layer:
                    node.layer += 1
                self.layers += 1

        # Create a new Node one layer on from the from_node
        new_node = Node(self.next_node, connection.from_node.layer + 1, activation_function)
        self.nodes.append(new_node)

        # Connect to the original Nodes and the bias Node
        self.add_connection(connection.from_node, new_node, history, weight=1)
        self.add_connection(new_node, connection.to_node, history)
        self.add_connection(self.nodes[self.bias_node_idx], new_node, history, weight=0)

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
    
    def clone(self) -> Genome:
        """Return a copy of this Genome."""

        clone = self.__class__(self.input_count, self.output_count)

        # Add clones of Nodes
        for node in self.nodes:
            clone.nodes.append(node.clone())

        # Add copies of Connections so they connect the new Nodes
        nodes_dict = {node.number: node for node in clone.nodes}
        for connection in self.connections:
            from_node = nodes_dict[connection.from_node.number]
            to_node = nodes_dict[connection.to_node.number]
            clone.connections.append(connection.clone(from_node, to_node))

        clone.layers = self.layers
        clone.bias_node_idx = self.bias_node_idx
        
        return clone
            
    def __repr__(self) -> str:
        """Return representation of this Genome."""
        return f'<Genome: Layers = {self.layers}, Nodes = {len(self.nodes)}, Connections = {len(self.connections)}>'