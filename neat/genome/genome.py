from __future__ import annotations
from typing import Iterable, Generator
from pathlib import Path, PosixPath
import pickle

from neat.genome.node import Node
from neat.genome.connection import Connection
from neat.genome.activation_functions import ActivationFunction, sigmoid
from neat.history import History


class Genome:
    """A Neural Network described by lists of Nodes and the Connections 
    between them."""

    def __init__(self, input_count: int, output_count: int) -> None:
        self.nodes: list[Node] = list()
        self.layers: int
        self.input_count: int = input_count
        self.output_count: int = output_count
        self.bias_node_idx: int = input_count

    @property
    def connections(self) -> Generator[Connection,None,None]:
        """Return the Connections in the Genome."""
        for node in self.nodes:
            yield from node.output_connections

    @property
    def innovation_numbers(self) -> set[int]:
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
        nodes_in_front = {layer: sum([nodes_in_layers[i] for i in range(layer + 1, self.layers)]) for layer in range(self.layers)}

        # Compute the number of connections a fully connected NN would have
        max_connections = 0
        for layer in range(self.layers):
            max_connections += nodes_in_front[layer] * nodes_in_layers[layer]

        return max_connections == len(list(self.connections))

    @property
    def nodes_dict(self) -> dict[int, Node]:
        """Return this Genome's Nodes as a dictionary with their numbers as the keys."""
        return {node.number: node for node in self.nodes}
    
    @property
    def connections_dict(self) -> dict[int, Connection]:
        """Return this Genome's Connections as a dictionary with their innovation numbers 
        as the keys."""
        return {connection.innovation_number: connection for connection in self.connections}
    
    @classmethod
    def new(cls, input_count: int, output_count: int, history: History) -> Genome:
        """Return a Genome with a list of Nodes containing the input, bias and output Nodes, 
        and a list of (random) Connections between them."""

        genome = cls(input_count, output_count)
        genome.layers = 2

        # Add input Nodes
        for _ in range(input_count):
            node = Node(number=genome.next_node, layer=0)
            genome.nodes.append(node)

        # Add bias Node
        node = Node(number=genome.bias_node_idx, layer=0)
        genome.nodes.append(node)

        # Add output Nodes
        for _ in range(output_count):
            node = Node(number=genome.next_node, layer=1, activation=sigmoid)
            genome.nodes.append(node)

        # Fully connect the network
        for from_node in genome.nodes[:genome.bias_node_idx + 1]:
            for to_node in genome.nodes[genome.bias_node_idx + 1:]:
                genome.add_connection(from_node, to_node, history)

        return genome
    
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
        new_connection.from_node.output_connections.append(new_connection)

    def add_node(self, connection: Connection, activation_function: ActivationFunction, history: History) -> None:
        """Disable the given Connection and then insert a Node inbetween the previous from- and 
        to-nodes and add new Connections between them.
        
        The weight of the Connection from the original from_node to the new Node will be 1.
        The weight of the Connection from the new Node to the original to_node will be the weight 
        of the original Connection.
        The bias will be connected to the new Node by a Connection with weight 0.
        The new Node will have the given activation function.
        """
        
        connection.enabled = False

        # Increment Node layer numbers if there is no space for a new Node
        if connection.to_node.layer - connection.from_node.layer == 1:
            for node in self.nodes:
                if node.layer > connection.from_node.layer:
                    node.layer += 1
            self.layers += 1

        # Create a new Node one layer on from the from_node
        new_node = Node(self.next_node, connection.from_node.layer + 1, activation_function)
        self.nodes.append(new_node)

        # Reorder the Nodes so they engage in the correct order
        self.nodes.sort(key=lambda node: (node.layer, node.number))

        # Connect to the original Nodes and the bias Node
        self.add_connection(connection.from_node, new_node, history, weight=1)
        self.add_connection(new_node, connection.to_node, history)
        self.add_connection(self.nodes[self.bias_node_idx], new_node, history, weight=0)

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
        clone_nodes = clone.nodes_dict
        for connection in self.connections:
            from_node = clone_nodes[connection.from_node.number]
            to_node = clone_nodes[connection.to_node.number]
            cloned_connection = connection.clone(from_node, to_node)
            cloned_connection.from_node.output_connections.append(cloned_connection)

        # Let all the Nodes know about each other

        clone.layers = self.layers
        clone.bias_node_idx = self.bias_node_idx
        
        return clone

    def save(self, folder: Path, filename: str) -> None:
        """Save this Genome instance in the given folder with the given filename using pickle.
        
        If a Genome save already exists in the same folder with the same filename it will be 
        overwritten.
        """

        destination = folder / f'{filename}.pickle'
        with destination.open('wb') as dest:
            pickle.dump(self, dest)

    @classmethod
    def load(cls, path: Path, filename: PosixPath | None = None) -> Genome:
        """Create a Genome instance from a pickle dump.
         
        If just the path is given it will be attempted to load from there.
        If both the path and filename are given the path will be treated as the directory to 
        look in for the filename. 
        """

        source = path / filename if filename else path

        try:
            with source.open('rb') as src:
                return pickle.load(src)
        except OSError:
            if filename:
                raise OSError(f'Unable to open Genome save \'{filename}\' in \'{path}\'.')
            else:
                raise OSError(f'Unable to open Genome save \'{path}\'.')
            
    def __repr__(self) -> str:
        """Return representation of this Genome."""
        return f'<Genome: Layers = {self.layers}, Nodes = {len(self.nodes)}, Connections = {len(self.connections)}>'