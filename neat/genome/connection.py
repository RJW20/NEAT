from __future__ import annotations
import random

from neat.genome.node import Node


class Connection:
    """Connection between two Nodes in the Neural Network of a Genome."""

    def __init__(self,
                 from_node: Node,
                 to_node: Node,
                 weight: float,
                 innovation_number: int,
                 enabled: bool = True
                 ) -> None:
        self.from_node: Node = from_node
        self.to_node: Node = to_node
        self.weight: float = weight
        self.innovation_number: int = innovation_number
        self.enabled: bool = enabled

    @classmethod
    def random_weight(cls, from_node: Node, to_node: Node, innovation_number: int) -> Connection:
        """Return an enabled connection between the given Nodes with a random weight ~U[-1,1]."""

        weight = random.uniform(-1, 1)
        return cls(from_node, to_node, weight, innovation_number)
    
    def clone(self, from_node: Node, to_node: Node) -> Connection:
        """Return a copy of this Connection but with different Nodes.
        
        This is so we can clone Nodes and then form new Connections between the cloned Nodes.
        """
        return self.__class__(from_node, to_node, self.weight, self.innovation_number, self.enabled)

    def __repr__(self) -> str:
        """Return representation of this Connection."""
        return f'<Connection: From = {self.from_node.number}, To = {self.to_node.number}, ' + \
               f'Innovation = {self.innovation_number}, Enabled = {self.enabled}>'