from __future__ import annotations
from pathlib import Path
import pickle

from NEAT.history.innovation import Innovation
from NEAT.genome.node import Node

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NEAT.genome import Genome

class History:
    """Contains all previous Innovations."""

    def __init__(self) -> None:
        self.innovations: list[Innovation] = []

    @property
    def next_innovation_number(self) -> int:
        """The number to assign to the next new Innovation."""
        return len(self.innovations)

    def get_innovation_number(self, genome: Genome, from_node: Node, to_node: Node) -> int:
        """Return the innovation number for a Genome mutation that is making a new Connection 
        (new Node or new Connection mutation).
        
        If the mutation is the first of its kind it will be assigned a new number, else 
        it will be matched up with a previous Innovation.
        """

        # Check all current Innovations
        for innovation in self.innovations:
            if innovation.match(genome, from_node, to_node):
                return innovation.number
            
        # Create a new Innovation
        new_innovation = Innovation(self.next_innovation_number, genome, from_node, to_node)
        self.innovations.append(new_innovation)
        return new_innovation.number
    
    def save(self, destination: Path) -> None:
        """Save this instance of History in the given destination with pickle.
        
        If a History save already exists in the same destination it will be overwritten.
        """

        with destination.open('wb') as dest:
            pickle.dump(self, dest)

    @classmethod
    def load(cls, source: Path) -> History:
        """Return the history saved in the given source.
        
        Throws an OSError if fails to open the save.
        """

        with source.open('rb') as src:
            return pickle.load(src)