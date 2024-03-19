from NEAT.innovation import Innovation
from NEAT.genome import Genome
from NEAT.node import Node


class History:
    """Contains all previous Innovations."""

    def __init__(self) -> None:
        self.innovations: list[Innovation] = []

    @property
    def next_innovation(self) -> int:
        """The number to assign to the next new Innovation."""
        return len(self.innovations)

    def get_innovation_number(self, genome: Genome, from_node: Node, to_node: Node) -> int:
        """Return the innovation number for a mutation.
        
        If the mutation is the first of its kind it will be assigned a new number, else 
        it will be matched up with a previous Innovation.
        """

        # Check all current Innovations
        for innovation in self.innovations:
            if innovation.match(genome, from_node, to_node):
                return innovation.number
            
        # Create a new Innovation
        new_innovation = Innovation(self.next_innovation, genome, from_node, to_node)
        self.innovations.append(new_innovation)
        return new_innovation.number