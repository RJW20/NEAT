from NEAT.genome import Genome
from NEAT.genome.node import Node


class Innovation:
    """Stores information about a new Connection.
    
    Each Innovation contains a unique number and a list of all other innovation 
    numbers that were present in the Genome when this Innovation occurred; this 
    uniquely identifies a new Connection between two Nodes.
    """

    def __init__(self, number: int, genome: Genome, from_node: Node, to_node: Node) -> None:
        self.number: int = number
        self.present_connections: set[int] = genome.innovation_numbers.copy()
        self.from_node_number: int = from_node.number
        self.to_node_number: int = to_node.number

    def match(self, genome: Genome, from_node: Node, to_node: Node) -> bool:
        """Return True if a Genome that is about to make a new Connection matches 
        this Innovation."""

        if not (self.from_node_number == from_node.number and self.to_node_number == to_node.number):
            return False

        if len(self.present_connections) != len(genome.connections):
            return False
        
        for connection in genome.connections:
            if connection.innovation_number not in self.present_connections:
                return False
            
        return True
