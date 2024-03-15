from NEAT.node import Node


class Innovation:
    """Stores information about a new Connection.
    
    Each Innovation contains a unique number and a list of all other innovation 
    numbers that were present in the Genome when this Innovation occurred; this 
    uniquely identifies a new Connection between two Nodes.
    """

    def __init__(self) -> None:
        self.number: int
        self.state_occurred: list[int]