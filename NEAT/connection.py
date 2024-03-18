from NEAT.node import Node


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

    def __repr__(self) -> str:
        """Return representation of this Connection."""
        return f'<Connection: From = {self.from_node.number}, To = {self.to_node.number}, 
                 Innovation = {self.innovation_number}, Enabled = {self.enabled}>'