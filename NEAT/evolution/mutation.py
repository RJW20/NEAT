import random

from NEAT.genome import Genome
from NEAT.genome.activation_functions import ActivationFunction
from NEAT.history import History



def mutate_weights(genome: Genome, weight_replacement_rate: float) -> None:
    """Mutate the weights of the given Genome.
    
    weight_replacement_rate is the rate at which a weight will be replaced over 
    perturbing it.
    """

    for connection in genome.connections:
        if random.uniform(0, 1) < weight_replacement_rate:
            connection.weight = random.uniform(-1, 1)
        else:
            connection.weight += random.gauss(0,1) * 0.2
            connection.weight = max(-1, connection.weight)
            connection.weight = min(1, connection.weight)


def add_connection(genome: Genome, history: History) -> None:
    """Add a new Connection with random weight ~U[-1,1] between two random Nodes in the 
    given Genome.
    
    If the Genome is fully-connected then it will be left unchanged.
    """

    if genome.fully_connected:
        return

    # Get two Nodes in different layers that are not connected, in the correct order
    from_node, to_node = random.choices(genome.nodes, k = 2).sorted(key=lambda node: node.layer)
    while from_node.layer == to_node.layer or from_node.connected_to(to_node):
        from_node, to_node = random.choices(genome.nodes, k = 2).sorted(key=lambda node: node.layer)

    # Add the Connection
    genome.add_connection(from_node, to_node, history)


def add_node(genome: Genome, node_activation: ActivationFunction, history: History) -> None:
    """Add a new Node inside a random Connection in the given Genome.
    
    The weight of the Connection from the original from_node to the new Node will be 1.
    The weight of the Connection from the new Node to the original to_node will be the weight 
    of the original Connection.
    The bias will be connected to the new Node by a Connection with weight 0.
    """

    # Get a Connection that is not from the bias Node
    connection = random.choice(genome.connections)
    while connection.from_node == genome.nodes(genome.bias_node_idx):
        connection = random.choice(genome.connections)

    # Add a Node in the middle of the Connection
    genome.add_node(connection, node_activation, history)



def mutate(
    genome: Genome,
    weights_rate: float,
    weight_replacement_rate: float,
    connection_rate: float,
    node_rate: float,
    node_activation: ActivationFunction,
    history: History,
) -> None:
    """Mutate the given Genome in place.

    The given Genome must have had genome.prepare_network() called at some point
    in its lifetime.
    
    weights_rate is the rate at which the Genome will have its Connection 
    weights mutated.
    weight_replacement_rate is the rate at which a Genome that is having its
    weights mutated will replace a weight over perturbing it.
    connection_rate is the rate at which a new Connection will be added.
    node_rate is the rate at which a new Node will be added.
    node_activation is the activation function assigned to new Nodes that are 
    added.

    Each Connection selected for mutation will have a value ~N(0,0.2) added 
    to its weight.
    If a weight becomes out of the range [-1,1] it will be clipped to it.
    """

    if random.uniform(0, 1) < weights_rate:
        mutate_weights(genome, weight_replacement_rate)

    if random.uniform(0, 1) < connection_rate:
        add_connection(genome, history)
                
    if random.uniform(0, 1) < node_rate:
        add_node(genome, node_activation, history)