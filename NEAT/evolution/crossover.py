import random

from NEAT.genome import Genome


def crossover(genome1: Genome, genome2: Genome, disabled_rate: float) -> Genome:
    """Return the Genome that is the result of crossing over the given Genomes.
    
    genome1 will be treated as the more fit Genome meaning both excess and disjoint 
    Connections will be inherited from it.
    disabled_rate is the rate at which an inherited Connection is disabled if it was 
    present in both Genomes and disabled in at least one of them.
    """

    result = Genome(genome1.input_count, genome2.input_count)

    # Since excess and disjoint Connections come from genome1, the resulting crossover 
    # can only have Nodes from it (and hence also has the same number of layers)
    for node in genome1.nodes:
        result.nodes.append(node.clone())
    result.layers = genome1.layers

    # Add (clones of) Connections from both Genomes
    genome2_connections = genome2.connections_dict
    result_nodes = result.nodes_dict
    for connection1 in genome1.connections:

        enabled = True

        # If the Connection is also in genome2
        try:
            connection2 = genome2_connections[connection1.innovation_number]

            # Consider disabling it if its disabled in either Genome
            if not connection1.enabled or not connection2.enabled:
                if random.uniform(0, 1) < disabled_rate:
                    enabled = False

            # Choose which Genome to clone the Connection from
            connection = connection1 if random.uniform(0, 1) < 0.5 else connection2          

        # Else will be excess or disjoint
        except KeyError:
            connection = connection1

        # Add a clone of the chosen Connection
        finally:
            from_node = result_nodes[connection.from_node.number]
            to_node = result_nodes[connection.to_node.number]
            result_connection = connection.clone(from_node, to_node)
            result_connection.enabled = enabled
            result.connections.append(result_connection)

    return result