import random

from NEAT.base_player import BasePlayer


def fitness_weighted_selection(parents: list[BasePlayer], total: int) -> list[BasePlayer]:
    """Picks total parents (with replacement) at a rate proportional to all parents fitness.

    Requires all parents have fitness >= 0.
    Requires one parent has fitness > 0.
    """

    fitness_table = [parent.fitness for parent in parents]
    try:
        parents = random.choices(parents, weights = fitness_table, k = total)
    except ValueError:
        raise Exception("To use fitness_weighted_selection at least one parent must have a strictly " + 
                        "positive fitness. Please edit your player's fitness calculation function.")

    return parents