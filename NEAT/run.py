from typing import Callable
from multiprocessing import Pool, cpu_count

from NEAT.base_player import BasePlayer
from NEAT.population import Population


def run(
    PlayerClass: type,
    simulate: Callable[[BasePlayer], BasePlayer],
    settings: dict,
) -> None:
    """Apply the NEAT algorithm to a Population of Players.
     
    The Player's class will be of given class PlayerClass, which must inherit from NEAT.BasePlayer.
    The simulate function must run a given Player in its environment and then assign it a (positive) 
    fitness signifying how well it performs.
    The settings dictionary must contain:
     - 3 entries determining the generation type and lifespan of the Population.
     - player_args: the PlayerClass' required instantiation arguments.
     - genome_settings: describing the Player's Genome architecture.
     - population_settings: describing the Population characteristics.
     - playback_settings: determining where and what to save each generation.
    """

    # Create the Population
    creation_type = settings['creation_type']
    match(creation_type):
        case 'new':
            population = Population.new(PlayerClass, settings)
        case 'load':
            try:
                load_folder = settings['population_settings']['save_folder']
                population = Population.load(PlayerClass, settings, load_folder)
            except KeyError:
                raise Exception(f'Setting population_settings[\'save_folder\'] must be present (and contain a viable save) \
                                to load a Population.')

    # Evolve the Population to the desired generation
    try:
        total_generations = settings['total_generations']
    except KeyError as e:
        raise Exception(f'Setting {e.args[0]} not found in settings.')

    cores_to_use = cpu_count() // 2

    while population.generation < total_generations:

        with Pool(cores_to_use) as pool:
            population.players = pool.map(simulate, population.players, chunksize=1)

        population.evolve()