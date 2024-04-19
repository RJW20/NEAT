from .player import Player
from .settings import simulation_settings


def simulate(player: Player) -> Player:
    """Run the player in its environment and assign it a fitness signifying how 
    well it performs.
    
    The assigned fitness must be positive (>=0).
    """




    player.fitness = 0
    return player