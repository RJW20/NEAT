from .player import Player
from .settings import simulation_settings


def simulate(player: Player) -> Player:
    """Run the player in its environment and assign it a fitness depending on how 
    well it performs."""


    
    player.fitness = 0
    return player