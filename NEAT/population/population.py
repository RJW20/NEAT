from NEAT.population.species import Species
from NEAT.history.history import History


class Population:
    """Contains all the different Species.
    
    Must be initiated with a settings dictionary as found in the documentation.
    """

    def __init__(self, settings: dict) -> None:
        self.species: list[Species]
        self.history: list[History]

        # Unload the settings
        try:
            self._excess_coefficient = settings['excess_coefficient']
            self._disjoint_coefficient = settings['disjoint_coefficient']
            self._weight_difference_coefficient = settings['weight_difference_coefficient']
            self._compatibility_threshold = settings['compatibility_threshold']
        except KeyError as e:
            print(f'Setting {e.args[0]} not found in settings.')