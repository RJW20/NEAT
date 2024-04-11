
player_args = {}


genome_settings = {

    'input_count': None,
    'output_count': None,
    'hidden_activation': None

}


population_settings = {

    'save_folder': '',
    'size': None,
    'cull_percentage': None,
    'max_staleness': None,

}


species_settings = {

    'excess_coefficient': None,
    'disjoint_coefficient': None,
    'weight_difference_coefficient': None,
    'compatibility_threshold': None,

    'max_staleness': None,
}


reproduction_settings = {

    'crossover_rate': None,
    'disabled_rate': None,
    'weights_rate': None,
    'weight_replacement_rate': None,
    'connection_rate': None,
    'node_rate': None,

}


playback_settings = {

    'save_folder': '',
    'number': -1,   #Set to -1 for entirety of each Species

}


settings = {

    'creation_type': 'new',
    'load_all_settings': True,  ###Any changes here need to be reflected in population.load docstring

    'player_args': player_args,
    'genome_settings': genome_settings,
    'population_settings': population_settings,
    'species_settings': species_settings,
    'reproduction_settings': reproduction_settings,
    'playback_settings': playback_settings,

}


simulation_settings = {}