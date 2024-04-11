defaults = {

    'genome_settings': {
        'hidden_activation': 'sigmoid',
    },

    'population_settings': {
        'cull_percentage': 0.5, 
        'max_staleness': 20,
    },

    'species_settings': {
        'excess_coefficient': 1,
        'disjoint_coefficient': 1,
        'weight_difference_coefficient': 0.4,
        'compatibility_threshold': 3,
        'max_staleness': 15,
    },

    'reproduction_settings': {
        'crossover_rate': 0.75,
        'disabled_rate': 0.75,
        'weights_rate': 0.8,
        'weight_replacement_rate': 0.1,
        'connection_rate': 0.1,
        'node_rate': 0.03,
    }

}


types = {

    'genome_settings': {
        'input_count': int,
        'output_count': int,
        'hidden_activation': str,
    },

    'population_settings': {
        'size': int,
        'cull_percentage': float, 
        'max_staleness': int,
        'save_folder': str,
    },

    'species_settings': {
        'excess_coefficient': float | int,
        'disjoint_coefficient': float | int,
        'weight_difference_coefficient': float | int,
        'compatibility_threshold': float | int,
        'max_staleness': int,
    },

    'reproduction_settings': {
        'crossover_rate': float,
        'disabled_rate': float,
        'weights_rate': float,
        'weight_replacement_rate': float,
        'connection_rate': float,
        'node_rate': float,
    },

    'playback_settings': {
        'save_folder': str,
        'number': int,
    }

}


def settings_handler(settings: dict) -> dict:
    """Make sure all settings exist (where applicable), are of the right type and 
    in their viable range.
    
    Also set default values where settings do not exist (if applicable).
    """

    # Check the settings has the necessary sub-dictionaries
    try:
        settings['player_args']
        settings['genome_settings']
        settings['population_settings']
        settings['playback_settings']
    except KeyError as e:
        raise Exception(f'Settings {e.args[0]} not found in settings.')
    
    # Create sub-dictionaries that have full default alternatives if needed
    try:
        settings['species_settings']
    except KeyError:
        settings['species_settings'] = dict()

    try:
        settings['species_settings']
    except KeyError:
        settings['reproduction_settings'] = dict()

    # Verify they are dictionaries
    for name in types.keys():
        settings_dict = settings[name]
        if not isinstance(settings_dict, dict):
            raise Exception(f'Settings {settings_dict} must be a dictionary')
        
    # Set default values where appropriate
    for name, default in defaults.items():
        for key, default_value in default.items():
            try:
                setting = settings[name][key]
                if not setting:
                    raise TypeError
            except (KeyError, TypeError):
                settings[name][key] = default_value
                print(f'Using default value {default_value} for \'{key}\' in {name}.')

    # Check values exist, are of right type and in their viable range
    for name, type_dict in types.items():
        settings_dict = settings[name]
        for key, type_value in type_dict.items():

            # Exists
            try:
                setting = settings[name][key]

                # Type
                if not isinstance(setting, type_value):
                    raise TypeError(f'Setting \'{key}\' in {name} must be of type {type_value}.')
                
                # Range
                if isinstance(setting, int):
                    # All ints > 0 except playback_settings['number']
                    if setting <= 0 and key != 'number':
                        raise ValueError(f'Setting \'{key}\' in {name} must be positive.')
                elif isinstance(setting, float):
                    # All floats in [.0, 1.0] except in species_settings
                    if setting < 0 or (setting > 1 and name != 'species_settings'):
                        raise ValueError(f'Setting \'{key}\' in {name} must in range [0.0, 1.0].')

            except KeyError as e:
                raise Exception(f'Setting \'{e.args[0]}\' must be included in settings.')
            
    return settings