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
    },

    'progress_settings': {
        'print_progress': True,
        'record_progress': False,
        'filename': 'progress',
        'bests': ['fitness'],
        'averages': ['fitness'],
        'include_species': True,
    },

    'playback_settings': {
        'save_folder': 'playback',
        'number': 1,
    },

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

    'progress_settings': {
        'print_progress': bool,
        'record_progress': bool,
        'filename': str,
        'bests': list,
        'averages': list,
        'include_species': bool,
    },

    'playback_settings': {
        'save_folder': str,
        'number': int,
    },

}


def settings_handler(settings: dict, silent: bool = False) -> dict:
    """Make sure all settings exist (where applicable), are of the right type and 
    in their viable range.
    
    Also set default values where settings do not exist (if applicable).
    Print to console all default values used if silent is False.
    """

    # Check the settings has the necessary sub-dictionaries
    try:
        settings['player_args']
        settings['genome_settings']
        settings['population_settings']
    except KeyError as e:
        raise Exception(f'Settings {e.args[0]} not found in settings.')
    
    # Create sub-dictionaries that have full default alternatives if needed
    for setting in ['species', 'reproduction', 'progress', 'playback']:

        try:
            settings[setting + '_settings']
        except KeyError:
            settings[setting + '_settings'] = dict()

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
                if setting is None:
                    raise TypeError
            except (KeyError, TypeError):
                settings[name][key] = default_value
                if not silent:
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
                if isinstance(setting, list):
                    # All lists are lists of strings
                    for value in setting:
                        if not isinstance(value, str):
                            raise TypeError(f'Attribute in {name}[{key}] must be of type str.')
                
                # Range
                if isinstance(setting, int) and name != 'progress_settings':
                    # bools set to False in progress settings register as int = 0
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