from collections import OrderedDict
from pathlib import Path
from csv import writer, reader, DictWriter

from neat.base_player import BasePlayer
from neat.population.species import Species


class ProgressHandler:

    def __init__(self, settings: dict, generation: int) -> None:
        self.print_progress: bool = settings['print_progress']
        self.record_progress: bool = settings['record_progress']
        self.filename: str = settings['filename']

        # Build an ordered dictionary of attributes we are keeping track of
        bests = settings['bests']
        averages = settings['averages']
        self.observables = OrderedDict()
        for attribute in bests:
            self.observables[attribute] = {'best'}
        for attribute in averages:
            try:
                self.observables[attribute].add('average')
            except KeyError:
                self.observables[attribute] = {'average'}

        self.include_species: bool = settings['include_species']

        # Initiate the record and column headers if needed.
        self.fieldnames: list
        self.create_record(generation)

    @property
    def settings(self) -> dict:
        """Recollect the attributes that make up the progress_settings dictionary."""

        settings = {
            'print_progress': self.print_progress,
            'record_progress': self.record_progress,
            'filename': self.filename,
            'bests': [attribute for attribute, measures in self.observables.items() if 'best' in measures],
            'averages': [attribute for attribute, measures in self.observables.items() if 'average' in measures],
            'include_species': self.include_species,
        }
        return settings
    
    def create_record(self, generation: int) -> None:
        """Create the record if it doesn't exist, if it does exist and generation=1 then an Exception
        will be thrown.
        
        Also create or read in the column headers.
        """

        record = Path(self.filename + '.csv')

        if generation == 1 and record.exists():
            raise Exception(f'A progress record already exists in {record}, please move it or change \'filename\' ' + \
                            'in progress_settings.')

        # Create file and column headers if needed
        if not record.exists():

            self.fieldnames = ['generation']
            for attribute, measures in self.observables.items():
                for measure in measures:
                    self.fieldnames.append(f'{measure}_{attribute}')
            if self.include_species:
                self.fieldnames.append('number_of_species')

            with record.open("w+") as csv_file:
                csv_writer = writer(csv_file, delimiter=',')
                csv_writer.writerow(self.fieldnames)

        # Read in the column headers
        else:
            with record.open('r') as csv_file:
                csv_reader = reader(csv_file, delimiter=',')
                self.fieldnames = next(csv_reader)


    def create_report(self, players: list[BasePlayer]) -> dict:
        """Create a dictionary of the observations and their values."""
        
        observations = OrderedDict()
        for attribute, measures in self.observables.items():
            try:
                observation = dict()
                values = [player.__dict__[attribute] for player in players]
                if 'best' in measures:
                    observation['best'] = max(values)
                if 'average' in measures:
                    observation['average'] = sum(values) / len(values)
                observations[attribute] = observation
            except KeyError as e:
                raise Exception(f'Attribute \'{e.args[0]}\' not found in the given PlayerClass, please check all ' + 
                                'attributes listed in progress_settings.')
        
        return observations

    def print_report(self, generation: int, report: dict, no_of_species: int) -> None:
        """Print all observations in the report to the console."""

        print(f'\nGeneration: {generation}')
        INDENT = " " * 4
        for attribute, observation in report.items():
            print(f'{INDENT}{attribute}: {", ".join([f'{key} = {value}' for key, value in observation.items()])}')
        if self.include_species:
            print(f'{INDENT}Number of Species: {no_of_species}')

    def record_report(self, generation: int, report: dict, no_of_species: int) -> None:
        """Add all observations in the report to a new line in the record.

        If extra observations have been made since the record was created they won't be included.
        """

        record = Path(self.filename + '.csv')

        # Turn report into valid row
        row = {'generation': generation}
        for attribute, observation in report.items():
            for key, value in observation.items():
                row[f'{key}_{attribute}'] = value
        if self.include_species:
            row['number_of_species'] = no_of_species

        # Add the row
        with record.open("a") as csv_file:
            csv_writer = DictWriter(csv_file, self.fieldnames, restval='', extrasaction='ignore')
            csv_writer.writerow(row)

    def report(self, generation: int, players: list[BasePlayer], species: list[Species]) -> None:
        """Print out and record progress as required."""

        report = self.create_report(players)
        no_of_species = len(species)
        self.print_report(generation, report, no_of_species)
        self.record_report(generation, report, no_of_species)