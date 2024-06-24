# NEAT
Implementation of the Neuro-Evolution of Augmenting Topologies [NEAT](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), written in Python.

## Basic Requirements
1. [Python](https://www.python.org/downloads/).
2. Souce code of a game (or problem) that you wish to generate a neural network for, which satisfies the following requirements:
   - The Player can be easily simulated in its environment (see [Simulator](#Simulator)).
   - It is possible to define when the Player is doing well (see [Fitness](#Fitness)).

## Getting Started
- If using [Poetry](https://python-poetry.org/docs/) simply use the command `poetry add git+https://github.com/RJW20/NEAT.git`.
- If not using Poetry, download just the `neat` folder and place it in the root of your current project directory.

In both cases, also download/copy all contents of `example` and place them in a `src/` folder or whatever directory it is you're going to be working in.

## Using the Template

### Player Class
Either fill out all methods described or let the Player class also extend the class for whatever Player exists in the game and ensure it has some of the methods there. You will need to fill out the look and think methods to set the Player's vision (i.e. Genome inputs) and feed them in to the neural network.

### Simulator
This function is called for every Player in the population during each generation. It should consist of a loop that consistently calls the look, think and move methods of the Player and terminates when the Player is deemed to have lost or is successful enough that you consider it to have beaten the game. At the end of the function, a fitness should be assigned to the Player before returning it.

### Fitness
This value is used to rank how well the Players have performed, and in turn the likelihood of a Player passing its genes into the next generation. A higher fitness value indicates a more successful Player than a lower value, and it should be ensured that this is always a positive value. In the simplest case, this could just be a Player's score.

### Configuring the Settings
Some options have a default value indicated in the example `settings.py`, which can be left as None or completely removed and the default value for that setting will be used. If an entire dictionary has default values that you wish to use you may completely remove the dictionary.

#### `player_args`
The arguments needed to initiate an instance of the Player class.

#### `genome_settings`
The parameters determining the initial configuration of the neural network that makes up each Genome:
- `input_count`: the number of inputs for the neural network.
- `output_count`: the number of options the Player has.
- `hidden_activation`: the activation function to use for all Nodes in hidden layers i.e. all layers except input and output.

#### `population_settings`
The properties of the Population that is being evolved:
- `size`: the number of Players per generation.
- `cull_percentage`: the percentage of Players to remove from each Species before creating offspring each generation.
- `max_staleness`: the number of generations to go without improvement before removing all but the 2 best performing Species.
- `save_folder`: folder to save each generation to (overwritten each time) so the program can be paused and resumed.

#### `species_settings`
The parameters controlling the separation of Players/Genomes into distinct Species:
- `excess_coefficient`, `disjoint_coefficient`, `weight_difference_coefficient`, `compatibility_threshold`: values used directly in the formula for determining if two Genomes are part of the same Species.
- `max_staleness`: the number of generations a Species can go without improvement before being removed.

#### `reproduction_settings`
The parameters controlling the creation of the next generation:
- `crossover_rate`: the rate at which offspring are created by crossover then mutation over just mutation.
- `disabled_rate`': the rate at which an inherited Connection is disabled if it was present in both parents and disabled in at least one of them.
- `weights_rate`: the rate at which a Genome will have its Connection weights mutated.
- `weight_replacement_rate`: the rate at which a Genome that is having its weights mutated will replace a weight over perturbing it.
- `connection_rate`: the rate at which a new Connection will be added to a Genome.
- `node_rate`: the rate at which a new Node will be added to a Genome.

#### `progress_settings`
The flags/values determining how and what progress to report at the end of each generation:
- `print_progress`: choose whether to print a record of the progress to the terminal at the end of each generation.
- `record_progress`: choose whether to build a record of the progress at the end of each generation in a csv file.
- `filename`: filename of csv file to output each generation's progress (if applicable).
- `bests`, `averages`: these must be numerical attributes of the Player class you use, and the best (max) and average of these attributes will be tracked.
- `include_species`: choose whether to include the number of Species in the progress report.

#### `playback_settings`
The values controlling how and where Genomes for playback are saved:
- `save_folder`: folder to save the top performing Genomes of each generation to.
- `number`: the number of Genomes from each Species to save (set to -1 for all).

#### `settings`
The dictionary controlling the initiation and duration of the algorithm, as well as collating all other settings into one place:
- `creation_type`: choose whether to start a Population of Players with randomized Genomes or load a previous save - when choosing to load they will be attempted to be loaded from `population_settings['save_folder']`.
- `load_all_settings`: choose whether to load the settings from the save or use the ones present in this file (if applicable) - only the `player_args`, `progress_settings` and `playback_settings` will be replaced.
- `total_generations`: the number of generations to run the Population until (a loaded Population will remember the generation it was saved at and still only run till this number)

#### `simulation_settings`
Any constants/variables needed for simulating the Players should be here for easy adjustments.

## Running the Algorithm
Run the function `main` in the example `main.py`. \
Genomes created during evolution and saved via the playback functionality may be viewed using my [Genome Utility](https://github.com/RJW20/NEAT-genome-utility.git)

