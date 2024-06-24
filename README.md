# NEAT
Implementation of the Neuro-Evolution of Augmenting Topologies [NEAT](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), written in Python.

## Basic Requirements
1. [Python](https://www.python.org/downloads/).
2. Souce code of a game (or problem) that you wish to generate a neural network for, which satisfies the following requirements:
   - The player can be easily simulated in it's environment (see [Simulator](#Simulator)).
   - It is possible to define when the player is doing well (see [Fitness](#Fitness)).

## Getting Started
- If using [Poetry](https://python-poetry.org/docs/) simply use the command `poetry add git+https://github.com/RJW20/NEAT.git`.
- If not using Poetry, download just the `neat` folder and place it in the root of your current project directory.
In both cases, also copy all files from `example` and place them in a `src/` folder or whatever folder it is your going to be working in.