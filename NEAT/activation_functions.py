import math
from typing import NewType, Callable, Literal


type ActivationFunction = Callable[[float], float]

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def relu(x: float) -> float:
    return max(0, x)

def linear(x: float) -> float:
    return x

def activation_by_name(name: Literal['sigmoid', 'relu', 'linear']) -> ActivationFunction:
    """Return activation function from name."""

    activations = {
        'sigmoid': sigmoid,
        'relu': relu,
        'linear': linear,
    }

    try:
        activation = activations[name]
    except KeyError:
        raise TypeError(f"Invalid activation function {name}.")

    return activation