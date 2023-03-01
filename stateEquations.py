import numpy as np
from constants import *

@np.vectorize
def idealGas(temperature: float | np.ndarray, pressure: float | np.ndarray) -> float | np.ndarray:
    """
    returns density accotding to ideal gas law
    """
    raise NotImplementedError("What's the mean molecural weight u donkey, huh?")
    return pressure/temperature*c.MeanMolecularWeight/c.BoltzmannConstant
