import numpy as np
from constants import *

def idealGas(temperature:float, pressure:float) -> float:
    """
    returns density accotding to ideal gas law
    """
    raise NotImplementedError("What's the mean molecural weight u donkey, huh?")
    return pressure/temperature*CMeanMolecularWeight/CBoltzmannConstant
