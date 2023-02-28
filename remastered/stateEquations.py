import numpy as np
from constants import *

@np.vectorize
def idealGas(temperature, pressure):
    """
    returns density accotding to ideal gas law
    """
    return pressure/temperature*CMeanMolecularWeight/CBoltzmannConstant
