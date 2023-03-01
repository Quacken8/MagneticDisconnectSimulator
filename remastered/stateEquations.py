import numpy as np
from constants import *

def idealGas(temperature, pressure):
    """
    returns density accotding to ideal gas law
    """
    return pressure/temperature*CMeanMolecularWeight/CBoltzmannConstant
