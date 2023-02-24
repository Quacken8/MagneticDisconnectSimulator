import numpy as np
from constants import *

def idealGas(temperature, density):
    """
    returns pressure accotding to ideal gas law
    """
    numberOfMolecules = density / CMeanMolecularWeight
    return numberOfMolecules*temperature*CBoltzmannConstant