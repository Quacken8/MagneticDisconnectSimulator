#!/usr/bin/env python3

"""
This script models stellar interior with absent flux tube
"""

import numpy as np
from dataStructure import SingleTimeDatapoint
import stateEquations as se
from gravity import gravity

def getCalmSunDatapoint() -> SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to calm sun (i.e. one without the flux tube). This model (especially the pressure) is necessary for the calculation of B
    """

    def pressureScaleHeight(P:float|np.ndarray, z:float|np.ndarray, T:float|np.ndarray) -> float | np.ndarray:
        """
        returns the pressure scale height z meters below the surface if the pressure there is P 
        """
        rho = se.idealGas(temperature = T, pressure = P) # TBD at first im just working with this simple  eq, later to be replaced with the sophisticated thing
        H = P/(rho*gravity(z))
        return H
    


    calmSun = SingleTimeDatapoint()
    return calmSun
