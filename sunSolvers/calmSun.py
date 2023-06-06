#!/usr/bin/env python3

"""
This script models stellar interior with absent flux tube
"""

import numpy as np

from dataHandling.dataStructure import SingleTimeDatapoint
from stateEquationsPT import StateEquationInterface

import constants as c
from typing import Type, Callable
import loggingConfig
import logging
L = loggingConfig.configureLogging(logging.INFO, __name__)
from sunSolvers.pressureSolvers import (
    integrateHydrostaticEquilibriumAndTemperatureGradient,
)


def getCalmSunDatapoint(
    StateEq: Type[StateEquationInterface],
    opacityFunction: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dlnP: float,
    lnSurfacePressure: float,
    surfaceTemperature: float,
    surfaceZ: float,
    maxDepth: float,
) -> SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to calm sun (i.e. one without the flux tube). This model (especially the pressure) is necessary for the calculation of B. It integrates hydrostatic equilibrium (which boils down to solving a set of two ODEs that are a function of logP)

    ----------
    Parameters
    ----------
    stateEq: a class with static functions that return the thermodynamic quantities as a function of temperature and pressure; see StateEquations.py for an example
    opacity: a function that returns the opacity as a function of pressure and temperature
    dlogP : [Pa] step in pressure gradient by which the integration happens
    logSurfacePressure : [Pa] boundary condition of surface pressure
    surfaceTemperature : [K] boundary condition of surface temperature
    maxDepth : [m] depth to which integrate
    guessTheZRange : if True, will estimate what pressure is at maxDepth using model S, adds a bit of padding (20 %) to it just ot be sure and uses scipy in a bit faster.
    You don't get the exactly correct z range, but it is ~3 times faster
    """
    L.info("integrating calm Sun")
    calmSun = integrateHydrostaticEquilibriumAndTemperatureGradient(
        StateEq=StateEq,
        opacityFunction=opacityFunction,
        dlnP=dlnP,
        lnBoundaryPressure=lnSurfacePressure,
        boundaryTemperature=surfaceTemperature,
        initialZ=surfaceZ,
        finalZ=maxDepth,
    )

    return calmSun


def main():
    """
    This function generates the calm sun model and saves it into a file
    """
    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity

    StateEq = MESAEOS
    opacityFunction = mesaOpacity
    dlnP = 1e-3
    surfaceZ = -2*c.Mm
    maxDepth = 20*c.Mm
    
    # initial values interpolated from model S
    from dataHandling.modelS import loadModelS
    modelS = loadModelS()
    lnSurfacePressure =  np.log(np.interp(surfaceZ, modelS.zs, modelS.pressures)).item()
    surfaceTemperature = np.interp(surfaceZ, modelS.zs, modelS.temperatures).item()

    
    calmSun = getCalmSunDatapoint(
        StateEq=StateEq,
        opacityFunction=opacityFunction,
        dlnP=dlnP,
        lnSurfacePressure=lnSurfacePressure,
        surfaceTemperature=surfaceTemperature,
        surfaceZ=surfaceZ,
        maxDepth=maxDepth,
    )
    calmSun.saveToFolder("calmSun")


if __name__ == "__main__":
    main()
