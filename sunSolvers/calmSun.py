#!/usr/bin/env python3

"""
This script models stellar interior with absent flux tube
"""

import numpy as np

from dataHandling.dataStructure import SingleTimeDatapoint
from dataHandling.modelS import loadModelS
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
    dlnp = 1e-2
    maxDepth = 20 * c.Mm
    surfaceZ = 0 * c.Mm

    modelS = loadModelS()
    surfaceT = np.interp(surfaceZ, modelS.zs, modelS.temperatures).item()
    surfaceP = np.interp(surfaceZ, modelS.zs, modelS.pressures).item()

    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity

    calmSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=dlnp,
        lnSurfacePressure=np.log(surfaceP),
        surfaceTemperature=surfaceT,
        surfaceZ=surfaceZ,
        maxDepth=maxDepth,
        opacityFunction=mesaOpacity,
    )

    from dataHandling.dataStructure import loadOneTimeDatapoint
    from dataHandling.dataVizualizer import plotSingleTimeDatapoint
    import matplotlib.pyplot as plt

    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(
        calmSun, toPlot, pltshow=False, label="Calm Sun pre load", log=True
    )
    axs = plotSingleTimeDatapoint(
        modelS, toPlot, axs=axs, pltshow=False, label="Model S", log=True
    )

    calmSun.saveToFolder("calmSun", rewrite=True)
    del calmSun
    calmSun = loadOneTimeDatapoint("calmSun")

    axs = plotSingleTimeDatapoint(
        calmSun, toPlot, axs=axs, pltshow=False, label="Calm Sun post load", log=True, linestyle="--"
    )
    plt.show()


if __name__ == "__main__":
    main()
