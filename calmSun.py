#!/usr/bin/env python3

"""
This script models stellar interior with absent flux tube
"""

import numpy as np

from dataStructure import SingleTimeDatapoint
from stateEquationsPT import StateEquationInterface

from stateEquationsPT import F_con, F_rad
from gravity import g, massBelowZ
from scipy.integrate import ode
import constants as c
from typing import Type, Callable
import logging 
logging.basicConfig(level=logging.DEBUG)
L = logging.getLogger(__name__)

def getCalmSunDatapoint(
    StateEq: Type[StateEquationInterface],
    opacity: Callable[[np.ndarray, np.ndarray], np.ndarray],
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
    """

    def setOfODEs(lnP:float, zlnTArray:np.ndarray) -> np.ndarray:
        """
        the set of ODEs that are derived from hydrostatic equilibrium
        dz/dlnP   = H(z,T,P)
        dlnT/dlnP = ∇(z,T,P)
        """
        z = zlnTArray[0]
        T = np.exp(zlnTArray[1])
        P = np.exp(lnP)
        gravAcc = np.array(g(z))
        m_z = massBelowZ(z)

        H = StateEq.pressureScaleHeight(temperature=T, pressure=P, gravitationalAcceleration=gravAcc)
        nablaAd = StateEq.adiabaticLogGradient(temperature=T, pressure=P)
        kappa = opacity(P, T)
        nablaRad = StateEq.radiativeLogGradient(temperature=T, pressure=P, massBelowZ=m_z, opacity=kappa)
        nabla = np.minimum(nablaAd, nablaRad)

        return np.array([H, nabla])
    
    # initial conditions
    currentZlnTValues = np.array([surfaceZ, np.log(surfaceTemperature)])
    currentZ = surfaceZ
    lnPressure = lnSurfacePressure

    # set up the scipy integrator
    ODEIntegrator = ode(setOfODEs)
    ODEIntegrator.set_integrator("dopri5") # TODO make sure this is a good choice for integrator
    ODEIntegrator.set_initial_value(currentZlnTValues, lnPressure)
    
    # set up the arrays that will be filled with the results
    calmSunZs = [currentZ]
    calmSunTs = [surfaceTemperature]
    calmSunPs = [np.exp(lnPressure)]

    # integrate
    L.info("Integrating calm sun")

    while currentZ < maxDepth:
        # integrate to the next pressure step
        nextZlnTValues = ODEIntegrator.integrate(ODEIntegrator.t + dlnP)
        nextZ = nextZlnTValues[0]
        nextT = np.exp(nextZlnTValues[1])
        nextP = np.exp(ODEIntegrator.t + dlnP)

        # append the results
        calmSunZs.append(nextZ)
        calmSunTs.append(nextT)
        calmSunPs.append(nextP)

        # update the current values
        currentZlnTValues = nextZlnTValues
        currentZ = nextZ

        if ODEIntegrator.successful() == False:
            raise Exception(f"Integration of calm sun failed at z={currentZ/c.Mm} Mm")
    

    calmSun = SingleTimeDatapoint(
        zs=np.array(calmSunZs),
        temperatures=np.array(calmSunTs),
        pressures=np.array(calmSunPs),
    )
    return calmSun


def main():
    pass

if __name__ == "__main__":
    main()
