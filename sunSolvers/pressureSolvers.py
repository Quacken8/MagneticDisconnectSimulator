#!/usr/bin/env python3
from dataStructure import SingleTimeDatapoint
import boundaryConditions as bcs
from gravity import g
import numpy as np
from typing import Callable, Type
import constants as c
from stateEquationsPT import StateEquationInterface
import logging
L = logging.getLogger(__name__)

def integratePressure(
    StateEq: Type[StateEquationInterface],
    opacity: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dlnP: float,
    lnSurfacePressure: float,
    surfaceTemperature: float,
    surfaceZ: float,
    maxDepth: float,
    guessTheZRange: bool = False
) -> SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to integrated pressure according to hydrostatic equilibrium in the Sun where both magnetic fields and inflow of material play a role FIXME is this even true

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

    if guessTheZRange == False:
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
    
    elif guessTheZRange==True:

        # get the guess of the integration domain

        paddingFactor = 0.05 # i.e. will, just to be sure, integrate to 120 % of the ln(pressure) expected at maxDepth 
        modelS = loadModelS()
        modelPs = modelS.pressures
        modelZs = modelS.zs
        modelInterpolation = interp1d(modelZs, modelPs)

        maxPGuess = modelInterpolation(maxDepth)
        # get rid of these they might be big
        del modelS, modelPs, modelZs, modelInterpolation

        maxLnPGuess = np.log(maxPGuess)*(1+paddingFactor)

        # set up the integration itself
        calmSunLnPs = np.arange(lnSurfacePressure, maxLnPGuess, dlnP)

        calmSunZs, calmSunLnTs = odeint(func = setOfODEs , y0 = currentZlnTValues, t = calmSunLnPs, tfirst = True, printmessg=True).T

        calmSunPs = np.exp(calmSunLnPs)
        calmSunTs = np.exp(calmSunLnTs)
    

    calmSun = SingleTimeDatapoint(
        zs=np.array(calmSunZs),
        temperatures=np.array(calmSunTs),
        pressures=np.array(calmSunPs),
    )
    return calmSun

def main():
    """test code for this file"""
    pass

if __name__ == "__main__":
    main()
