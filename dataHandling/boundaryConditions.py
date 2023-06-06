#!/usr/bin/env python3
from typing import Type
import numpy as np
from dataHandling.dataStructure import SingleTimeDatapoint
from stateEquationsPT import StateEquationInterface, MESAEOS
import constants as c
from scipy.integrate import simpson
from scipy.optimize import newton, brentq
import loggingConfig
import logging

from sunSolvers.pressureSolvers import integrateHydrostaticEquilibrium

L = loggingConfig.configureLogging(logging.INFO, __name__)


def getBottomB(
    externalPressure: float | np.ndarray, bottomPressure: float | np.ndarray
) -> float | np.ndarray:
    """
    straight up from the thin tube approximation
    """
    return np.sqrt(np.sqrt(2 * c.mu0 * (externalPressure - bottomPressure)))


def getTopB() -> float:
    """
    in the paper it is said that it is *typically* set to 2k G. Not sure what *typically* means tho lol
    """
    return 2e3 * c.Gauss

def massOfFluxTube(densities, Bs, zs, totalMagneticFlux):
    """
    equation 13 in Schüssler and Rempel 2018
    """
    return totalMagneticFlux * simpson(densities / Bs, zs)

def deltaMass(bottomB, bottomDensity, totalMagneticFlux, dt, upflowVelocity):
    """
    equation 15 in Schüssler and Rempel 2018

    approimation of how the total mass should change if the values of p, B and rho change at the bottom
    """
    return totalMagneticFlux * upflowVelocity * dt * bottomDensity / bottomB

def getAdjustedBottomPressure(
    currentState: SingleTimeDatapoint,
    newTs: np.ndarray,
    bottomExternalPressure: float,
    dlnP: float,
    dt: float,
    upflowVelocity: float,
    totalMagneticFlux: float,
    StateEq: Type[StateEquationInterface] = MESAEOS,
) -> float:
    """
    boundary condition of pressure is only given on the bottom
    returns pressure at the bottom of the flux tube calculated from assumption that it should change based on the inflow of material trough the bottom boundary. Schüssler and Rempel 2018 eq 15
    mass(p + dp) - ( mass(p) + deltaMass ) = 0
    """



    # -------------------
    # cache stuff for toFindRootOf(bottomPressure) that doesnt use the bottomPressure

    # first get old mass from current state
    oldRhos = np.array(StateEq.density(currentState.temperatures, currentState.pressures))
    oldPs = currentState.pressures
    oldBs = currentState.bs
    oldZs = currentState.zs
    oldMass = massOfFluxTube(oldRhos, oldBs, oldZs, totalMagneticFlux)

    # second get the mass adjustment
    massAdjustment = deltaMass(
        bottomDensity=oldRhos[-1],
        bottomB=oldBs[-1],
        totalMagneticFlux=totalMagneticFlux, 
        dt=dt,
        upflowVelocity=upflowVelocity,
    )

    def toFindRootOf(bottomPressure):
        """
        returns the value of the function that should be zero, i.e.
        toFindRootOf = mass(p + dp) - ( mass(p) + deltaMass )
        """

        # third get new pressures from the bottom one (i.e. the variable we are solving for)
        newDatapoint = integrateHydrostaticEquilibrium(
            StateEq=StateEq,
            referenceZs=currentState.zs,
            referenceTs=newTs,
            dlnP = dlnP,
            lnBoundaryPressure=np.log(bottomPressure),
            initialZ=currentState.zs[-1],
            finalZ=currentState.zs[0],
            regularizeGrid=True
        )
        newZs = newDatapoint.zs
        newRhos = StateEq.density(newDatapoint.temperatures, newDatapoint.pressures)

        # and the new mass
        newMass = massOfFluxTube(newRhos, np.interp(newZs, oldZs, oldBs), newZs, totalMagneticFlux)

        # and return the zero
        return newMass - oldMass - massAdjustment
    

    # and now just fund the root

    # first try newton
    try:
        newP = brentq(toFindRootOf, a=oldPs[-1], b=bottomExternalPressure) # FIXME get good bounds
    except RuntimeError:
        newPGuess = oldPs[-1]
        newP = newton(toFindRootOf, x0=newPGuess)
        # if that doesnt work, try brentq
    
    return newP



if __name__ == "__main__":
    pass
