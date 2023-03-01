#!/usr/bin/env python3
from dataStructure import SingleTimeDatapoint
import boundaryConditions as bcs

def getNewTs(currentState:SingleTimeDatapoint, dt:float):
    undefined


def getNewPs():
    """
    integrates pressure from the assumption of hydrostatic equilibrium (eq 6)
    dp/dz = rho(p(z), T(z)) g(z)
    """
    bottomPressure = bcs.getBottomPressure()
    def g(z):
        """
        gravitational acceleration in m/s^2 z meters below the surface
        """
        return undefined
    
    undefined

def getNewYs(innerPs, outerPs, totalMagneticFlux):
    """
    solves differential equation 5 to get y = sqrt(B) = y(z)
    """
    undefined