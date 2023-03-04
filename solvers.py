#!/usr/bin/env python3
from dataStructure import SingleTimeDatapoint
import boundaryConditions as bcs
from gravity import g

def getNewTs(currentState:SingleTimeDatapoint, dt:float):
    raise NotImplementedError()


def getNewPs(currentState:SingleTimeDatapoint, dt:float, upflowVelocity:float, totalMagneticFlux:float, bottomExternalPressure:float):
    """
    integrates pressure from the assumption of hydrostatic equilibrium (eq 6)
    dp/dz = rho(p(z), T(z)) g(z)
    """
    bottomPressure = bcs.getBottomPressure(currentState=currentState, dt=dt, upflowVelocity=upflowVelocity, totalMagneticFlux=totalMagneticFlux, bottomExternalPressure=bottomExternalPressure)
    # g(z)
    
    raise NotImplementedError()

def getNewYs(innerPs, outerPs, totalMagneticFlux):
    """
    solves differential equation 5 to get y = sqrt(B) = y(z)
    """
    raise NotImplementedError()
