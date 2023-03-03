#!/usr/bin/env python3

"""
This script models stellar interior with absent flux tube
"""

import numpy as np
from dataStructure import SingleTimeDatapoint
import stateEquations as se
from gravity import gravity
import warnings
from scipy.integrate import ode as scipyODE

def getCalmSunDatapoint(dlogP:float, logSurfacePressure:float, surfaceTemperature:float) -> SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to calm sun (i.e. one without the flux tube). This model (especially the pressure) is necessary for the calculation of B. It integrates hydrostatic equilibrium (which boils down to solving a set of two ODEs that are a function of logP)

    ----------
    Parameters
    ----------
    dlogP : [Pa] step in pressure gradient by which the integration happens
    logSurfacePressure : [Pa] boundary condition of surface pressure
    surfaceTemperature : [K] boundary condition of surface temperature
    """

    def pressureScaleHeight(logP:float|np.ndarray, z:float|np.ndarray, T:float|np.ndarray) -> float | np.ndarray:
        """
        returns the pressure scale height z meters below the surface if the pressure there is P = exp(log(P)). log is used cuz it's numerically more stable 
        """
        warnings.warn(
            "Uses the ideal gas rn, change it to something sofisticated first")
        P = np.exp(logP)
        rho = se.idealGas(temperature = T, pressure = P) # TBD at first im just working with this simple  eq, later to be replaced with the sophisticated thing
        H = P/(rho*gravity(z))
        return H
    
    def advectiveGradient(logP: float | np.ndarray, z: float | np.ndarray, T: float | np.ndarray) -> float | np.ndarray:
        warnings.warn(
            "Uses the ideal gas rn, change it to something sofisticated first")
        # TBD at first im just working with this simple  eq, later to be replaced with the sophisticated thing
        P = np.exp(logP)
        return se.idealGasConvectiveGradient(temperature=T, pressure=P)
    
    # the two above functions are actually right hand sides of differential equations dz/dlogP and dT/dlogP respectively. They share the independedt variable logP. To solve them we put them together into one array and use scipy integrator

    def setOfODEs(logP : float|np.ndarray, zTArray: np.ndarray)-> np.ndarray:
        """
        the set of ODEs that tie logP, T and z together
        dz/dlogP = H(T, P, z)
        dT/dlogP = ∇(T, P)
        first index corresponds to z/(logP), second index to function of T(logP)
        """
        z = zTArray[0]
        T = zTArray[1]

        H = pressureScaleHeight(logP, z, T)
        nablaAdj = advectiveGradient(logP, z, T)

        return np.array([H, nablaAdj])
    
    ODEIntegrator = scipyODE(setOfODEs)
    ODEIntegrator.set_integrator("dopri5") # this is the RK integrator of order (4)5. Is this the best option? TBD

    surfaceZTValues = np.array([0, surfaceTemperature]) # z = 0 is the definition of surface
    ODEIntegrator.set_initial_value(surfaceZTValues, logSurfacePressure)
    
    # now for each step in logP integrate 


    calmSun = SingleTimeDatapoint()
    return calmSun
