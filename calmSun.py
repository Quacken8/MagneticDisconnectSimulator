#!/usr/bin/env python3

"""
This script models stellar interior with absent flux tube
"""

import numpy as np
from dataStructure import SingleTimeDatapoint
from stateEquations import MockupIdealGas as StateEq
from gravity import g
from scipy.integrate import ode as scipyODE
import constants as c

def getCalmSunDatapoint(dlogP:float, logSurfacePressure:float, surfaceTemperature:float, maxDepth:float) -> SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to calm sun (i.e. one without the flux tube). This model (especially the pressure) is necessary for the calculation of B. It integrates hydrostatic equilibrium (which boils down to solving a set of two ODEs that are a function of logP)

    ----------
    Parameters
    ----------
    dlogP : [Pa] step in pressure gradient by which the integration happens
    logSurfacePressure : [Pa] boundary condition of surface pressure
    surfaceTemperature : [K] boundary condition of surface temperature
    maxDepth : [Mm] depth to which integrate
    """
    maxDepth *= c.Mm

    def pressureScaleHeight(logP:float|np.ndarray, z:float|np.ndarray, T:float|np.ndarray) -> float | np.ndarray:
        """
        returns the pressure scale height z meters below the surface if the pressure there is P = exp(log(P)). log is used cuz it's numerically more stable 
        """
        P = np.exp(logP)
        rho = StateEq.density(temperature = T, pressure = P) # TBD at first im just working with this simple  eq, later to be replaced with the sophisticated thing
        H = P/(rho*g(z))
        return H
    
    def advectiveGradient(logP: float | np.ndarray, z: float | np.ndarray, T: float | np.ndarray) -> float | np.ndarray:
        # TBD at first im just working with this simple  eq, later to be replaced with the sophisticated thing
        P = np.exp(logP)
        return StateEq.convectiveGradient(temperature=T, pressure=P)
    
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
    
    # now for each logP (by stepping via dlogP) we find T and z and save them to these arrays

    calmSunZs = []
    calmSunTs = []
    calmSunLogPs = []

    currentZ = 0
    while (currentZ < maxDepth):  # can this be paralelized by not going with cycles through all logPs, but using a preset array of logPs? TBD

        currentZ = ODEIntegrator.y[0]
        currentT = ODEIntegrator.y[1]
        currentLogP = ODEIntegrator.t
        calmSunZs.append(currentZ)
        calmSunTs.append(currentT)
        calmSunLogPs.append(currentLogP)

        currentLogP += dlogP    # step into higher pressure
        ODEIntegrator.integrate(currentLogP)    # get new T and z from this
        if not ODEIntegrator.successful(): break # TBD hey why is this not finishing a viable break condition viz old code ???? raise RuntimeError(f"Integration didn't complete correctly at logP = {currentLogP}")

    calmSunPs = np.exp(calmSunLogPs)

    rhos = StateEq.density(temperature = np.array(calmSunTs), pressure = calmSunPs)
    entropies = StateEq.entropy(temperature = np.array(calmSunTs), pressure = calmSunPs)
    nablaAds = StateEq.adiabaticLogGradient(temperature = np.array(calmSunTs), pressure = calmSunPs)
    cps = StateEq.cp(temperature = np.array(calmSunTs), pressure = calmSunPs)
    cvs = StateEq.cv(temperature=np.array(calmSunTs), pressure=calmSunPs)
    deltas = StateEq.delta(temperature=np.array(calmSunTs), pressure=calmSunPs)
    F_rads = StateEq.F_rad(temperature=np.array(calmSunTs), pressure=calmSunPs)
    F_cons = StateEq.F_con(temperature=np.array(calmSunTs), pressure=calmSunPs)
    B_0s = np.zeros(len(calmSunZs)) # calm sun doesn't have these

    calmSun = SingleTimeDatapoint(temperatures=np.array(calmSunTs), zs = np.array(calmSunZs), pressures=np.exp(np.array(calmSunLogPs)), rhos=rhos, entropies=entropies, nablaAds=nablaAds, cps = cps, cvs=cvs, deltas = deltas, B_0s = B_0s, F_cons = F_cons, F_rads = F_rads)
    return calmSun

def main():
    """
    debugging function for this model used to compare the outcomes to model S
    """
    dlogP = 0.001
    surfaceTemperature = 3500
    logSurfacePressure = np.log(1e4)
    maxDepth = 13*c.Mm
    calmSun = getCalmSunDatapoint(dlogP=dlogP, logSurfacePressure=logSurfacePressure, maxDepth=maxDepth, surfaceTemperature=surfaceTemperature)

    from dataStructure import Data
    data = Data(finalT=1,numberOfTSteps=2)
    data.appendDatapoint(calmSun)
    data.appendDatapoint(calmSun)
    data.saveToFolder("calmSun", rewriteFolder=True)
    

if __name__ == "__main__":
    main()