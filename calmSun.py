#!/usr/bin/env python3

"""
This script models stellar interior with absent flux tube
"""

import numpy as np
from dataStructure import SingleTimeDatapoint
from stateEquations import IdealGas as StateEq
from stateEquations import F_con, F_rad
from gravity import g
from scipy.integrate import ode as scipyODE
import constants as c


# FIXME - this is made regarding the log of pressure however the new state eq tables will probably work with T and Rho as their variables; therefore the of integration dP/dz should rly go to dP/drho dRho/dz cuz the new tables also support dp/drho
# FIXME - or rather dlogP/dP dP/drho drho/dz of course

def getCalmSunDatapoint(convectiveLength:float, dlogP: float, logSurfacePressure: float, surfaceTemperature: float, maxDepth: float) -> SingleTimeDatapoint:
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

    def pressureScaleHeight(logP: np.ndarray, z: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        returns the pressure scale height z meters below the surface if the pressure there is P = exp(log(P)). log is used cuz it's numerically more stable 
        """
        gravitationalAcc = g(z)
        P = np.exp(logP)
        return StateEq.pressureScaleHeight(temperature = T, pressure = P, gravitationalAcceleration=gravitationalAcc) # type: ignore oh my GOD why is pylance so bad at scipy

    def actualLogGradient(logP: np.ndarray, z: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        computes the actual log gradient the way the bc thesis does it (eq. 4.4)
        """
        gravitationalAcc = g(z)
        P = np.exp(logP)
        convectiveG = StateEq.convectiveLogGradient(temperature=T, pressure=P)
        radiativeG = StateEq.radiativeLogGradient(temperature=T, pressure=P, gravitationalAcceleration=gravitationalAcc) # type: ignore oh my GOD why is pylance so bad at scipy
        return np.minimum(radiativeG, convectiveG)

    # the two above functions are actually right hand sides of differential equations dz/dlogP and dT/dlogP respectively. They share the independedt variable logP. To solve them we put them together into one array and use scipy integrator

    def setOfODEs(logP: np.ndarray, zTArray: np.ndarray) -> np.ndarray:
        """
        the set of ODEs that tie logP, T and z together
        dz/dlogP = H(T, P, z)
        dT/dlogP = ∇(T, P, z)
        first index corresponds to z/(logP), second index to function of T(logP)
        """
        z = zTArray[0]
        T = zTArray[1]

        H = pressureScaleHeight(logP, z, T)
        nablaAdj = actualLogGradient(logP, z, T)

        return np.array([H, nablaAdj])

    ODEIntegrator = scipyODE(setOfODEs)
    # TODO this is the RK integrator of order (4)5. Is this the best option?
    ODEIntegrator.set_integrator("dopri5")

    # z = 0 is the definition of surface
    surfaceZTValues = np.array([0, surfaceTemperature])
    ODEIntegrator.set_initial_value(surfaceZTValues, logSurfacePressure)

    # now for each logP (by stepping via dlogP) we find T and z and save them to these arrays

    calmSunZs = []
    calmSunTs = []
    calmSunLogPs = []

    currentZ = 0
    while (currentZ < maxDepth):  # TODO can this be paralelized by not going with cycles through all logPs, but using a preset array of logPs?

        currentZ = ODEIntegrator.y[0]
        currentT = ODEIntegrator.y[1]
        currentLogP = ODEIntegrator.t
        calmSunZs.append(currentZ)
        calmSunTs.append(currentT)
        calmSunLogPs.append(currentLogP)

        currentLogP += dlogP    # step into higher pressure
        ODEIntegrator.integrate(currentLogP)    # get new T and z from this
        if not ODEIntegrator.successful():
            # TODO hey why is this not finishing a viable break condition viz old code ???? raise RuntimeError(f"Integration didn't complete correctly at logP = {currentLogP}")
            break

    calmSunPs = np.exp(calmSunLogPs)
    calmSunTs = np.array(calmSunTs)

    rhos = StateEq.density(temperature=calmSunTs, pressure=calmSunPs)
    entropies = StateEq.entropy(
        temperature=calmSunTs, pressure=calmSunPs)
    nablaAds = StateEq.adiabaticLogGradient(
        temperature=calmSunTs, pressure=calmSunPs)
    cps = StateEq.cp(temperature=calmSunTs, pressure=calmSunPs)
    cvs = StateEq.cv(temperature=calmSunTs, pressure=calmSunPs)
    deltas = StateEq.delta(temperature=calmSunTs, pressure=calmSunPs)
    F_rads = F_rad(temperature=calmSunTs, pressure=calmSunPs)

    gs = g(calmSunZs)
    nablaRads = StateEq.radiativeLogGradient(temperature = calmSunTs, pressure = calmSunPs, gravitationalAcceleration=gs)
    Hps = StateEq.pressureScaleHeight(temperature=calmSunTs, pressure=calmSunPs, gravitationalAcceleration=gs)
    mus = StateEq.meanMolecularWeight(calmSunTs, calmSunPs)

    F_cons = F_con(temperature=calmSunTs, pressure=calmSunPs, c_p = cps, gravitationalAcceleration=gs, radiativeGrad=nablaRads, adiabaticGrad=nablaAds, meanMolecularWeight=mus, pressureScaleHeight=Hps, convectiveLength=convectiveLength)

    B_0s = np.zeros(len(calmSunZs))  # calm sun doesn't have these

    calmSun = SingleTimeDatapoint(temperatures=calmSunTs, zs=np.array(calmSunZs), pressures=np.exp(np.array(
        calmSunLogPs)), rhos=rhos, entropies=entropies, nablaAds=nablaAds, cps=np.array(cps), cvs=cvs, deltas=deltas, B_0s=B_0s, F_cons=np.array(F_cons), F_rads=np.array(F_rads))
    return calmSun


def main():
    """
    debugging function for this model used to compare the outcomes to model S
    """
    dlogP = 0.001
    surfaceTemperature = 3500
    logSurfacePressure = np.log(1e4)
    maxDepth = 13*c.Mm
    convectimeMixingLength = ???
    calmSun = getCalmSunDatapoint(dlogP=dlogP, logSurfacePressure=logSurfacePressure,
                                  maxDepth=maxDepth, surfaceTemperature=surfaceTemperature, convectiveLength=convectimeMixingLength)

    from dataStructure import Data
    data = Data(finalT=1, numberOfTSteps=2)
    data.appendDatapoint(calmSun)
    data.appendDatapoint(calmSun)
    data.saveToFolder("calmSun", rewriteFolder=True)


if __name__ == "__main__":
    main()
