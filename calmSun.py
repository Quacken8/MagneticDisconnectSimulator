#!/usr/bin/env python3

"""
This script models stellar interior with absent flux tube
"""

import numpy as np
import warnings
from dataStructure import SingleTimeDatapoint
from stateEquationsPT import StateEquationInterface

warnings.warn("Ur using ideal gas here")
from stateEquationsPT import F_con, F_rad
from gravity import g
from scipy.integrate import ode as scipyODE
import constants as c
from typing import Type


# FIXME - this is made regarding the log of pressure however the new state eq tables will probably work with T and Rho as their variables; therefore the of integration dP/dz should rly go to dP/drho dRho/dz cuz the new tables also support dp/drho
# FIXME - or rather dlogP/dP dP/drho drho/dz of course


# def getCalmSunDatapointRho(convectiveLength:float, dlogRho: float, logSurfaceDensity: float, surfaceTemperature: float, maxDepth: float) -> SingleTimeDatapoint:
#     """
#     returns a datapoint that corresponds to calm sun (i.e. one without the flux tube). This model (especially the pressure) is necessary for the calculation of B. It integrates hydrostatic equilibrium (which boils down to solving a set of two ODEs that are a function of logP)

#     ----------
#     Parameters
#     ----------
#     dlogP : [Pa] step in pressure gradient by which the integration happens
#     logSurfacePressure : [Pa] boundary condition of surface pressure
#     surfaceTemperature : [K] boundary condition of surface temperature
#     maxDepth : [Mm] depth to which integrate
#     """
#     maxDepth *= c.Mm

#     def actualLogGradient(logRho: np.ndarray, z: np.ndarray, T: np.ndarray) -> np.ndarray:
#         """
#         computes the actual log gradient the way the bc thesis does it (eq. 4.4)
#         """
#         gravitationalAcc = g(z)
#         Rho = np.exp(logRho)
#         convectiveG = StateEq.convectiveLogGradient(temperature=T, density=Rho)
#         radiativeG = StateEq.radiativeLogGradient(temperature=T, density=Rho, gravitationalAcceleration=gravitationalAcc) # type: ignore oh my GOD why is pylance so bad at scipy
#         return np.minimum(radiativeG, convectiveG)

#     def nabTP(logRho: np.ndarray, z: np.ndarray, T: np.ndarray, P: np.ndarray) -> np.ndarray:
#         nabla = actualLogGradient(logRho, z, T)
#         return nabla*T/P

#     # the two above functions are actually right hand sides of differential equations dz/dlogP and dT/dlogP respectively. They share the independedt variable logP. To solve them we put them together into one array and use scipy integrator

#     def setOfODEs(logRho: np.ndarray, zTArray: np.ndarray) -> np.ndarray:
#         """
#         the set of ODEs that tie logRho, T and z together
#         dz/dlnRho = ∂P/∂Rho (g(1-∂P/∂T nabTP))^-1
#         dT/dlnRho = Rho nabTP (1-∂P/∂T nabTP)^-1
#         first index corresponds to z/(logRho), second index to function of T(logRho)
#         """
#         z = zTArray[0]
#         T = zTArray[1]

#         rho = np.exp(logRho)
#         partPpartT = StateEq.partialPpartialT(density = rho, temperature = T)
#         partPpartRho = StateEq.partialPpartialRho(density = rho, temperature = T)
#         P = StateEq.pressure(density = rho, temperature = T)
#         denominator = 1-partPpartT*nabTP(logRho, z, T, P)

#         return np.array([1/(denominator*g(z)), rho*nabTP/denominator])*partPpartRho

#     ODEIntegrator = scipyODE(setOfODEs)
#     # TODO this is the RK integrator of order (4)5. Is this the best option?
#     ODEIntegrator.set_integrator("dopri5")

#     # z = 0 is the definition of surface
#     surfaceZTValues = np.array([0, surfaceTemperature])
#     ODEIntegrator.set_initial_value(surfaceZTValues, logSurfaceDensity)

#     # now for each logRho (by stepping via dlogRho) we find T and z and save them to these arrays

#     calmSunZs = []
#     calmSunTs = []
#     calmSunLogRhos = []

#     currentZ = 0
#     while (currentZ < maxDepth):  # TODO can this be paralelized by not going with cycles through all logRhos, but using a preset array of logRhos?

#         currentZ = ODEIntegrator.y[0]
#         currentT = ODEIntegrator.y[1]
#         currentLogRho = ODEIntegrator.t
#         calmSunZs.append(currentZ)
#         calmSunTs.append(currentT)
#         calmSunLogRhos.append(currentLogRho)

#         currentLogRho += dlogRho    # step into higher density
#         ODEIntegrator.integrate(currentLogRho)    # get new T and z from this
#         if not ODEIntegrator.successful():
#             # TODO hey why is this not finishing a viable break condition viz old code ???? raise RuntimeError(f"Integration didn't complete correctly at logRho = {currentLogRho}")
#             break

#     calmSunRhos = np.exp(calmSunLogRhos)
#     calmSunTs = np.array(calmSunTs)

#     pressures = StateEq.density(temperature=calmSunTs, densities=calmSunRhos)
#     entropies = StateEq.entropy(
#         temperature=calmSunTs, densities=calmSunRhos)
#     nablaAds = StateEq.adiabaticLogGradient(
#         temperature=calmSunTs, densities=calmSunRhos)
#     cps = StateEq.cp(temperature=calmSunTs, densities=calmSunRhos)
#     cvs = StateEq.cv(temperature=calmSunTs, densities=calmSunRhos)
#     deltas = StateEq.delta(temperature=calmSunTs, densities=calmSunRhos)
#     F_rads = F_rad(temperature=calmSunTs, densities=calmSunRhos)

#     gs = np.array(g(calmSunZs))
#     nablaRads = StateEq.radiativeLogGradient(temperature = calmSunTs, densities=calmSunRhos, gravitationalAcceleration=gs)
#     Hps = StateEq.pressureScaleHeight(temperature=calmSunTs, densities=calmSunRhos, gravitationalAcceleration=gs)
#     mus = StateEq.meanMolecularWeight(calmSunTs, calmSunRhos)

#     F_cons = F_con(temperature=calmSunTs, densities=calmSunRhos, c_p = cps, gravitationalAcceleration=gs, radiativeGrad=nablaRads, adiabaticGrad=nablaAds, meanMolecularWeight=mus, pressureScaleHeight=Hps, convectiveLength=convectiveLength)

#     B_0s = np.zeros(len(calmSunZs))  # calm sun doesn't have these

#     calmSun = SingleTimeDatapoint(temperatures=calmSunTs, zs=np.array(calmSunZs), pressures=pressures, rhos=calmSunRhos, entropies=entropies, nablaAds=nablaAds, cps=cps, cvs=cvs, deltas=deltas, B_0s=B_0s, F_cons=F_cons, F_rads=F_rads)
#     return calmSun


def getCalmSunDatapoint(
    StateEq: Type[StateEquationInterface],
    dlnP: float,
    logSurfacePressure: float,
    surfaceTemperature: float,
    maxDepth: float,
) -> SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to calm sun (i.e. one without the flux tube). This model (especially the pressure) is necessary for the calculation of B. It integrates hydrostatic equilibrium (which boils down to solving a set of two ODEs that are a function of logP)

    ----------
    Parameters
    ----------
    stateEq: a class with static functions that return the thermodynamic quantities as a function of temperature and pressure; see StateEquations.py for an example
    dlogP : [Pa] step in pressure gradient by which the integration happens
    logSurfacePressure : [Pa] boundary condition of surface pressure
    surfaceTemperature : [K] boundary condition of surface temperature
    maxDepth : [Mm] depth to which integrate
    """
    maxDepth *= c.Mm

    def pressureScaleHeight(
        logP: np.ndarray, z: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        """
        returns the pressure scale height z meters below the surface if the pressure there is P = exp(log(P)). log is used cuz it's numerically more stable
        """
        gravitationalAcc = g(z)
        P = np.exp(logP)
        H = StateEq.pressureScaleHeight(temperature=T, pressure=P, gravitationalAcceleration=gravitationalAcc)  # type: ignore oh my GOD why is pylance so bad at scipy
        return H

    def actualLogGradient(logP: np.ndarray, z: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        computes the actual log gradient the way the bc thesis does it (eq. 4.4)
        """
        gravitationalAcc = g(z)
        P = np.exp(logP)
        convectiveG = StateEq.adiabaticLogGradient(temperature=T, pressure=P)
        radiativeG = StateEq.radiativeLogGradient(temperature=T, pressure=P, gravitationalAcceleration=gravitationalAcc)  # type: ignore oh my GOD why is pylance so bad at scipy
        return np.minimum(radiativeG, convectiveG)

    # the two above functions are actually right hand sides of differential equations dz/dlogP and dT/dlogP respectively. They share the independedt variable logP. To solve them we put them together into one array and use scipy integrator

    def setOfODEs(logP: np.ndarray, zlnTArray: np.ndarray) -> np.ndarray:
        """
        the set of ODEs that tie lnP, lnT and z together
        dz/dlnP = H(T, P, z)
        dlnT/dlnP = ∇(T, P, z)
        first index corresponds to z(lnP), second index to function of lnT(lnP)
        """
        z = zlnTArray[0]
        T = np.exp(zlnTArray[1])

        H = pressureScaleHeight(logP, z, T)
        nabla = actualLogGradient(logP, z, T)

        return np.array([H, nabla])

    ODEIntegrator = scipyODE(setOfODEs)
    # TODO this is the RK integrator of order (4)5. Is this the best option?
    ODEIntegrator.set_integrator("dopri5")

    # z = 0 is the definition of surface
    surfaceZTValues = np.array([0, np.log(surfaceTemperature)])
    ODEIntegrator.set_initial_value(surfaceZTValues, logSurfacePressure)

    # now for each lnP (by stepping via dlnP) we find T and z and save them to these arrays

    calmSunZs = []
    calmSunLnTs = []
    calmSunLnPs = []

    currentZ = 0
    
    while (
        currentZ < maxDepth
    ):  # TODO can this be paralelized by not going with cycles through all logPs, but using a preset array of logPs?
        currentZ = ODEIntegrator.y[0]
        currentLnT = ODEIntegrator.y[1]
        currentLnP = ODEIntegrator.t
        calmSunZs.append(currentZ)
        calmSunLnTs.append(currentLnT)
        calmSunLnPs.append(currentLnP)

        currentLnP += dlnP  # step into higher pressure
        try:
            ODEIntegrator.integrate(currentLnP)  # get new T and z from this
        except Exception as ex:
            print(f"last known z = {currentZ}, last known lnP = {currentLnP}")
            raise ex
        if not ODEIntegrator.successful():
            # TODO hey why is this not finishing a viable break condition viz old code ????
            raise RuntimeError(
                f"Integration didn't complete correctly at lnP = {currentLnP}"
            )

    calmSunPs = np.exp(calmSunLnPs)
    calmSunTs = np.exp(calmSunLnTs)

    rhos = StateEq.density(temperature=calmSunTs, pressure=calmSunPs)
    nablaads = StateEq.adiabaticLogGradient(temperature=calmSunTs, pressure=calmSunPs)

    calmSun = SingleTimeDatapoint(
        temperatures=calmSunTs,
        zs=np.array(calmSunZs),
        pressures=calmSunPs,
        rhos=rhos,
        nablaads=nablaads,
    )
    return calmSun


def main():
    pass

if __name__ == "__main__":
    main()
