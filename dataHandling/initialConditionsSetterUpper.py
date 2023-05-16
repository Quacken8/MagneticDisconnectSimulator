#!/usr/bin/env python3
import numpy as np
from dataHandling.dataStructure import SingleTimeDatapoint, Data, subsampleArray
from stateEquationsPT import MESAEOS
from opacity import mesaOpacity
from scipy.optimize import fsolve
import constants as c
from sunSolvers.pressureSolvers import integrateAdiabaticHydrostaticEquilibrium
from dataHandling.modelS import loadModelS
import loggingConfig
import logging
L = loggingConfig.configureLogging(logging.INFO, __name__)


def getInitialConditions(
    numberOfZSteps: int,
    maxDepth: float,
    minDepth: float,
    surfaceTemperature: float = 3500,  # value used by Schüssler and Rempel 2005
) -> SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to flux tube at t=0 with linear magnetic field but otherwise model S
    """
    maxDepth *= c.Mm
    minDepth *= c.Mm
    zs = np.linspace(minDepth, maxDepth, num=numberOfZSteps)

    modelS = loadModelS()

    initialPs = np.interp(zs, modelS.zs, modelS.pressures)
    initialTs = np.interp(zs, modelS.zs, modelS.temperatures)
    initialBs = 0

    raise NotImplementedError()


def getBartaInit(
    p0_ratio: float, maxDepth: float, surfaceZ: float, dlnP: float = 1e-2
) -> SingleTimeDatapoint:
    """
    Returns an adiabatic calm sun but with top pressure scaled relative to model S by p0_ratio and top temperature corresponding to bottom entropy (by which acting as if the whole tube is adiabatic)
    From Bárta's work

    Args:
        p0_ratio (float): ratio of initial pressure to the pressure at the top of the model S
        maxDepth (float): [m] depth to which integrate
        surfaceZ (float): [m] depth of the surface
        dlnP (float, optional): [Pa] step in pressure gradient by which the integration happens. Defaults to 1e-2.
    """

    modelS = loadModelS()

    def surfaceTfromBottomS(
        bottomS: float, surfaceP: float, guessT: float = 3500
    ) -> float:
        """
        returns the temperature at the surface of the tube with entropy bottomS
        """
        T = fsolve(lambda T: bottomS - MESAEOS.entropy(T, surfaceP), x0=guessT)[0]
        return T

    sunSurfacePressure = np.interp(surfaceZ, modelS.zs, modelS.pressures).item()
    lnSurfacePressure = np.log(sunSurfacePressure * p0_ratio)
    bottomS = modelS.derivedQuantities["entropies"][-1]
    L.debug(f"bottomS: {bottomS}")
    surfaceTemperature = surfaceTfromBottomS(
        bottomS, np.exp(lnSurfacePressure)
    )

    initialSun = integrateAdiabaticHydrostaticEquilibrium(
        StateEq=MESAEOS,
        dlnP=dlnP,
        initialZ=surfaceZ,
        lnBoundaryPressure=lnSurfacePressure,
        finalZ=maxDepth,
        boundaryTemperature=surfaceTemperature,
    )

    return initialSun


def mockupDataSetterUpper(zLength: int = 10) -> SingleTimeDatapoint:
    """
    mockup data setter upper that just makes bunch of weird datapoints instead of the pressures and other datapoints of length zLength
    """
    ones = np.arange(zLength)
    maxdepth = 4
    zs = np.linspace(0, maxdepth, num=zLength)

    toReturn = SingleTimeDatapoint(
        temperatures=ones,
        pressures=ones * 10,
        B_0s=ones * 100,
        F_rads=ones * 1000,
        F_cons=ones * 10000,
        entropies=ones * 2,
        nablaads=ones * 4,
        deltas=ones * 6,
        zs=zs,
        rhos=ones * 7,
        cps=ones * 3,
        cvs=ones * 11,
    )
    return toReturn

def main():
    pass

if __name__ == "__main__":
    main()
