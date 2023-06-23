#!/usr/bin/env python3
import numpy as np
from pyrsistent import b
from dataHandling import dataStructure
from stateEquationsPT import MESAEOS
import opacity
from scipy import optimize
import constants as c
from sunSolvers import pressureSolvers
from dataHandling import modelS
import loggingConfig
import logging

L = loggingConfig.configureLogging(logging.INFO, __name__)
from dataHandling.boundaryConditions import getTopB


def initialBottomB(z: float) -> float:
    """
    returns the initial magnetic field strength at depth z inspired by schussler and rempel 2005
    """
    bAt12andhalfMm = 1e3 * c.Gauss
    bAt0Mm = getTopB()

    return np.interp(z, [0, 12.5 * c.Mm], [bAt0Mm, bAt12andhalfMm]).item()


def getInitialConditions(
    numberOfZSteps: int,
    maxDepth: float,
    minDepth: float,
    surfaceTemperature: float = 3500,  # value used by Schüssler and Rempel 2005
) -> dataStructure.SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to flux tube at t=0 with linear magnetic field but otherwise model S
    """
    maxDepth *= c.Mm
    minDepth *= c.Mm
    zs = np.linspace(minDepth, maxDepth, num=numberOfZSteps)

    model = modelS.loadModelS()

    raise NotImplementedError()
    initialPs = model.pressures(zs)
    initialTs = model.temperatures(zs)
    initialBs = 0


def getAdiabaticInitialConditions(
    numberOfZSteps: int,
    maxDepth: float,
    minDepth: float,
    surfaceTemperature: float = 3500,  # value used by Schüssler and Rempel 2005
) -> dataStructure.SingleTimeDatapoint:
    """
    Creates adiabatic calm sun with given surface temperature and linearly interpolated magnetic field
    """

    initialSun = pressureSolvers.integrateAdiabaticHydrostaticEquilibrium(
        StateEq=MESAEOS,
        dlnP=1e-2,
        initialZ=minDepth,
        lnBoundaryPressure=np.log(1e5),  #
        finalZ=maxDepth,
        boundaryTemperature=surfaceTemperature,
        regularizeGrid=True,
    )

    return initialSun


def getBartaInit(
    p0_ratio: float,
    maxDepth: float,
    surfaceZ: float,
    dlnP: float = 1e-2,
    bottomB: float | None = None,
    topB: float | None = None,
) -> dataStructure.SingleTimeDatapoint:
    """
    Returns an adiabatic calm sun but with top pressure scaled relative to model S by p0_ratio and top temperature corresponding to bottom entropy (by which acting as if the whole tube is adiabatic)
    From Bárta's work

    Args:
        p0_ratio (float): ratio of initial pressure to the pressure at the top of the model S
        maxDepth (float): [m] depth to which integrate
        surfaceZ (float): [m] depth of the surface
        dlnP (float, optional): [Pa] step in pressure gradient by which the integration happens. Defaults to 1e-2.
    """

    model = modelS.loadModelS()

    def surfaceTfromBottomS(
        bottomS: float, surfaceP: float, guessT: float = 3500
    ) -> float:
        """
        returns the temperature at the surface of the tube with entropy bottomS
        """
        T = optimize.fsolve(
            lambda T: bottomS - MESAEOS.entropy(T, surfaceP), x0=guessT
        )[0]
        return T

    sunSurfacePressure = np.interp(surfaceZ, model.zs, model.pressures).item()
    lnSurfacePressure = np.log(sunSurfacePressure * p0_ratio)
    bottomS = np.interp(maxDepth, model.zs, model.derivedQuantities["entropies"])
    surfaceTemperature = surfaceTfromBottomS(
        bottomS=bottomS, surfaceP=np.exp(lnSurfacePressure)
    )
    L.debug(f"bottomS: {bottomS}")
    L.debug(f"surfaceT: {surfaceTemperature}")
    L.debug(f"surfaceP: {sunSurfacePressure * p0_ratio}")

    initialSun = pressureSolvers.integrateAdiabaticHydrostaticEquilibrium(
        StateEq=MESAEOS,
        dlnP=dlnP,
        initialZ=surfaceZ,
        lnBoundaryPressure=lnSurfacePressure,
        finalZ=maxDepth,
        boundaryTemperature=surfaceTemperature,
        regularizeGrid=True,
    )
    if bottomB is None:
        bottomB = initialBottomB(maxDepth)
    if topB is None:
        topB = getTopB()
    bs = np.linspace(bottomB, topB, num=initialSun.zs.size)
    initialSun.bs = bs
    initialSun.allVariables["bs"] = bs

    return initialSun


def mockupDataSetterUpper(zLength: int = 10) -> dataStructure.SingleTimeDatapoint:
    """
    mockup data setter upper that just makes bunch of weird datapoints instead of the pressures and other datapoints of length zLength
    """
    ones = np.arange(zLength)
    maxdepth = 4
    zs = np.linspace(0, maxdepth, num=zLength)

    toReturn = dataStructure.SingleTimeDatapoint(
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
