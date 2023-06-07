#!/usr/bin/env python3
from math import sin
from dataHandling.dataStructure import SingleTimeDatapoint
from gravity import g, massBelowZ
import numpy as np
from typing import Callable, Type
from scipy.integrate import ode, odeint, solve_ivp
from scipy.interpolate import interp1d
import constants as c
from dataHandling.modelS import loadModelS
from stateEquationsPT import StateEquationInterface
import loggingConfig
import logging

L = loggingConfig.configureLogging(logging.INFO, __name__)


def integrateHydrostaticEquilibriumAndTemperatureGradient(
    StateEq: Type[StateEquationInterface],
    opacityFunction: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dlnP: float,
    lnBoundaryPressure: float,
    boundaryTemperature: float,
    initialZ: float,
    finalZ: float,
    regularizeGrid: bool = False,
) -> SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to integrated pressure according to hydrostatic equilibrium in the Sun where both magnetic fields and inflow of material play a role FIXME is this even true
    solves the set of ODEs
    dz/dlnP   = H(z,T,P)
    dlnT/dlnP = ∇(z,T,P)

    ----------
    Parameters
    ----------
    stateEq: a class with static functions that return the thermodynamic quantities as a function of temperature and pressure; see StateEquations.py for an example

    opacity: a function that returns the opacity as a function of pressure and temperature

    dlogP : [Pa] step in pressure gradient by which the integration happens

    logBoundaryPressure : [Pa] boundary condition of surface or bottom pressure

    boundaryTemperature : [K] boundary condition of surface or bottom temperature

    maxDepth : [m] depth to which integrate

    guessTheZRange : if True, will estimate what pressure is at maxDepth using model S, adds a bit of padding (20 %) to it just ot be sure
    and uses scipy in a bit faster.
    You don't get the exactly correct z range, but it is ~3 times faster

    homogenizeGrid : if True, the grid will be made equidistant by linear interpolation at the end
    """

    def setOfODEs(lnP: float, zlnTArray: np.ndarray) -> np.ndarray:
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

        H = StateEq.pressureScaleHeight(
            temperature=T, pressure=P, gravitationalAcceleration=gravAcc
        )
        nablaAd = StateEq.adiabaticLogGradient(temperature=T, pressure=P)
        kappa = opacityFunction(P, T)
        nablaRad = StateEq.radiativeLogGradient(
            temperature=T, pressure=P, massBelowZ=m_z, opacity=kappa
        )
        nabla = np.minimum(nablaAd, nablaRad)

        return np.array([H, nabla])

    # initial conditions
    currentZlnTValues = np.array([initialZ, np.log(boundaryTemperature)])

    # get the guess of the integration domain

    paddingFactor = 0.05  # i.e. will, just to be sure, integrate to 102 % of the ln(pressure) expected at maxDepth
    modelS = loadModelS()
    modelPs = modelS.pressures
    modelZs = modelS.zs
    finalPGuess = np.interp(finalZ, modelZs, modelPs)

    # get rid of these they might be big
    del modelS, modelPs, modelZs

    bottomUp = (
        initialZ > finalZ
    )  # if we are going from bottom up to have correct padding we need to go to (pressure - padding)
    finalLnPGuess = np.log(finalPGuess) * (
        1 + (-paddingFactor if bottomUp else paddingFactor)
    )

    # set up the integration itself
    sunLnPs = np.linspace(
        lnBoundaryPressure,
        finalLnPGuess,
        int(np.abs((finalLnPGuess - lnBoundaryPressure)) / dlnP),
    )
    if bottomUp:
        sunLnPs = sunLnPs[::-1]

    sunZs, sunLnTs = odeint(
        func=setOfODEs,
        y0=currentZlnTValues,
        t=sunLnPs,
        tfirst=True,
    ).T

    sunPs = np.exp(sunLnPs)
    sunTs = np.exp(sunLnTs)

    sun = SingleTimeDatapoint(
        zs=np.array(sunZs),
        temperatures=np.array(sunTs),
        pressures=np.array(sunPs),
    )
    if regularizeGrid:
        sun.regularizeGrid()
    return sun


def integrateAdiabaticHydrostaticEquilibrium(
    StateEq: Type[StateEquationInterface],
    dlnP: float,
    lnBoundaryPressure: float,
    boundaryTemperature: float,
    initialZ: float,
    finalZ: float,
    regularizeGrid: bool = False,
) -> SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to integrated pressure according to hydrostatic equilibrium in the Sun where both magnetic fields and inflow of material play a role FIXME is this even true
    solves the set of ODEs
    dz/dlnP   = H(z,T,P)
    dlnT/dlnP = ∇*ad*(z,T,P)

    ----------
    Parameters
    ----------
    stateEq: a class with static functions that return the thermodynamic quantities as a function of temperature and pressure; see StateEquations.py for an example

    opacity: a function that returns the opacity as a function of pressure and temperature

    dlogP : [Pa] step in pressure gradient by which the integration happens

    logBoundaryPressure : [Pa] boundary condition of surface or bottom pressure

    boundaryTemperature : [K] boundary condition of surface or bottom temperature

    maxDepth : [m] depth to which integrate

    guessTheZRange : if True, will estimate what pressure is at maxDepth using model S, adds a bit of padding (20 %) to it just ot be sure
    and uses scipy in a bit faster.
    You don't get the exactly correct z range, but it is ~3 times faster

    homogenizeGrid : if True, the grid will be made equidistant by linear interpolation at the end
    """

    def setOfODEs(lnP: float, zlnTArray: np.ndarray) -> np.ndarray:
        """
        the set of ODEs that are derived from hydrostatic equilibrium
        dz/dlnP   = H(z,T,P)
        dlnT/dlnP = ∇ad(z,T,P)
        """
        z = zlnTArray[0]
        T = np.exp(zlnTArray[1])
        P = np.exp(lnP)
        gravAcc = np.array(g(z))

        H = StateEq.pressureScaleHeight(
            temperature=T, pressure=P, gravitationalAcceleration=gravAcc
        )
        nablaAd = StateEq.adiabaticLogGradient(temperature=T, pressure=P)

        return np.array([H, nablaAd])

    # initial conditions
    currentZlnTValues = np.array([initialZ, np.log(boundaryTemperature)])

    # get the guess of the integration domain

    paddingFactor = 0.05  # i.e. will, just to be sure, integrate to 120 % of the ln(pressure) expected at maxDepth
    modelS = loadModelS()
    modelPs = modelS.pressures
    modelZs = modelS.zs
    finalPGuess = np.interp(finalZ, modelZs, modelPs)

    # get rid of these they might be big
    del modelS, modelPs, modelZs

    bottomUp = (
        initialZ > finalZ
    )  # if we are going from bottom up to have correct padding we need to go to (pressure - padding)
    finalLnPGuess = np.log(finalPGuess) * (
        1 + (-paddingFactor if bottomUp else paddingFactor)
    )

    # set up the integration itself
    sunLnPs = np.linspace(
        lnBoundaryPressure,
        finalLnPGuess,
        int(np.abs((finalLnPGuess - lnBoundaryPressure)) / dlnP),
    )
    if bottomUp:
        sunLnPs = sunLnPs[::-1]

    sunZs, sunLnTs = odeint(
        func=setOfODEs,
        y0=currentZlnTValues,
        t=sunLnPs,
        tfirst=True,
    ).T

    sunPs = np.exp(sunLnPs)
    sunTs = np.exp(sunLnTs)

    sun = SingleTimeDatapoint(
        zs=np.array(sunZs),
        temperatures=np.array(sunTs),
        pressures=np.array(sunPs),
    )
    if regularizeGrid:
        sun.regularizeGrid()
    return sun


def integrateHydrostaticEquilibrium(
    StateEq: Type[StateEquationInterface],
    referenceTs: np.ndarray,
    referenceZs: np.ndarray,
    dlnP: float,
    lnBoundaryPressure: float,
    initialZ: float,
    finalZ: float,
    interpolableYs: np.ndarray | None = None,
    regularizeGrid: bool = False,
) -> SingleTimeDatapoint:
    """
    Integrates the hydrostatic equilibrium equation d z/dlnP = H if temperatures at zs are known

    Args:
        StateEq (Type[StateEquationInterface]): State equation to use in a form of a static class
        dlnP (float): step in ln(pressure) by which to integrate
        lnBoundaryPressure (float): natural log of the boundary pressure
        initialZ (float): depth at which to start the integration and also at which the boundary condition is set
        finalZ (float): final depth to which to integrate
        regularizeGrid (bool, optional): if True will return results on regular grid in zs. Defaults to False.
        interpolableYs (np.ndarray, optional): if not None, will interpolate these values to get Bs on the new z grid. Defaults to None.

    Returns:
        SingleTimeDatapoint: datapoint with calculated zs, pressures but interpolated temperatures
    """
    assert len(referenceTs) == len(
        referenceZs
    ), "temperatures and zs must have the same length"

    def rightHandSide(lnP: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        right hand side of the hydrostatic equilibrium equation dz/dlnP = H
        """
        pressure = np.exp(lnP)
        temperature = np.interp(z, referenceZs, referenceTs)
        gravAcc = g(z)
        H = StateEq.pressureScaleHeight(
            temperature=temperature,
            pressure=pressure,
            gravitationalAcceleration=gravAcc,
        )
        return H

    # guess the z range
    paddingFactor = 0.05  # i.e. will, just to be sure, integrate to 120 % of the ln(pressure) expected at maxDepth
    modelS = loadModelS()
    modelPs = modelS.pressures
    modelZs = modelS.zs
    finalPGuess = np.interp(finalZ, modelZs, modelPs)

    # integrate it with scipy
    bottomUp = (
        initialZ > finalZ
    )  # if we are going from bottom up to have correct padding we need to go to (pressure - padding)
    finalLnPGuess = np.log(finalPGuess) * (
        1 + (-paddingFactor if bottomUp else paddingFactor)
    )
    sunLnPs = np.linspace(
        lnBoundaryPressure,
        finalLnPGuess,
        int(np.abs((finalLnPGuess - lnBoundaryPressure)) / dlnP),
    )
    if bottomUp:
        sunLnPs = sunLnPs[::-1]

    # sunZs = odeint(func=rightHandSide, y0=initialZ, t=sunLnPs, tfirst=True)[:,0]
    # TODO odeint is faster than solve_ivp, however it seems to be running into some problems. The full output shows some weird values for debugging values... I'm not sure if it's a good idea to use it with the linear interpolation of Ts
    sunZs = solve_ivp(
        fun=rightHandSide,
        y0=np.array([initialZ]),
        t_span=(sunLnPs[0], sunLnPs[-1]),
        t_eval=sunLnPs,
    ).y[0]

    if regularizeGrid:
        regularZs = np.linspace(sunZs[0], sunZs[-1], len(sunZs))
        sunLnPs = np.interp(regularZs, sunZs, sunLnPs)
        sunZs = regularZs

    if interpolableYs is not None:
        sunBs = np.interp(sunZs, referenceZs, interpolableYs * interpolableYs)
    else:
        sunBs = None
    sunPs = np.exp(sunLnPs)
    newSun = SingleTimeDatapoint(
        np.interp(sunZs, referenceZs, referenceTs), sunPs, zs=sunZs, bs=sunBs
    )
    if regularizeGrid:
        newSun.regularizeGrid()
    return newSun


def main():
    """test code for this file"""
    pass


if __name__ == "__main__":
    main()
