#!/usr/bin/env python3
import numpy as np
from stateEquationsPT import StateEquationInterface, F_con, F_rad
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import constants as c
from typing import Type
import loggingConfig
import logging
L = loggingConfig.configureLogging(logging.INFO, __name__)
from dataHandling.dataStructure import SingleTimeDatapoint
from typing import Callable
from gravity import g, massBelowZ
from sunSolvers.handySolverStuff import centralDifferencesMatrix


def oldTSolver(
    currentState: SingleTimeDatapoint,
    dt: float,
    StateEq: Type[StateEquationInterface],
    opacityFunction: Callable[[np.ndarray, np.ndarray], np.ndarray],
    surfaceTemperature: float,
) -> np.ndarray:
    """
    solves the T equation the same way Bárta did
    by solving
    M·v = b (eq 4.17 in Bárta)
    where M is a tridiag matrix
    v are the temperatures at different zs at time t+dt
    b is a vector containing derivatives of F_conv and rho cp T/dt at time t

    requires that the grid is equidistant. If it's not, it will be made that way

    returns the new temperatures
    """
    zs = currentState.zs
    # make sure the zs are equidistant, otherwise the matrix equation will be wrong
    if not np.allclose(np.diff(zs), np.diff(zs)[0]):
        currentState.regularizeGrid()
    dz = np.diff(zs)[0]

    Ts = currentState.temperatures
    Ps = currentState.pressures

    opacities = opacityFunction(Ts, Ps)
    rhos = StateEq.density(Ts, Ps)
    cps = StateEq.cp(Ts, Ps)
    m_zs = massBelowZ(zs)
    F_cons = F_con(
        convectiveAlpha=0,
        temperature=Ts,
        pressure=Ps,
        meanMolecularWeight=StateEq.meanMolecularWeight(Ts, Ps),
        adiabaticGrad=StateEq.adiabaticLogGradient(Ts, Ps),
        radiativeGrad=StateEq.radiativeLogGradient(Ts, Ps, m_zs, opacities),
        c_p=cps,
        pressureScaleHeight=StateEq.pressureScaleHeight(Ts, Ps, g(zs)),
        opacity=opacities,
        gravitationalAcceleration=g(zs),
    )

    bottomH = StateEq.pressureScaleHeight(Ts[-1], Ps[-1], g(zs[-1]))
    bottomNablaAd = StateEq.adiabaticLogGradient(Ts[-1], Ps[-1])
    bottomAdiabaticT = Ts[-1] * (
        1 + bottomNablaAd / bottomH * dz
    )  # this is the temperature at the bottom of the domain which is supposed to be the result of adiabatic gradient as said in Bárta and Schüssler & Rempel (2005)

    centeredDifferencesM = centralDifferencesMatrix(zs)

    A = (
        16 * c.SteffanBoltzmann * Ts * Ts * Ts / (3 * opacities * rhos)
    )  # -coefficient in front of dT/dz in F_rad, i.e. F_rad = -A dT/dz

    gradA = centeredDifferencesM.dot(
        A
    )  # this is an array of centered differences used to approximate first derivatives

    # now that teverything is ready prepare the matrix equation M·v = b

    lambdas = 2 * A / (dz * dz)
    mus = gradA - 0.5 * lambdas
    nus = -gradA - 0.5 * lambdas
    M = diags([mus, lambdas, nus], [-1, 0, 1], shape=(len(zs), len(zs)))  # type: ignore

    fs = (
        centeredDifferencesM.dot(F_cons) + 2 * rhos * cps * Ts / dt
    )  # TODO rederive the equation just to be sure that 2 is supposed to be there
    bs = fs[:]
    bs[0] -= (
        mus[0] * surfaceTemperature
    )  # TODO ye here it doesnt make sense either, Bárta didnt specify the index, but it should be 0, right?
    bs[-1] -= nus[-1] * bottomAdiabaticT

    # solve the matrix equation

    Ts = spsolve(M, bs)

    return Ts


def rightHandSideOfTEq(
    convectiveAlpha: float,
    zs: np.ndarray,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    StateEq: StateEquationInterface,
    opacity: Callable,
) -> np.ndarray:
    """
    right hand side of this equation from Schüssler & Rempel (2005) (eq. 8)
    how temperature changes in time at a fixed depth z
    dT/dt = -1/(rho cp) d/dz (F_rad + F_conv)

    Parameters:
    zs: np.ndarray [m] depths
    temperatures: np.ndarray [K] at depths zs
    pressures: np.ndarray [Pa] at depths zs
    """
    # NOTE this might be confusing since the zs change from one time to another
    # TODO check if you'll even need this, maybe bárta's way is better

    cps = StateEq.cp(temperatures, pressures)
    rhos = StateEq.density(temperatures, pressures)
    mus = StateEq.meanMolecularWeight(temperatures, pressures)
    nablaAds = StateEq.adiabaticLogGradient(temperatures, pressures)
    kappas = opacity(temperatures, pressures)
    gs = g(zs)
    massBelowZs = massBelowZ(zs)
    nablaRad = StateEq.radiativeLogGradient(
        temperatures, pressures, opacity=kappas, massBelowZ=massBelowZs
    )
    Hps = StateEq.pressureScaleHeight(
        temperatures, pressures, gravitationalAcceleration=gs
    )

    FplusFs = F_rad(temperatures, pressures) + F_con(
        convectiveAlpha=convectiveAlpha,
        temperature=temperatures,
        pressure=pressures,
        meanMolecularWeight=mus,
        adiabaticGrad=nablaAds,
        radiativeGrad=nablaRad,
        c_p=cps,
        pressureScaleHeight=Hps,
        opacity=kappas,
        gravitationalAcceleration=gs,
    )  # FIXME this is a nightmare, remake
    dFplusFdz = np.gradient(FplusFs, zs)

    return -dFplusFdz / (rhos * cps)


def main():
    pass


if __name__ == "__main__":
    main()
