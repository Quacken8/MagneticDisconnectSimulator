#!/usr/bin/env python3
import numpy as np
from stateEquationsPT import StateEquationInterface
from scipy import sparse
from scipy.sparse import linalg
import constants as c
from typing import Type
import loggingConfig
import logging

L = loggingConfig.configureLogging(logging.INFO, __name__)
from dataHandling.dataStructure import SingleTimeDatapoint
from typing import Callable
import gravity
from sunSolvers.handySolverStuff import centralDifferencesMatrix


def oldTSolver(
    currentState: SingleTimeDatapoint,
    dt: float,
    StateEq: Type[StateEquationInterface],
    opacityFunction: Callable[[np.ndarray, np.ndarray], np.ndarray],
    surfaceTemperature: float,
    convectiveAlpha: float,
) -> np.ndarray:
    """
    solves the T equation
    dT/dt = -1/(rho cp) d/dz (F_rad + F_conv)
    the same way Bárta did
    by solving
    M·v = b (eq 4.17 in Bárta)
    where M is a tridiag matrix
    v are the temperatures at different zs at time t+dt
    b is a vector containing derivatives of F_conv and rho cp T/dt at time t

    requires that the grid is equidistant. If it's not, it will be made that way

    returns the new temperatures

    here's the derivation of the matrix equation:
    dT/dt = -1/(rho cp) d/dz (F_rad + F_conv)
    rho cp dT/dt = -(d/dz (F_rad) + d/dz (F_conv))
    rho cp dT/dt = -(d/dz (A dT/dz) + d/dz (F_conv))
    rho cp dT/dt = -(dA/dz dT/dz + A d^2T/dz^2 + d/dz (F_conv))

    now if we discretize the equation, making it implicit, denoting the temperatures
    at time t with T and at time t+dt with T', we get

    rho cp (T' - T)/dt = -((dA/dz dT'/dz + A d^2T'/dz^2 + d/dz (F_conv))

    now we can discretize the spacial derivatives using centered differences
    and putting together the terms with Ts at the same depth we get

    mu_i T'_{i-1} + lambda_i T'_i + nu_i T'_{i+1} = d/dz (F_conv) + rho cp T/dt
    M·T' = b
    M·v = b

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
    cps = StateEq.Cp(Ts, Ps)
    m_zs = gravity.massBelowZ(zs)
    gs = gravity.g(zs)
    f_cons = StateEq.f_con(
        convectiveAlpha=convectiveAlpha,
        temperature=Ts,
        pressure=Ps,
        opacity=opacities,
        massBelowZ=m_zs,
        gravitationalAcceleration=gs,
    )

    bottomH = StateEq.pressureScaleHeight(Ts[-1], Ps[-1], gravity.g(zs[-1]))
    bottomNablaAd = StateEq.adiabaticLogGradient(Ts[-1], Ps[-1])
    bottomAdiabaticT = Ts[-1] * (
        1 + bottomNablaAd / bottomH * dz
    )  # this is the temperature at the bottom of the domain which is supposed to be the result of adiabatic gradient as said in Bárta and Schüssler & Rempel (2005)

    centeredDifferencesM = centralDifferencesMatrix(zs)

    A = (
        16 * c.SteffanBoltzmann * Ts * Ts * Ts / (3 * opacities * rhos)
    )  # -coefficient in front of dT/dz in F_rad, i.e. F_rad = -A dT/dz

    # now that teverything is ready prepare the matrix equation M·v = b

    lambdas = -2 * A / (dz * dz) - rhos * cps / dt
    gradA = centeredDifferencesM.dot(A)
    mus = gradA - A / (dz * dz)
    nus = -gradA - A / (dz * dz)
    M = sparse.diags([mus[1:], lambdas, nus[:-1]], [-1, 0, 1], shape=(len(zs), len(zs))).tocsr()  # type: ignore

    fs = (
        centeredDifferencesM.dot(f_cons) + rhos * cps * Ts / dt
    )  # TODO rederive the equation
    bs = fs[:]
    bs[0] -= (
        mus[0] * surfaceTemperature
    )  # TODO ye here it doesnt make sense either, Bárta didnt specify the index, but it should be 0, right?
    bs[-1] -= nus[-1] * bottomAdiabaticT

    # solve the matrix equation

    Ts = linalg.spsolve(M, bs)

    return Ts


def simpleTSolver(
    currentState: SingleTimeDatapoint,
    dt: float,
    StateEq: Type[StateEquationInterface],
    opacityFunction: Callable[[np.ndarray, np.ndarray], np.ndarray],
    surfaceTemperature: float,
    convectiveAlpha: float,
) -> np.ndarray:
    """
    First order forward Euler solver for the temperature equation
    """
    L.warn("Using simple temperature solver")
    dTdt = rightHandSideOfTEq(
        convectiveAlpha=convectiveAlpha,
        zs=currentState.zs,
        temperatures=currentState.temperatures,
        pressures=currentState.pressures,
        StateEq=StateEq,
        opacityFunction=opacityFunction,
    )
    newTs = currentState.temperatures + dt * dTdt

    newTs[0] = surfaceTemperature

    return newTs


def rightHandSideOfTEq(
    convectiveAlpha: float,
    zs: np.ndarray,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    StateEq: Type[StateEquationInterface],
    opacityFunction: Callable[[np.ndarray, np.ndarray], np.ndarray],
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

    cps = StateEq.Cp(temperatures, pressures)
    rhos = StateEq.density(temperatures, pressures)
    opacity = opacityFunction(temperatures, pressures)
    Tgrad = np.gradient(temperatures, zs)
    massBelowZ = gravity.massBelowZ(zs)
    gravitationalAcceleration = gravity.g(zs)

    # FplusFs = 4*c.aRad*c.speedOfLight*c.G/3 * m * T4/(kappas* Ps * r*r) * nablaRads
    frads = StateEq.f_rad(
        temperatures, pressures, opacity=opacity, Tgrad=Tgrad
    )
    fcons = StateEq.f_con(
        convectiveAlpha=convectiveAlpha,
        temperature=temperatures,
        pressure=pressures,
        opacity=opacity,
        massBelowZ=massBelowZ,
        gravitationalAcceleration=gravitationalAcceleration,
    )
    FplusFs = frads + fcons
    dFplusFdz = np.gradient(FplusFs, zs)

    return -dFplusFdz / (rhos * cps)


def main():
    pass


if __name__ == "__main__":
    main()
