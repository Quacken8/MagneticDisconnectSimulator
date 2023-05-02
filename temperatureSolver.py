#!/usr/bin/env python3
import numpy as np
from stateEquationsPT import StateEquationInterface, F_con, F_rad
from scipy.linalg import spmatrix
from scipy.sparse import diags
import logging
L = logging.getLogger(__name__)
from dataStructure import SingleTimeDatapoint
from typing import Callable
from gravity import g, massBelowZ


#region matrices of differences

def centralDifferencesMatrix(steps: np.ndarray) -> spmatrix:
    """
    Given an array of steps (i.e. x_i) at which a function f is evaluated (i.e. f_i) returns a tridiagonal matrix such that Af are the central differences of f
    On edges the differences are approximated by one sided differences
    
    for uniform stepsize h

              ( -1   1   0   0   0  0 )
              (-1/2  0  1/2  0   0  0 )
    A = 1/h * (  0 -1/2  0  1/2  0  0 )
              (  0   0 -1/2  0  1/2 0 )
              (  0   0   0   0  -1  1 )

    """
    stepsizes = np.diff(steps)
    numberOfSteps = len(steps)

    underDiag = -1/(stepsizes[:-1]+stepsizes[1:]); underDiag = np.append(underDiag, -1/stepsizes[-1])
    overDiag = 1/(stepsizes[:-1]+stepsizes[1:]); overDiag = np.insert(overDiag, 0, 1/stepsizes[0])
    diag = np.zeros(numberOfSteps); diag[0] = -1/stepsizes[0]; diag[-1] = 1/stepsizes[-1]

    centeredDifferences = diags(
            [underDiag, diag, overDiag], [-1, 0, 1], shape=(numberOfSteps, numberOfSteps) # type: ignore I'm unsure why the [-1, 0, 1] is throwing an error, this is literally the example from the documentation
        )
    return centeredDifferences

def forwardDifferencesMatrix(steps: np.ndarray) -> spmatrix:
    """
    Given an array of steps (i.e. x_i) at which a function f is evaluated (i.e. f_i) returns a tridiagonal matrix such that Af are the forward differences of f
    On edges the differences are approximated by one sided differences
    
    for uniform stepsize h

              ( -1   1   0   0   0  )
              (  0  -1   1   0   0  )
    A = 1/h * (  0   0  -1   1   0  )
              (  0   0   0  -1   1  )
              (  0   0   0  -1   1  )
    """

    dx = np.diff(steps)
    numberOfSteps = len(steps)

    underDiag = np.zeros(numberOfSteps-1); underDiag[-1] = -1/dx[-1]
    diag = np.append(-1/dx, 1/dx[-1])
    overDiag = 1/dx

    forwardDifferences = diags([underDiag, diag, overDiag], [-1, 0, 1], shape=(numberOfSteps, numberOfSteps)) # type: ignore I'm unsure why the [-1, 0, 1] is throwing an error, this is literally the example from the documentation
    return forwardDifferences

def backwardDifferencesMatrix(steps: np.ndarray) -> spmatrix:
    """
    Given an array of steps (i.e. x_i) at which a function f is evaluated (i.e. f_i) returns a tridiagonal matrix such that Af are the backward differences of f
    On edges the differences are approximated by one sided differences
    
    for uniform stepsize h

              ( -1   1   0   0   0  )
              ( -1   1   0   0   0  )
    A = 1/h * (  0  -1   1   0   0  )
              (  0   0  -1   1   0  )
              (  0   0   0  -1   1  )
    """
    dx = np.diff(steps)
    numberOfSteps = len(steps)

    overDiag = np.zeros(numberOfSteps-1); overDiag[0] = 1/dx[0]
    diag = np.insert(1/dx, 0, -1/dx[0])
    underDiag = -1/dx

    backwardDifferences = diags([underDiag, diag, overDiag], [-1, 0, 1], shape=(numberOfSteps, numberOfSteps)) # type: ignore I'm unsure why the [-1, 0, 1] is throwing an error, this is literally the example from the documentation
    return backwardDifferences

def secondCentralDifferencesMatrix(steps: np.ndarray) -> spmatrix:
    """Given an array of steps (i.e. x_i) at which a function f is evaluated (i.e. f_i) returns a tridiagonal matrix such that Af are the second central differences of f

    Args:
        steps (np.ndarray): _description_

    Returns:
        spmatrix: _description_
    """
    forward = forwardDifferencesMatrix(steps)
    backward = backwardDifferencesMatrix(steps)

    secondCentral = forward.dot(backward)
    # becuase of the nature of the one sided differences
    # and their behaviour at the edges
    # the first row gets obliterated
    secondCentral[0, :] = secondCentral[1, :]  
    
    return secondCentral

# endregion

def firstOrderTSolver(currentState: SingleTimeDatapoint, dt: float, StateEq: StateEquationInterface) -> np.ndarray:
    """
    solves the T equation 
    dT/dt = -1/(rho cp) d/dz (F_rad + F_conv)
    in the first order
    """

    zs = currentState.zs
    Ts = currentState.temperatures
    Ps = currentState.pressures

    F_cons = F_con(Ts, Ps)
    F_rads = F_rad(Ts, Ps)
    cps = StateEq.cp(Ts, Ps)

    stepsizes = np.diff(zs)
    dF_rads = np.concatenate( (
        [(F_rads[1]-F_rads[0])/stepsizes[0]],
        (F_rads[2:]-F_rads[:-2])/(stepsizes[:-1] + stepsizes[1:]),
        [(F_rads[-1]-F_rads[-2])/stepsizes[-1]]) )
    dF_cons = np.concatenate( (
        [(F_cons[1]-F_cons[0])/stepsizes[0]],
        (F_cons[2:]-F_cons[:-2])/(stepsizes[:-1] + stepsizes[1:]),
        [(F_cons[-1]-F_cons[-2])/stepsizes[-1]]) )

    dTdt = -1/(rhos*cps) * (dF_cons + dF_rads)

    L.warn("This is just first order solver")
    return Ts + dt*dTdt

def oldTSolver(currentState: SingleTimeDatapoint, dt: float) -> np.ndarray:
    """
    solves the T equation the same way Bárta did
    by solving 
    Mb = 0
    where M is a tridiag matrix
    b is 
    """
    L.warn("Ur using the old T solver based on Bárta's work")
    raise NotImplementedError()
    zs = currentState.zs
    Ts = currentState.temperatures
    Ps = currentState.pressures
    rhos = currentState.rhos
    F_cons = currentState.F_cons
    F_rads = currentState.F_rads
    cps = currentState.cps
    opacities = 0.1 # TODO: get opacities

    stepsizes = zs[1:] - zs[:-1]

    B = 16*c.SteffanBoltzmann*Ts*Ts*Ts/(3*opacities*rhos) # -coefficient in front of dT/dz in F_rad, i.e. F_rad = -B dT/dz

    A = np.concatenate( (
        [(B[1]-B[0])/stepsizes[0]],
        (B[2:]-B[:-2])/(stepsizes[:-1] + stepsizes[1:]),
        [(B[-1]-B[-2])/stepsizes[-1]]) )    # this is an array of centered differences used to approximate first derivatives
    
    raise NotImplementedError()

def rightHandSideOfTEq(convectiveAlpha: float, zs: np.ndarray, temperatures: np.ndarray, pressures: np.ndarray, StateEq: StateEquationInterface, opacity:Callable) -> np.ndarray:
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

    cps = StateEq.cp(temperatures, pressures)
    rhos = StateEq.density(temperatures, pressures)
    mus = StateEq.meanMolecularWeight(temperatures, pressures)
    nablaAds = StateEq.adiabaticLogGradient(temperatures, pressures)
    kappas = opacity(temperatures, pressures)
    gs = g(zs)
    massBelowZs = massBelowZ(zs)
    nablaRad = StateEq.radiativeLogGradient(temperatures, pressures, opacity=kappas, massBelowZ=massBelowZs)
    Hps = StateEq.pressureScaleHeight(temperatures, pressures, gravitationalAcceleration=gs)

    FplusFs = F_rad(temperatures, pressures)+F_con(convectiveAlpha = convectiveAlpha, temperature = temperatures, pressure = pressures, meanMolecularWeight=mus, adiabaticGrad=nablaAds, radiativeGrad=nablaRad, c_p=cps, pressureScaleHeight=Hps, opacity=kappas, gravitationalAcceleration=gs) # FIXME this is a nightmare, remake
    dFplusFdz = np.gradient(FplusFs, zs) # TODO this might be slow, so watch out

    return -dFplusFdz/(rhos * cps)


def main():    
    N = 100
    xs = np.sort(np.random.random(N))*2*np.pi

    secondDif = secondCentralDifferencesMatrix(xs)

    f = np.sin(xs)
    expectedDf = -np.sin(xs)
    aprox = secondDif.dot(f)

    df = np.gradient(f, xs)
    ddf = np.gradient(df, xs)
    numpyVersion = ddf

    import matplotlib.pyplot as plt
    plt.plot(xs, expectedDf, label="exact solution of d² sin x/dx²")
    plt.scatter(xs, aprox,      marker = ".", label="second differences δ² sin x/δx²", c = "red")
    plt.scatter(xs, numpyVersion, marker = ".", label="numpy version of second differences", c = "black")
    plt.ylim(-3,3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()