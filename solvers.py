#!/usr/bin/env python3
from dataStructure import SingleTimeDatapoint
import boundaryConditions as bcs
from gravity import g
import numpy as np
from scipy.sparse import diags as scipyDiagsMatrix
from scipy.sparse.linalg import spsolve as scipySparseSolve
import constants as c
import warnings


def firstOrderTSolver(currentState: SingleTimeDatapoint, dt: float) -> np.ndarray:
    """
    solves the T equation 
    dT/dt = -1/(rho cp) d/dz (F_rad + F_conv)
    using scipys ODE integrator
    """

    zs = currentState.zs
    Ts = currentState.temperatures
    rhos = currentState.rhos
    F_cons = currentState.F_cons
    F_rads = currentState.F_rads
    cps = currentState.cps

    stepsizes = zs[1:] - zs[:-1]
    dF_rads = np.concatenate( (
        [(F_rads[1]-F_rads[0])/stepsizes[0]],
        (F_rads[2:]-F_rads[:-2])/(stepsizes[:-1] + stepsizes[1:]),
        [(F_rads[-1]-F_rads[-2])/stepsizes[-1]]) )
    dF_cons = np.concatenate( (
        [(F_cons[1]-F_cons[0])/stepsizes[0]],
        (F_cons[2:]-F_cons[:-2])/(stepsizes[:-1] + stepsizes[1:]),
        [(F_cons[-1]-F_cons[-2])/stepsizes[-1]]) )

    dTdt = -1/(rhos*cps) * (dF_cons + dF_rads)

    warnings.warn("This is just first order solver")
    return Ts + dt*dTdt

def oldTSolver(currentState: SingleTimeDatapoint, dt: float) -> np.ndarray:
    """
    solves the T equation the same way Bárta did
    by solving 
    Mb = 0
    where M is a tridiag matrix
    b is 
    """
    warnings.warn("Ur using the old T solver based on Bárta's work")
    zs = currentState.zs
    Ts = currentState.temperatures
    Ps = currentState.pressures
    rhos = currentState.rhos
    F_cons = currentState.F_cons
    F_rads = currentState.F_rads
    cps = currentState.cps
    opacities = ???

    stepsizes = zs[1:] - zs[:-1]

    B = 16*c.SteffanBoltzmann*Ts*Ts*Ts/(3*opacities*rhos) # -coefficient in front of dT/dz in F_rad, i.e. F_rad = -B dT/dz

    A = np.concatenate( (
        [(B[1]-B[0])/stepsizes[0]],
        (B[2:]-B[:-2])/(stepsizes[:-1] + stepsizes[1:]),
        [(B[-1]-B[-2])/stepsizes[-1]]) )    # this is an array of centered differences used to approximate first derivatives
    

def getNewPs(
    currentState: SingleTimeDatapoint,
    dt: float,
    upflowVelocity: float,
    totalMagneticFlux: float,
    bottomExternalPressure: float,
):
    """
    integrates pressure from the assumption of hydrostatic equilibrium (eq 6)
    dp/dz = rho(p(z), T(z)) g(z)
    """
    bottomPressure = bcs.getBottomPressure(
        currentState=currentState,
        dt=dt,
        upflowVelocity=upflowVelocity,
        totalMagneticFlux=totalMagneticFlux,
        bottomExternalPressure=bottomExternalPressure,
    )
    # g(z)

    raise NotImplementedError()


def getNewYs(innerPs, outerPs, totalMagneticFlux):
    """
    solves differential equation 5 to get y = sqrt(B) = y(z)
    Φ/2π d²y/dz² y = y⁴ - 2μ₀ (pₑ - pᵢ)
    """
    raise NotImplementedError()


def oldYSolver(
    zs: np.ndarray,
    innerPs: np.ndarray,
    outerPs: np.ndarray,
    totalMagneticFlux: float,
    yGuess: np.ndarray,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """
    solver of the differential equation taken from the bc thesis using tridiagonal matrix
    this may be very slow cuz it's of the first order

    it tries to solve the equation
    A y_n+1 = b_n = y^3_n - 2mu/y(p_e-p_i)

    it works according to this flowchart

                   ┌─────────────────────┐
                   │                     │
              ┌────▼─┐                   │
              │get ys│                   │
        ┌─────┴┬─┬───┴──────────────┐    │
        │try Ay│ │try y^3-2mu/y(p-p)│    │
        ├──────┴─┴────┬─────────────┘    │
        │correction = │                  │
        │   = y' =    └──────────────┐   │
        │Solve(Ay'=y-(y^3-2mu/y(p-p))│   │
        ├──────────────────────┬─────┘   │
    ┌──►│y_new = y + corrFac*y'│         │
    │   ├───────┬─┬────────────┴──────┐  │
    │   │try Ayn│ │try yn^3-2mu/ynp-p)│  │
    │   └─┬─────┴─┴─────┬─────────────┘  │
    │     │ Worse?      │Better?         │
    │   ┌─▼──────────┐  └────────────────┘
    └───┤corrFac*=0.5│
        └────────────┘
    """

    warnings.warn("Ur using the old Y solver based on Bárta's work")

    numberOfZSteps = zs.size
    stepsizes = zs[:-1] - zs[1:]

    matrixOfSecondDifferences = (
        0.5
        * totalMagneticFlux
        / np.pi
        * scipyDiagsMatrix(
            [1, -2, 1], [-1, 0, 1], shape=(numberOfZSteps, numberOfZSteps)
        )
    )  # type: ignore # ye idk why vscode says that array of ints is an error, it's in the documentation and it literally works: that's why the type: ignore
    # FIXME - add division by step size
    raise NotImplementedError(
        "division by stepsizes is missing; is the second derivative in the thesis just wrong?"
    )
    P_eMinusP_i = outerPs - innerPs

    def exactRightHandSide(y: np.ndarray) -> np.ndarray:
        return y * y * y - 2 * c.mu0 * P_eMinusP_i / y

    rightSide = exactRightHandSide(yGuess)
    guessRightSide = matrixOfSecondDifferences @ yGuess

    guessError = np.linalg.norm(rightSide - guessRightSide)

    correctionFactor = 1
    while guessError > tolerance:

        changeInbetweenSteps = scipySparseSolve(
            matrixOfSecondDifferences, rightSide - guessRightSide
        )
        changeInbetweenSteps[0] = 0
        changeInbetweenSteps[-1] = 0

        newYGuess = yGuess + correctionFactor * changeInbetweenSteps

        newRightSide = exactRightHandSide(newYGuess)
        newGuessRightSide = matrixOfSecondDifferences @ newYGuess

        newGuessError = np.linalg.norm(newGuessRightSide - guessRightSide)

        if newGuessError < guessError:
            yGuess = newYGuess
            rightSide = newRightSide
            guessRightSide = newGuessRightSide
            guessError = newGuessError

            correctionFactor *= 1.1
        else:
            correctionFactor *= 0.5

    return yGuess


def main():
    """test code for this file"""
    length = 5
    zs = np.linspace(0, 50, length)
    innerPs = np.ones(length) * 3
    outerPs = np.ones(length) * 5
    totalMagneticflux = 4
    yGuess = zs[:]
    oldYSolver(zs, innerPs, outerPs, totalMagneticflux, yGuess)


if __name__ == "__main__":
    main()
