#!/usr/bin/env python3
from dataStructure import SingleTimeDatapoint
import boundaryConditions as bcs
from gravity import g
import numpy as np
from scipy.sparse import diags as scipyDiagsMatrix
from scipy.sparse.linalg import spsolve as scipySparseSolve
import constants as c

def getNewTs(currentState:SingleTimeDatapoint, dt:float):
    raise NotImplementedError()


def getNewPs(currentState:SingleTimeDatapoint, dt:float, upflowVelocity:float, totalMagneticFlux:float, bottomExternalPressure:float):
    """
    integrates pressure from the assumption of hydrostatic equilibrium (eq 6)
    dp/dz = rho(p(z), T(z)) g(z)
    """
    bottomPressure = bcs.getBottomPressure(currentState=currentState, dt=dt, upflowVelocity=upflowVelocity, totalMagneticFlux=totalMagneticFlux, bottomExternalPressure=bottomExternalPressure)
    # g(z)
    
    raise NotImplementedError()

def getNewYs(innerPs, outerPs, totalMagneticFlux):
    """
    solves differential equation 5 to get y = sqrt(B) = y(z)
    Φ/2π d²y/dz² y = y⁴ - 2μ₀ (pₑ - pᵢ)
    """
    raise NotImplementedError()


def oldYSolver(innerPs: np.ndarray, outerPs: np.ndarray, totalMagneticFlux:float, yGuess : np.ndarray, tolerance : float = 1e-5) -> np.ndarray:
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
        │             └──────────────┐   │
        │Solve(Ay'=y-(y^3-2mu/y(p-p))│   │
        ├──────────────────────┬─────┘   │
    ┌──►│y_new = y + corrFac*y'│         │
    │   ├──────────────────────┴──────┐  │
    │   │try Ay'  try  y'^3-2mu/y'p-p)│  │
    │   └─┬─────────────┬─────────────┘  │
    │     │ Worse?      │Better?         │
    │   ┌─▼──────────┐  └────────────────┘
    └───┤corrFac*=0.5│
        └────────────┘
    """

    numberOfZSteps = yGuess.size
    matrixOfSecondDifferences = 0.5*totalMagneticFlux/np.pi * scipyDiagsMatrix(
        [1, -2, 1], [-1, 0, 1], shape=(numberOfZSteps, numberOfZSteps))  # ye idk what this error is about, it works


    P_eMinusP_i = outerPs - innerPs

    def exactRightHandSide(y:np.ndarray)->np.ndarray:
        return y*y*y - 2*c.mu0*P_eMinusP_i/y
    
    rightSide = exactRightHandSide(yGuess)
    guessRightSide = matrixOfSecondDifferences@yGuess

    guessError = np.linalg.norm(rightSide - guessRightSide)
    #TODO - comment this section properly 
    correctionFactor = 1
    while guessError > tolerance:

        changeInbetweenSteps = scipySparseSolve(matrixOfSecondDifferences, rightSide-guessRightSide)
        changeInbetweenSteps[0] = 0
        changeInbetweenSteps[-1] = 0

        newYGuess = yGuess + correctionFactor*changeInbetweenSteps

        newRightSide = exactRightHandSide(newYGuess)
        newGuessRightSide = matrixOfSecondDifferences@newYGuess

        if np.linalg.norm(newGuessRightSide - guessRightSide) < guessError:
            yGuess = newYGuess
            rightSide = newRightSide
            guessRightSide = newGuessRightSide
            
            guessError = np.linalg.norm(rightSide - guessRightSide)

            correctionFactor *= 1.1
        else:
            correctionFactor*=0.5

    return yGuess
