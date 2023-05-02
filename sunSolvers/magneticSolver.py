#!/usr/bin/env python3
import numpy as np
import logging
L = logging.getLogger(__name__)

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

    L.warn("Ur using the old Y solver based on Bárta's work")

    numberOfZSteps = zs.size
    stepsizes = zs[1:] - zs[:-1]

    # setup of matrix of centered differences; after semicolon is the part that takes care of border elements: one sided differencees



    

    raise NotImplementedError("Hey u sure this is mathematically correct?")
    matrixOfSecondDifferences = 0.5 * totalMagneticFlux / np.pi * centeredDifferences@centeredDifferences #

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

