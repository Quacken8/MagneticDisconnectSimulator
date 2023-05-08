#!/usr/bin/env python3
import numpy as np
import logging
import constants as c
from scipy.integrate import solve_bvp

L = logging.getLogger(__name__)


def rightHandSideOfYEq(y, innerP, outerP, totalMagneticFlux):
    """
    right hand side of the differential equation 5
    """
    return (y * y * y - 2 * c.mu0 * (outerP - innerP) / y) / (
        totalMagneticFlux / (2 * np.pi)
    )


def setOfODEs(yYgradArray, innerP, outerP, totalMagneticFlux):
    """
    turns the second order diff eq into a set of two first order diff eqs
    by introducing ygrad = dy/dz
    returns an array of [ygrad, ygrad'], i.e. [y', y'']
    """
    y = yYgradArray[0]
    ygrad = yYgradArray[1]

    return np.array([ygrad, rightHandSideOfYEq(y, innerP, outerP, totalMagneticFlux)])


def integrateMagneticEquation(zs, innerPs, outerPs, totalMagneticFlux):
    """
    solves differential equation 5 to get y = sqrt(B) = y(z)
    Φ/2π d²y/dz² y = y⁴ - 2μ₀ (pₑ - pᵢ)
    """

    # boundaryConditions
    bottomPe, bottomPi = (
        outerPs[-1],
        innerPs[-1],
    )  # TODO check if ur taking the correct thingy
    bottomY = np.sqrt(np.sqrt(2 * c.mu0 * (bottomPe - bottomPi)))
    topY = np.sqrt(
        2000 * c.Gauss
    )  # boundary condition used by Schüssler & Rempel (2005)

    def boundaryConditions(yBottom, yTop):
        return np.array([yBottom - bottomY, yTop - topY])

    # initial guess
    # here we assume that linear function
    yGuess = np.linspace(bottomY, topY, zs.size)
    yGradGuess = np.ones(zs.size) * (topY - bottomY) / (zs[-1] - zs[0])
    yYgradGuess = np.array([yGuess, yGradGuess])

    # integration
    ( # we get a bunch of results from the scipy solver
        solutionSpline, # spline of the solution
        _,  # none if we werent looking for extra parameters as scipy can (and we didnt)
        outZs,  # the output will be evaluated at different Zs
        outYGradYs, # yGradY at output Zs
        outDerivative, # derivative of yGradY wrt z
        residuals,  # relative residuals
        niter,  # number of iterations 
        status, # why the integration stopped; 0 means success, 1 means max number of meshNodes reached, 2 means singular Jacobian encountered during integration
        message, # verbal description of the termination reason
        success, # True if integration was successful
    ) = solve_bvp(
        setOfODEs,
        boundaryConditions,
        zs,
        yYgradGuess,
        p=[innerPs, outerPs, totalMagneticFlux],
        max_nodes=10000,  # FIXME copilot suggested this
        tol=1e-5,  # FIXME copilot suggested this
    )

    if not success:
        raise RuntimeError(
            f"Integration of magnetic equation failed with message {message}"
        )

    return yYgradArray[:, 0]


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
    matrixOfSecondDifferences = (
        0.5 * totalMagneticFlux / np.pi * centeredDifferences @ centeredDifferences
    )  #

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
