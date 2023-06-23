#!/usr/bin/env python3
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import constants as c
from scipy import integrate
from scipy.sparse import linalg
from dataHandling import boundaryConditions as boundary
from sunSolvers import handySolverStuff
import loggingConfig
import logging
L = loggingConfig.configureLogging(logging.INFO, __name__)


def rightHandSideOfYEq(y, innerP, outerP, totalMagneticFlux):
    """
    right hand side of the differential equation 5
    d²y/dz²= (y⁴ - 2μ₀ (pₑ - pᵢ))/(Φ/2π y)
    """
    return (y * y * y - 2 * c.mu0 * (outerP - innerP) / y) / (
        totalMagneticFlux / (2 * np.pi)
    )


def setOfODEs(z, yYgradArray, innerPZ, outerPZ, totalMagneticFlux):
    """
    turns the second order diff eq into a set of two first order diff eqs
    y'' = (y⁴ - 2μ₀ (pₑ - pᵢ))/(Φ/2π y)
    by introducing ygrad = dy/dz
    returns an array of [ygrad, ygrad'], i.e. [y', y'']

    args:
        yYgradArray: array of [y, ygrad]
        innerPZ: tuple of (zs, innerPs) where zs is an array of depths at which the pressure is given
        outerPZ: tuple of (zs, outerPs) where zs is an array of depths at which the pressure is given
    """
    y = yYgradArray[0]
    ygrad = yYgradArray[1]

    outerP = np.interp(z, outerPZ[0], outerPZ[1])
    innerP = np.interp(z, innerPZ[0], innerPZ[1])

    return np.array([ygrad, rightHandSideOfYEq(y, innerP, outerP, totalMagneticFlux)])


def integrateMagneticEquation(zs, innerPs, outerPs, totalMagneticFlux, yGuess = None, tolerance = 1e-3):
    """
    solves differential equation 5 to get y = sqrt(B) = y(z)
    Φ/2π d²y/dz² y = y⁴ - 2μ₀ (pₑ - pᵢ)
    """

    # boundaryConditions
    bottomPe, bottomPi = (
        outerPs[-1],
        innerPs[-1],
    )  # TODO check if ur taking the correct thingy
    bottomY = np.sqrt(boundary.getBottomB(bottomPe, bottomPi))
    topY = np.sqrt(boundary.getTopB())

    def boundaryConditions(yYgradBottom, yGradTop):
        return np.array([yYgradBottom[0] - bottomY, yGradTop[0] - topY])

    # initial guess
    # here we assume linear function
    if yGuess is None:
        yGuess = np.linspace(bottomY, topY, zs.size)
        yGradGuess = np.ones_like(zs) * (topY - bottomY) / (zs[-1] - zs[0])
    else:
        yGradGuess = np.gradient(yGuess, zs)
    yYgradGuess = np.array([yGuess, yGradGuess])

    simplerODEs = partial(setOfODEs, innerPZ=(zs, innerPs), outerPZ=(zs, outerPs), totalMagneticFlux=totalMagneticFlux)

    # integration
    integrationResult = integrate.solve_bvp(
        simplerODEs,
        boundaryConditions,
        zs,
        yYgradGuess,
        max_nodes=zs.size * 10, # TODO - optimize this
        tol=tolerance
    )

    success = integrationResult.success

    if not success:
        message = integrationResult.message
        raise RuntimeError(
            f"Integration of magnetic equation failed: {message}"
        )
    L.debug(f"integration of magnetic equation took {integrationResult.niter} iterations")
    # the integration solves the equation on its own grid, so we need to interpolate it back
    yYgradArray = integrationResult.y
    yOnOldZs = np.interp(zs, integrationResult.x, yYgradArray[0])
    return yOnOldZs

def oldYSolver(
    zs: np.ndarray,
    innerPs: np.ndarray,
    outerPs: np.ndarray,
    totalMagneticFlux: float,
    yGuess: np.ndarray,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """
    solver of the differential equation Φ/2π d²y/dz² y = y⁴ - 2μ₀ (pₑ - pᵢ)
    taken from the bc thesis using tridiagonal matrix
    this may be very slow cuz it's of the first order

    it tries to solve the equation
    A y_n+1 = b_n = (y^3_n - 2mu/y(p_e-p_i)) / (Φ/2π)

    it works according to this flowchart

    .               ┌─────────────────────┐\n
    .               │                     │\n
    .          ┌────▼─┐                   │\n
    .          │get ys│                   │\n
    .    ┌─────┴┬─┬───┴──────────────┐    │\n
    .    │try Ay│ │try y^3-2mu/y(p-p)│    │\n
    .    ├──────┴─┴────┬─────────────┘    │\n
    .    │correction = │                  │\n
    .    │   = y' =    └───────────────┐  │\n
    .    │Solve(Ay'=Ay-(y^3-2mu/y(p-p))│  │\n
    .    ├──────────────────────┬──────┘  │\n
    .┌──►│y_new = y + corrFac*y'│         │\n
    .│   ├───────┬─┬────────────┴──────┐  │\n
    .│   │try Ayn│ │try yn^3-2mu/ynp-p)│  │\n
    .│   └─┬─────┴─┴─────┬─────────────┘  │\n
    .│     │ Worse?      │Better?         │\n
    .│   ┌─▼──────────┐  └────────────────┘\n
    .└───┤corrFac*=0.5│                    \n
    .    └────────────┘                    \n
    """

    matrixOfSecondDifferences = handySolverStuff.secondCentralDifferencesMatrix(
        zs, constantBoundaries=True
    )
    inverseSecondDiff = linalg.inv(matrixOfSecondDifferences.tocsc())

    rightSide = rightHandSideOfYEq(yGuess, innerPs, outerPs, totalMagneticFlux)
    leftSide = matrixOfSecondDifferences.dot(yGuess)

    guessError = np.linalg.norm(rightSide - leftSide)

    correctionFactor = 1
    while guessError > tolerance:
        if correctionFactor == 0:
            raise RuntimeError(
                f"Correction factor is zero. This means that the solver is stuck. The error is {guessError}"
            )
        changeInbetweenSteps = inverseSecondDiff.dot(rightSide - leftSide)
        # changeInbetweenSteps[0] = changeInbetweenSteps[-1] = 0 TODO - check if this is to be used

        newYGuess = yGuess + correctionFactor * changeInbetweenSteps

        newRightSide = rightHandSideOfYEq(newYGuess, innerPs, outerPs, totalMagneticFlux)
        newLeftSide = matrixOfSecondDifferences.dot(newYGuess)

        newGuessError = np.linalg.norm(newRightSide - newLeftSide)

        if newGuessError < guessError:
            yGuess = newYGuess
            rightSide = newRightSide
            leftSide = newLeftSide
            guessError = newGuessError
            correctionFactor *= 1.1  # TODO - optimize this
        else:
            correctionFactor *= 0.5

    return yGuess
