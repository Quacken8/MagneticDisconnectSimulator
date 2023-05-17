#!/usr/bin/env python3
import numpy as np
from dataHandling.dataStructure import SingleTimeDatapoint
from stateEquationsPT import IdealGas as StateEq
import constants as c
from scipy.integrate import simps
from scipy.optimize import newton, brentq
import loggingConfig
import logging
L = loggingConfig.configureLogging(logging.INFO, __name__)


def getBottomB(
    externalPressure: float | np.ndarray, bottomPressure: float | np.ndarray
) -> float | np.ndarray:
    """
    straight up from the thin tube approximation
    """
    return np.sqrt(np.sqrt(2 * c.mu0 * (externalPressure - bottomPressure)))


def getTopB() -> float:
    """
    in the paper it is said that it is *typically* set to 2k G. Not sure what *typically* means tho lol
    """
    return 2e3 * c.Gauss


def getAdjustedBottomPressure(
    currentState: SingleTimeDatapoint,
    bottomExternalPressure: float,
    dt: float,
    upflowVelocity: float,
    totalMagneticFlux: float,
) -> float:
    """
    boundary condition of pressure is only given on the bottom
    returns pressure at the bottom of the flux tube calculated from assumption that it should change based on the inflow of material trough the bottom boundary. Schüssler and Rempel 2018 eq 15
    """

    def massOfFluxTube(densities, Bs, zs, totalMagneticFlux):
        """
        equation 13 in Schüssler and Rempel 2018
        """
        return totalMagneticFlux * simps(densities / Bs, zs)

    def massAfterPressureAdjustment(
        unadjustedMass, bottomB, bottomDensity, totalMagneticFlux, dt, upflowVelocity
    ):
        """
        equation 15 in Schüssler and Rempel 2018

        approimation of how the total mass should change if the values of p, B and rho change at the bottom
        """
        return (
            unadjustedMass
            + totalMagneticFlux * upflowVelocity * dt * bottomDensity / bottomB
        )

    zs = currentState.zs
    currentPs = currentState.pressures[:]
    currentTs = currentState.temperatures[:]
    currentRhos = StateEq.density(pressure=currentPs, temperature=currentTs)

    currentBs = currentState.bs[:]

    # see for what adjustment of pressure at the bottom (dictated by inflow of material) will the mass of the whole tube change according to approximation via function massAfterPressureAdjustment

    bottomRho = currentRhos[
        -1
    ]  # note that in "massAfterPressureAdjustment" we use the current density. Schüssler and Rempel 2018 explicitly argues that change of bottomP affects both bottomB and bottomRho, however the effect on magnetic field is much stronger than that on the density. Same reasoning applies to the first argument of the massOfFluxTube

    arrayDelta = np.zeros(currentState.numberOfZSteps)
    arrayDelta[
        -1
    ] = 1  # purpose of this variable is to have the change of a variable at the bottom of the flux tube; just multiply this by a scalar and you can add it to the whole array

    currentBottomB = getBottomB(
        externalPressure=bottomExternalPressure, bottomPressure=currentPs[-1]
    )

    DeltaP = 0
    try:
        initialGuess = currentPs[-1]
        DeltaP, r = newton( # type: ignore ye idk what the deal is here
            lambda DeltaP: massAfterPressureAdjustment(
            massOfFluxTube(
                currentRhos, currentBs, zs=zs, totalMagneticFlux=totalMagneticFlux
            ),
            currentBottomB,
            bottomRho,
            totalMagneticFlux,
            dt,
            upflowVelocity,
        )
        - massOfFluxTube(
            currentRhos,
            currentBs + arrayDelta * DeltaP,
            zs=zs,
            totalMagneticFlux=totalMagneticFlux,
        ),
        x0 = initialGuess,
        full_output = True,
        )
        if not r.converged: raise RuntimeError("Newton method did not converge")

    except RuntimeError:
        L.warn("Newton method did not converge, trying Brent's method")
        upperBoundary = currentPs[-1]*1.1 # FIXME get a better guess for these
        lowerBoundary = bottomExternalPressure
        DeltaP = brentq(
            lambda DeltaP: massAfterPressureAdjustment(
                massOfFluxTube(
                    currentRhos, currentBs, zs=zs, totalMagneticFlux=totalMagneticFlux
                ),
                currentBottomB,
                bottomRho,
                totalMagneticFlux,
                dt,
                upflowVelocity,
            )
            - massOfFluxTube(
                currentRhos,
                currentBs + arrayDelta * DeltaP,
                zs=zs,
                totalMagneticFlux=totalMagneticFlux,
            ),
            a = lowerBoundary,
            b = upperBoundary,
        )

    # returns the new bottom pressure
    return currentPs[-1] + DeltaP
