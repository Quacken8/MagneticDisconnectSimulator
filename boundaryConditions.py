#!/usr/bin/env python3
import numpy as np
from dataStructure import SingleTimeDatapoint
import stateEquations
import constants as c
from scipy.integrate import romb as RombIntegrate

def getBottomB(externalPressure: float | np.ndarray, bottomPressure: float | np.ndarray) -> float | np.ndarray:
    """
    straight up from the thin tube approximation
    """
    return np.sqrt(2*c.mu0*(externalPressure-bottomPressure))

def getTopB()->float:
    """
    in the paper it is said that it is *typically* set to 2k G. Not sure what *typically* means tho lol
    """
    return 2e3 * c.Gauss

def getBottomPressure(currentState:SingleTimeDatapoint, dt:float,  upflowVelocity:float, totalMagneticFlux:float)->float:
    """
    boundary condition of pressure is only given on the bottom
    returns pressure at the bottom of the flux tube calculated from assumption that it should change based on the inflow of material trough the bottom boundary. Schüssler and Rempel 2018 eq 15
    """
    
    def massOfFluxTube(densities, Bs, dz, totalMagneticFlux):
        """
        equation 13 in Schüssler and Rempel 2018
        """
        return totalMagneticFlux*RombIntegrate(densities/Bs, dx = dz)

    def massAfterPressureAdjustment(unadjustedMass, bottomB, bottomDensity, totalMagneticFlux, dt, upflowVelocity):
        """
        equation 15 in Schüssler and Rempel 2018

        approimation of how the total mass should change if the values of p, B and rho change at the bottom
        """
        return unadjustedMass + totalMagneticFlux*upflowVelocity*dt*bottomDensity/bottomB

    currentPs = currentState.pressures[:]
    currentTs = currentState.temperatures[:]
    currentRhos = stateEquations.idealGas(pressure = currentPs, temperature = currentTs)
    currentBs = currentState.B_0s[:]


    #see for what adjustment of pressure at the bottom (dictated by inflow of material) will the mass of the whole tube change according to approximation via function massAfterPressureAdjustment

    bottomRho = currentRhos[-1] # note that in "massAfterPressureAdjustment" we use the current density. Schüssler and Rempel 2018 explicitly argues that change of bottomP affects both bottomB and bottomRho, however the effect on magnetic field is much stronger than that on the density. Same reasoning applies to the first argument of the massOfFluxTube
    
    arrayDelta = np.zeros(currentState.numberOfZSteps)
    arrayDelta[-1] = 1  # purpose of this variable is to have the change of a variable at the bottom of the flux tube; just multiply this by a scalar and you can add it to the whole array

    currentBottomB = getBottomB()
    dz = currentState.dz
    dP = Solve(
        massAfterPressureAdjustment(
            massOfFluxTube(currentRhos, currentBs, dz = dz, totalMagneticFlux = totalMagneticFlux), 
            currentBottomB, bottomRho, totalMagneticFlux, dt, upflowVelocity)
        ==
        massOfFluxTube(currentRhos, currentBs + arrayDelta*dP, dz=dz,
                       totalMagneticFlux=totalMagneticFlux),
        dP
        )

    # returns the new bottom pressure
    return currentPs[-1] + dP
