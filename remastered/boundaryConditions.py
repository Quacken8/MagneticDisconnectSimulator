import numpy as np
from dataStructure import SingleTimeDatapoint
import stateEquations
from constants import *
from scipy.integrate import romb as RombIntegrate

def getBottomB(externalPressure:float, bottomPressure:float):
    """
    straight up from the thin tube approximation
    """
    return np.sqrt(2*Cmu0*(externalPressure-bottomPressure))

def getTopB():
    """
    in the paper it is said that it is *typically* set to 2k G. Not sure what *typically* means tho lol
    """
    return 2e3 * CGauss

def getBottomPressure(currentState:SingleTimeDatapoint, dt:float, zs, upflowVelocity:float, totalMagneticFlux:float):
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
    dz = currentState.dz
    dP = Solve(
        massAfterPressureAdjustment(
            massOfFluxTube(currentRhos, currentBs, dx = dz, totalMagneticFlux), 
            bottomB, bottomRho, totalMagneticFlux, dt, upflowVelocity)
        ==
        massOfFluxTube(currentRhos, Bs + deltaB(dP), dx = dz, totalMagneticFlux),
        dP
        )

    # delta variables are just a bunch of zeros the length of the flux tube with a single nonzero delta value

    # both bottomB(dP) and Bs + deltaB(dP) has to come from the differential eq, right? But in the old code it's only the latter that is calculated as function of dP? See line 267 of solvers.py: initial_mass is set static, outside the funcion func() that is solved in newton!!

    # returns the new bottom pressure
    return currentPs[-1] + dP
