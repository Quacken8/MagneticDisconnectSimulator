import numpy as np
from constants import *

def getBottomB(externalPressure:float, bottomPressure:float):
    """
    straight up from the thin tube approximation
    """
    return np.sqrt(8*np.pi*(externalPressure-bottomPressure))

def getBottomPressure():
    """
    boundary condition of pressure is only given on the bottom
    """
    
    def massOfFluxTube(densities, Bs, zs, totalMagneticFlux):
        """
        equation 13 in Schüssler and Rempel 2018
        """
        return totalMagneticFlux*Integrate(densities/Bs, zs)

    def massAfterPressureAdjustment(unadjustedMass, bottomB, bottomDensity, totalMagneticFlux, timestep, upflowVelocity):
        """
        equation 15 in Schüssler and Rempel 2018
        """
        return unadjustedMass + totalMagneticFlux*upflowVelocity*timestep*bottomDensity/bottomB

    get mass from integration of rho/B 
    its a function of p cuz of state eq ??
    m(p + Δ p ) = m(p) + phi rho / B * timestep * inflow
    solve that using newton

    get bottom p

    distribute that p throughout the column via hydrostatic eq 
    