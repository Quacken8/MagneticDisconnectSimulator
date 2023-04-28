#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import UnivariateSpline as ScipySpline
import constants as c


def volumeOfSphericalLayer(
    topRadius: float | np.ndarray, width: float | np.ndarray
) -> float | np.ndarray:
    """
    volume of a spherical layer which's (??english?) top boundary has radius r and the top boundary has radius r+width
    """
    return (
        4
        / 3
        * np.pi
        * (
            topRadius * topRadius * topRadius
            - (topRadius - width) * (topRadius - width) * (topRadius - width)
        )
    )

pathToModeS = "externalData/model_S_new.dat"
modelRs, modelRhos = np.loadtxt(pathToModeS, skiprows=1, usecols=(0, 3), dtype=float).T
modelLength = len(modelRs)
assert modelLength == len(modelRhos)

modelRs = modelRs[::-1]
modelRhos = modelRhos[::-1]

currentWidth = np.diff(modelRs, prepend=0)
currentVolume = volumeOfSphericalLayer(modelRs, currentWidth)
currentMass = currentVolume * modelRhos
massesBelowR = np.cumsum(currentMass)

modelZs = (modelRs.max() - modelRs)[::-1]

mBelowZSpline = ScipySpline(modelZs, massesBelowR[::-1], s=0, k=1, ext=3)

def massBelowZ(z: float | np.ndarray) -> np.ndarray:
    """
    Sun's mass z meters below the surface
    """
    return np.array(mBelowZSpline(z))

gravitationalAccelerations = c.G * massesBelowR / (modelRs * modelRs)

gravitationalAccelerationsInZs = gravitationalAccelerations[::-1]


gSpline = ScipySpline(modelZs, gravitationalAccelerationsInZs, s=0, k=1, ext=3)
def g(z: float | np.ndarray) -> np.ndarray:
    """
    gravitational acceleration in m/s^2 z meters below the surface
    uses the model S 
    """
    return np.array(gSpline(z))



def main():
    """debug function for the gravity code"""
    pass



if __name__ == "__main__":
    main()
