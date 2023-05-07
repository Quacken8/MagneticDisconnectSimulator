#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import UnivariateSpline
from initialConditionsSetterUpper import loadModelS
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

modelS = loadModelS()
modelSZs = modelS.zs
modelRhosBottomUp = modelS.derivedQuantities["rhos"][::-1]
del modelS

centerOfTheSun = modelSZs[-1]
modelRsBottomUp = (centerOfTheSun - modelSZs)[::-1]

widthsOfLayers = np.diff(modelRsBottomUp, prepend=0)
volumesOfLayers = volumeOfSphericalLayer(modelRsBottomUp, widthsOfLayers)
massesOfLayers = volumesOfLayers * modelRhosBottomUp
massesBelowR = np.cumsum(massesOfLayers)


mBelowZSpline = UnivariateSpline(modelSZs, massesBelowR[::-1], s=0, k=1, ext=3)

def massBelowZ(z: float | np.ndarray) -> np.ndarray:
    """
    Sun's mass z meters below the surface
    """
    return np.array(mBelowZSpline(z))

gravitationalAccelerations = c.G * massesBelowR[1:] / (modelRsBottomUp[1:] * modelRsBottomUp[1:])

gravitationalAccelerationsInZs = gravitationalAccelerations[::-1]


gSpline = UnivariateSpline(modelSZs[:-1], gravitationalAccelerationsInZs, s=0, k=1, ext=3)
def g(z: float | np.ndarray) -> np.ndarray:
    """
    gravitational acceleration in m/s^2 z meters below the surface
    uses the model S 
    """
    return np.array(gSpline(z))



def main():
    """debug function for the gravity code"""
    import matplotlib.pyplot as plt
    plt.plot(modelSZs[:-1], gravitationalAccelerations)
    plt.show()



if __name__ == "__main__":
    main()
