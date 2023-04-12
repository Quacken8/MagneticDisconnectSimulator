#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import UnivariateSpline as ScipySpline
import constants as c

"""
gravitational acceleration in m/s^2 z meters below the surface
uses the model S 
"""


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

massesBelowR = np.zeros(modelLength)

modelRs = modelRs[::-1]
modelRhos = modelRhos[::-1]

for i in range(1, modelLength):
    """
    ─────R[i+1]──────────────────────────M[i+1]─
                    ▲                      │
        Rho[i+1]    │ W[i+1]               │
                    ▼                      │
    ──────R[i]───────────────────────M[i]──┼────
                    ▲                  │   │
         Rho[i]     │ W[i]             │   │
                    ▼                  │   │
    ─────R[i-1]─────────────────M[i-1]─┼───┼────
                                   │   │   │
                                   ▼   ▼   ▼
    """

    currentTopRadius = modelRs[i]
    currentWidth = modelRs[i] - modelRs[i - 1]
    currentVolume = volumeOfSphericalLayer(currentTopRadius, currentWidth)
    currentDensity = modelRhos[
        i
    ]  # TODO - ask about where in the cell of model S the radius R is cuz rn im expecting it to be at the top
    currentMass = currentVolume * currentDensity

    massesBelowR[i] = massesBelowR[i - 1] + currentMass

gravitationalAccelerations = c.G * massesBelowR / (modelRs * modelRs)

modelZs = modelRs
gravitationalAccelerationsInZs = gravitationalAccelerations[::-1]
"""
linear interpolation of Sun's gravity z meters below the surface
"""
g = ScipySpline(modelZs, gravitationalAccelerationsInZs, s=0, k=1, ext=3)


def main():
    """debug function for the gravity code"""

    print(gravitationalAccelerations.max())

    import matplotlib.pyplot as plt

    plt.plot(modelZs, gravitationalAccelerationsInZs)
    plt.plot()
    plotZs = np.linspace(0, 1e6)
    toplot = g(plotZs)
    plt.plot(plotZs, toplot)
    plt.show()


if __name__ == "__main__":
    main()
