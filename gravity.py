#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import UnivariateSpline as ScipySpline
import constants as c

"""
gravitational acceleration in m/s^2 z meters below the surface
uses the model S 
"""

def volumeOfSphericalLayer(radius:float|np.ndarray, width:float|np.ndarray)->float|np.ndarray:
    """
    volume of a spherical layer which's (??english?) bottom boundary has radius r and the top boundary has radius r+width
    """
    return 4/3*np.pi * ((radius + width)*(radius + width)*(radius + width) - radius*radius*radius)

modelZs   = np.loadtxt("model_S_raw.dat", usecols=0)
rs = modelZs.max()-modelZs

# the model indexes with depth, therefore the array has to be flipped
modelRhos = np.loadtxt("model_S_raw.dat", usecols=3)[::-1]
masses = np.zeros(modelZs.size)

for i in range(1, masses.size):
    masses[i] = masses[i-1] + modelRhos[i] * volumeOfSphericalLayer(rs[i], rs[i]-rs[i-1])

gravitationalAccelerations = (c.G*masses/(rs*rs))[::-1]
# reversed back so it can be used with depths z again

"""
linear interpolation of Sun's gravity z meters below the surface
"""
g = ScipySpline(modelZs, gravitationalAccelerations, s=0, k=1)
