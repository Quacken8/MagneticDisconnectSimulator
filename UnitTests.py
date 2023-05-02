#!/usr/bin/env python3
from dataStructure import (
    Data,
    SingleTimeDatapoint,
    createDataFromFolder,
    dictionaryOfVariables,
)
import numpy as np
from initialConditionsSetterUpper import mockupDataSetterUpper, loadModelS
from stateEquationsPT import IdealGas
from sunSolvers.calmSun import getCalmSunDatapoint
from dataVizualizer import plotSingleTimeDatapoint

# from solvers import oldYSolver
from initialConditionsSetterUpper import loadModelS
import constants as c
import os
from matplotlib import pyplot as plt
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)
L = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

pathToModelS = "externalData/model_S_new.dat"
modelS = loadModelS()

def testCalmSunVsModelS():
    modelSPressures = modelS.pressures
    modelSTemperatures = modelS.temperatures
    modelSZs = modelS.zs

    from scipy.interpolate import interp1d
    surfaceZ = 0*c.Mm
    surfaceT = interp1d(modelSZs, modelSTemperatures)(surfaceZ)
    surfaceP = interp1d(modelSZs, modelSPressures)(surfaceZ)

    dlnp = 1e-3
    maxDepth = 160*c.Mm

    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity

    import time
    now = time.time()
    calmSun = getCalmSunDatapoint(StateEq = MESAEOS, dlnP=dlnp, lnSurfacePressure=np.log(surfaceP), surfaceTemperature=surfaceT, surfaceZ=surfaceZ, maxDepth=maxDepth, opacity=mesaOpacity, guessTheZRange=True)
    print("time elapsed: ", time.time()-now)
    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(calmSun, toPlot, pltshow=False, label="Calm Sun with MESA kappa", log = False)
    #axs = plotSingleTimeDatapoint(calmSunWithModelSOpacity, toPlot, axs = axs, pltshow=False, label="Calm Sun with model S kappa", log = False)
    plotSingleTimeDatapoint(modelS, toPlot, axs=axs, label="Model S", log = True)
    plt.legend()
    plt.show()

def testFiniteDifferences():
    from sunSolvers.temperatureSolver import secondCentralDifferencesMatrix
    N = 100
    xs = np.sort(np.random.random(N))*2*np.pi

    secondDif = secondCentralDifferencesMatrix(xs)

    f = np.sin(xs)
    expectedDf = -np.sin(xs)
    aprox = secondDif.dot(f)

    df = np.gradient(f, xs)
    ddf = np.gradient(df, xs)
    numpyVersion = ddf

    import matplotlib.pyplot as plt
    plt.plot(xs, expectedDf, label="exact solution of d² sin x/dx²")
    plt.scatter(xs, aprox,      marker = ".", label="second differences δ² sin x/δx²", c = "red")
    plt.scatter(xs, numpyVersion, marker = ".", label="numpy version of second differences", c = "black")
    plt.ylim(-3,3)
    plt.legend()
    plt.show()

def main():
    testCalmSunVsModelS()

if __name__ == "__main__":
    main()
