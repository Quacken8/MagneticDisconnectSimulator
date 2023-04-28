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
from calmSun import getCalmSunDatapoint
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

    surfaceT = modelSTemperatures[0]
    surfaceP = modelSPressures[0]

    dlnp = 1e-1
    maxDepth = 20*c.Mm

    from stateEquationsPT import IdealGas
    from opacity import mesaOpacity

    calmSun = getCalmSunDatapoint(StateEq = IdealGas, dlnP=dlnp, lnSurfacePressure=np.log(surfaceP), surfaceTemperature=surfaceT, surfaceZ=0, maxDepth=maxDepth, opacity=mesaOpacity)

    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(calmSun, toPlot, pltshow=False, label="Calm Sun")
    plotSingleTimeDatapoint(modelS, toPlot, axs=axs, label="Model S")
    plt.legend()
    plt.show()

def testIDLOutput():
    rs, Ps, Ts, kappas, nablas = np.loadtxt("debuggingReferenceFromSvanda/idlOutput.dat", skiprows = 1, unpack=True)
    zs = rs - rs[0]


    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity
    from gravity import massBelowZ

    myOpacity = mesaOpacity(Ps, Ts)
    M_r = massBelowZ(zs)
    myNablas = MESAEOS.radiativeLogGradient(Ts, Ps, M_r, myOpacity)

    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].plot(zs, nablas, label="Svanda")
    axs[0].plot(zs, myNablas, label="Mine")
    axs[0].legend()
    axs[0].set_ylabel("nabla rad")
    axs[1].plot(zs, kappas, label="Svanda")
    axs[1].plot(zs, myOpacity, label="Mine")
    axs[1].set_ylabel("kappa")
    axs[1].set_xlabel("z [m]")
    plt.show()

def main():
    testCalmSunVsModelS()

if __name__ == "__main__":
    main()
