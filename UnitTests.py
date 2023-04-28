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
    modelSZs = modelS.zs

    from scipy.interpolate import interp1d
    surfaceZ = 0*c.Mm
    surfaceT = interp1d(modelSZs, modelSTemperatures)(surfaceZ)
    surfaceP = interp1d(modelSZs, modelSPressures)(surfaceZ)

    dlnp = 1e-1
    maxDepth = 160*c.Mm

    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity, modelSNearestOpacity

    calmSun = getCalmSunDatapoint(StateEq = MESAEOS, dlnP=dlnp, lnSurfacePressure=np.log(surfaceP), surfaceTemperature=surfaceT, surfaceZ=surfaceZ, maxDepth=maxDepth, opacity=mesaOpacity)
    calmSunWithModelSOpacity = getCalmSunDatapoint(StateEq = MESAEOS, dlnP=dlnp, lnSurfacePressure=np.log(surfaceP), surfaceTemperature=surfaceT, surfaceZ=surfaceZ, maxDepth=maxDepth, opacity=modelSNearestOpacity)

    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(calmSun, toPlot, pltshow=False, label="Calm Sun with MESA kappa", log = True)
    axs = plotSingleTimeDatapoint(calmSunWithModelSOpacity, toPlot, axs = axs, pltshow=False, label="Calm Sun with model S kappa", log = True)
    plotSingleTimeDatapoint(modelS, toPlot, axs=axs, label="Model S", log = False)
    plt.legend()
    plt.show()

def testIDLOutput():
    zs, Ps, Ts, kappas, nablas = np.loadtxt("debuggingReferenceFromSvanda/idlOutput.dat", skiprows = 1, unpack=True)


    from stateEquationsPT import MESAEOS
    from opacity import modelSNearestOpacity
    from gravity import massBelowZ

    myOpacity = modelSNearestOpacity(Ps, Ts)
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

def compareModelSVsMESAOpacity():
    modelSPressures = modelS.pressures
    modelSTemperatures = modelS.temperatures
    modelSZs = modelS.zs

    from opacity import mesaOpacity, modelSNearestOpacity

    mesaOpacities = mesaOpacity(modelSPressures, modelSTemperatures)
    modelSNearest = modelSNearestOpacity(modelSPressures, modelSTemperatures)

    plt.loglog(modelSZs / c.Mm, mesaOpacities, label="MESA")
    plt.loglog(modelSZs / c.Mm, modelSNearest, label="Model S")
    plt.legend()
    plt.xlabel("z [Mm]")
    plt.ylabel("Opacity [m^2/kg]")
    plt.show()

def main():
    compareModelSVsMESAOpacity()

if __name__ == "__main__":
    main()
