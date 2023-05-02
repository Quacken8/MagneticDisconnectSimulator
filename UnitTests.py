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
    surfaceZ = 1*c.Mm
    surfaceT = interp1d(modelSZs, modelSTemperatures)(surfaceZ)
    surfaceP = interp1d(modelSZs, modelSPressures)(surfaceZ)

    dlnp = 1e-1
    maxDepth = 80*c.Mm

    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity

    import time
    now = time.time()
    calmSun = getCalmSunDatapoint(StateEq = MESAEOS, dlnP=dlnp, lnSurfacePressure=np.log(surfaceP), surfaceTemperature=surfaceT, surfaceZ=surfaceZ, maxDepth=maxDepth, opacity=mesaOpacity)
    print("time elapsed: ", time.time()-now)
    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(calmSun, toPlot, pltshow=False, label="Calm Sun with MESA kappa", log = False)
    #axs = plotSingleTimeDatapoint(calmSunWithModelSOpacity, toPlot, axs = axs, pltshow=False, label="Calm Sun with model S kappa", log = False)
    plotSingleTimeDatapoint(modelS, toPlot, axs=axs, label="Model S", log = True)
    plt.legend()
    plt.show()

def testIDLOutput():
    zs, Ps, Ts, rhos, kappas, nablas, Hs = np.loadtxt("debuggingReferenceFromSvanda/idlOutput.dat", skiprows = 1, unpack=True)


    from stateEquationsPT import MESAEOS
    from opacity import modelSNearestOpacity
    from gravity import massBelowZ, g

    myOpacity = modelSNearestOpacity(Ps, Ts)
    M_r = massBelowZ(zs)
    gs = 274
    myNablas = MESAEOS.radiativeLogGradient(Ts, Ps, M_r, myOpacity)
    myHs = MESAEOS.pressureScaleHeight(Ts, Ps, gs)
    myRhos = MESAEOS.density(Ps, Ts)

    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].plot(zs, nablas, label="Svanda")
    axs[0].plot(zs, myNablas, label="Mine")
    axs[0].legend()
    axs[0].set_ylabel("nabla rad")
    axs[1].plot(zs, kappas, label="Svanda")
    axs[1].plot(zs, myOpacity, label="Mine")
    axs[1].set_ylabel("kappa")
    axs[1].set_xlabel("z [m]")
    axs[1].legend()

    fig2, axs2 = plt.subplots(1,1, sharex=True)
    axs2.plot(zs, Hs, label="Svanda")
    axs2.plot(zs, myHs, label="Mine")
    axs2.set_ylabel("H")
    axs2.set_xlabel("z [m]")
    axs2.set_yscale("log")
    axs2.legend()

    fig3, axs3 = plt.subplots(1,1, sharex=True)
    axs3.plot(zs, rhos, label="Svanda")
    axs3.plot(zs, myRhos, label="Mine")
    axs3.set_ylabel("rho")
    axs3.set_xlabel("z [m]")
    axs3.set_yscale("log")
    axs3.legend()
    
    plt.show()


def main():
    testCalmSunVsModelS()
    testIDLOutput()

if __name__ == "__main__":
    main()
