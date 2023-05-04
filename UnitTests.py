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
from scipy.integrate import odeint
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

    dlnp = 1e-2
    maxDepth = 160*c.Mm

    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity, modelSNearestOpacity

    import time
    now = time.time()
    calmSun = getCalmSunDatapoint(StateEq = MESAEOS, dlnP=dlnp, lnSurfacePressure=np.log(surfaceP), surfaceTemperature=surfaceT, surfaceZ=surfaceZ, maxDepth=maxDepth, opacity=mesaOpacity, guessTheZRange=True)
    calmSunWithModelSOpacity = getCalmSunDatapoint(StateEq = MESAEOS, dlnP=dlnp, lnSurfacePressure=np.log(surfaceP), surfaceTemperature=surfaceT, surfaceZ=surfaceZ, maxDepth=maxDepth, opacity=modelSNearestOpacity, guessTheZRange=True)
    print("time elapsed: ", time.time()-now)
    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(calmSun, toPlot, pltshow=False, label="Calm Sun with MESA kappa", log = False)
    axs = plotSingleTimeDatapoint(calmSunWithModelSOpacity, toPlot, axs=axs, pltshow=False, label="Calm Sun with model S kappa", log = False)
    plotSingleTimeDatapoint(modelS, toPlot, axs=axs, label="Model S", log = True)
    plt.legend()
    plt.show()

def testIDLOutput():
    modelSPressures = modelS.pressures
    modelSTemperatures = modelS.temperatures
    modelSZs = modelS.zs
    from opacity import modelSNearestOpacity
    from stateEquationsPT import MESAEOS
    from gravity import g, massBelowZ
    zs, p, T, rho, kapa, nablaRad, H = np.loadtxt("debuggingReferenceFromSvanda/idlOutput.dat", unpack=True, skiprows=1)
    myRho = MESAEOS.density(modelSTemperatures, modelSPressures)
    myKapa = modelSNearestOpacity(modelSPressures, modelSTemperatures)
    myGs = g(modelSZs)
    myM_r = massBelowZ(modelSZs)
    myH = MESAEOS.pressureScaleHeight(modelSTemperatures, modelSPressures, myGs)
    myNablaRad = MESAEOS.radiativeLogGradient(modelSTemperatures, modelSPressures, myM_r, myKapa)

    svandaGs = g(zs)
    svandaM_r = massBelowZ(zs)
    myHFromSvanda = MESAEOS.pressureScaleHeight(T, p, svandaGs)
    myKappaFromSvanda = modelSNearestOpacity(p, T)
    myNablaRadFromSvanda = MESAEOS.radiativeLogGradient(T, p, svandaM_r, myKappaFromSvanda)


    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].loglog(modelSZs, myRho, label="MESA rho (z modelu S)")
    axs[0, 0].loglog(zs, rho, label="IDL rho")
    axs[0, 0].set_title("rho")
    axs[0, 0].set_xlim(5e5, 20e6)
    axs[0, 0].legend()

    axs[0, 1].loglog(modelSZs, modelSPressures, label="model S P")
    axs[0, 1].loglog(zs, p, label="IDL P")
    axs[0, 1].set_title("P")
    axs[0, 1].set_xlim(5e5, 20e6)
    axs[0, 1].legend()

    axs[0, 2].loglog(modelSZs, modelSTemperatures, label="model S T")
    axs[0, 2].loglog(zs, T, label="IDL T")
    axs[0, 2].set_title("T")
    axs[0, 2].set_xlim(5e5, 20e6)
    axs[0, 2].legend()

    axs[1, 0].loglog(modelSZs, myKapa, label="model S interpolovaná kappa")
    axs[1, 0].loglog(zs, kapa, label="IDL kapa")
    axs[1, 0].loglog(zs, myKappaFromSvanda, label="model S interpolovaná kappa z IDL PT", linestyle="--")
    axs[1, 0].set_title("kapa")
    axs[1, 0].set_xlim(5e5, 20e6)
    axs[1, 0].legend()

    axs[1, 1].loglog(modelSZs, myNablaRad, label="MESA ∇rad z model S kappa")
    axs[1, 1].loglog(zs, nablaRad, label="IDL ∇rad")
    axs[1, 1].loglog(zs, myNablaRadFromSvanda, label="MESA ∇rad z model S kappa IDL PT", linestyle="--")
    axs[1, 1].set_title("nablaRad")
    axs[1, 1].set_xlim(5e5, 20e6)
    axs[1, 1].legend()

    axs[1, 2].loglog(modelSZs, myH, label="MESA H z modelz S")
    axs[1, 2].loglog(zs, H, label="IDL H")
    axs[1, 2].loglog(zs, myHFromSvanda, label="MESA H z modelu S IDL PT", linestyle="--")
    axs[1, 2].set_title("H")
    axs[1, 2].set_xlim(5e5, 20e6)
    axs[1, 2].legend()

    plt.show()

def testNewIDLOutput():
    zsS, pressureS, temperatureS, kappaS =modelS.zs, modelS.pressures, modelS.temperatures, modelS.derivedQuantities["kappas"]
    idlZs, idlP, idlT, idlKappa = np.loadtxt("debuggingReferenceFromSvanda/anotherIdlOutput.dat", unpack=True, skiprows=1)

    from opacity import modelSNearestOpacity, mesaOpacity
    mesaFromIDL = mesaOpacity(idlP, idlT)
    mesaFromS = mesaOpacity(pressureS, temperatureS)
    myKappaS = modelSNearestOpacity(pressureS, temperatureS)
    myKappaIDL = modelSNearestOpacity(idlP, idlT)

    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].loglog(zsS, mesaFromS, label="model S kappa from mesa", linewidth = 10)
    axs[0].loglog(zsS, kappaS, label="model S kappa", linewidth = 5)
    axs[0].loglog(zsS, myKappaS, label="model S kappa from my interpolation")
    axs[0].loglog(idlZs, mesaFromIDL, label="IDL kappa from mesa", linewidth = 10)
    axs[0].loglog(idlZs, idlKappa, label="IDL kapa", linewidth = 5)
    axs[0].loglog(idlZs, myKappaIDL, label="IDL kappa from my interpolation")
    axs[0].set_title("kapa")
    axs[0].legend()

    axs[1].loglog(zsS, pressureS, label="modelS")
    axs[1].loglog(idlZs, idlP, label = "IDL")
    axs[1].set_title("P")
    axs[1].legend()

    axs[2].loglog(zsS, temperatureS, label="modelS")
    axs[2].loglog(idlZs, idlT, label = "IDL")
    axs[2].set_title("T")
    axs[2].legend()

    plt.show()

def testCalmSunVsIDLS():
    idlZs, idlPs, idlTs = np.loadtxt("debuggingReferenceFromSvanda/anotherIdlOutput.dat", unpack=True, skiprows=1, usecols = (0,1,2))

    modelSPressures = idlPs
    modelSTemperatures = idlTs
    modelSZs = idlZs

    idlDatapoint = SingleTimeDatapoint(idlTs, idlPs, idlZs)



    from scipy.interpolate import interp1d
    surfaceZ = 0*c.Mm
    surfaceT = interp1d(modelSZs, modelSTemperatures)(surfaceZ)
    surfaceP = interp1d(modelSZs, modelSPressures)(surfaceZ)

    dlnp = 1e-3
    maxDepth = 160*c.Mm

    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity, modelSNearestOpacity

    import time
    now = time.time()
    calmSun = getCalmSunDatapoint(StateEq = MESAEOS, dlnP=dlnp, lnSurfacePressure=np.log(surfaceP), surfaceTemperature=surfaceT, surfaceZ=surfaceZ, maxDepth=maxDepth, opacity=modelSNearestOpacity, guessTheZRange=True)
    print("time elapsed: ", time.time()-now)
    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(calmSun, toPlot, pltshow=False, label="Calm Sun with MESA kappa", log = False)
    #axs = plotSingleTimeDatapoint(calmSunWithModelSOpacity, toPlot, axs = axs, pltshow=False, label="Calm Sun with model S kappa", log = False)
    plotSingleTimeDatapoint(idlDatapoint, toPlot, axs=axs, label="Model S", log = True)
    plt.legend()
    plt.show()

def testCalmSunBottomUp():
    modelSPressures = modelS.pressures
    modelSTemperatures = modelS.temperatures
    modelSZs = modelS.zs

    from scipy.interpolate import interp1d
    surfaceZ = 160*c.Mm
    surfaceT = interp1d(modelSZs, modelSTemperatures)(surfaceZ)
    surfaceP = interp1d(modelSZs, modelSPressures)(surfaceZ)

    dlnp = -1e-3
    maxDepth = 0.1*c.Mm

    from stateEquationsPT import MESAEOS
    from opacity import modelSNearestOpacity

    import time
    now = time.time()
    calmSun = getCalmSunDatapoint(StateEq = MESAEOS, dlnP=dlnp, lnSurfacePressure=np.log(surfaceP), surfaceTemperature=surfaceT, surfaceZ=surfaceZ, maxDepth=maxDepth, opacity=modelSNearestOpacity, guessTheZRange=True)
    print("time elapsed: ", time.time()-now)
    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(calmSun, toPlot, pltshow=False, label="Calm Sun with MESA kappa", log = False)
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
    plt.scatter(xs, aprox,      marker = ".", label="second differences δ² sin x/δx²", c = "red") # type: ignore
    plt.scatter(xs, numpyVersion, marker = ".", label="numpy version of second differences", c = "black") # type: ignore
    plt.ylim(-3,3)
    plt.legend()
    plt.show()

def testModelSKappavsMESAKappa():
    ps = modelS.pressures
    ts = modelS.temperatures
    kappas = modelS.derivedQuantities["kappas"]
    zs = modelS.zs
    from opacity import mesaOpacity, modelSNearestOpacity
    mesaOpacities = mesaOpacity(ps, ts)
    interpolatedOpacities = modelSNearestOpacity(ps, ts)

    plt.loglog(zs, kappas, label="model S kappa")
    plt.loglog(zs, interpolatedOpacities, label="interpolated model S kappa", linestyle = "--")
    plt.loglog(zs, mesaOpacities, label="mesa kappa")
    plt.legend()
    plt.show()

def main():
    
    testCalmSunVsModelS()
    testModelSKappavsMESAKappa()
    testFiniteDifferences()


    testIDLOutput()

if __name__ == "__main__":
    main()
