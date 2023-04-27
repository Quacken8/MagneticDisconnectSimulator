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


def testDataStructureSaveLoad() -> None:
    datapoint = mockupDataSetterUpper(zLength=17)

    data = Data(finalT=3 * c.hour, numberOfTSteps=4)

    data.appendDatapoint(datapoint)
    data.appendDatapoint(datapoint)
    data.appendDatapoint(datapoint)
    data.appendDatapoint(datapoint)

    foldername = "LoadTest"
    try:
        data.saveToFolder(foldername)
        loadedData = createDataFromFolder(foldername=foldername)
    finally:
        os.system(f"rm -r {foldername}")

    for i, _ in enumerate(data.times):
        savedVariables = dictionaryOfVariables(data.datapoints[i])
        loadedVariables = dictionaryOfVariables(loadedData.datapoints[i])

        for key, saved, loaded in zip(
            savedVariables.keys(), savedVariables.values(), loadedVariables.values()
        ):
            try:
                assert np.allclose(saved, loaded)
            except TypeError:
                pass

def testCalmSunBasedOnModelSData() -> None:
    # get model S datapoint
    modelS = loadModelS()
    surfacePressure = modelS.pressures[0]
    surfaceTemperature = modelS.temperatures[0]
    surfaceZ = modelS.zs[0]

    # get calm sun datapoint
    from stateEquationsPT import IdealGas
    maxDepth = 16 * c.Mm
    dlnP = 1e-1
    calmSun = getCalmSunDatapoint(
        lnSurfacePressure=np.log(surfacePressure),
        surfaceTemperature=surfaceTemperature,
        surfaceZ=surfaceZ,
        maxDepth=maxDepth,
        dlnP=dlnP,
        StateEq=IdealGas,
    )

    # have a look at it with your peepers
    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(modelS, toPlot, pltshow=False, label="model S")
    plotSingleTimeDatapoint(calmSun, toPlot, axs=axs, label="calm sun")
    plt.show()



def testAdiabaticGradientBasedOnModelS() -> None:
    modelS = loadModelS(1000)

    modelTs = modelS.temperatures
    modelPs = modelS.pressures
    modelZs = modelS.zs
    modelNablas = modelS.derivedQuantities["nablaads"]
    from stateEquationsPT import IdealGas

    idealNablas = IdealGas.adiabaticLogGradient(temperature=modelTs, pressure=modelPs)

    plt.scatter(modelZs, modelNablas, label="model")
    plt.scatter(modelZs, idealNablas, label="ideal")
    plt.legend()
    plt.show()

def testModelSDensity() -> None:
    """
    test that the density of model S is similar to the density from ideal gas
    """
    modelSDatapoint = loadModelS(1000)
    modelSpressure = modelSDatapoint.pressures
    modelStemperature = modelSDatapoint.temperatures
    idealRhos = IdealGas.density(modelStemperature, modelSpressure)

    axs = plotSingleTimeDatapoint(modelSDatapoint, ["rhos"], pltshow=False)
    axs["rhos"].loglog(modelSDatapoint.zs / c.Mm, idealRhos, label="ideal")
    plt.legend()
    plt.show()

def testVizualization() -> None:
    from initialConditionsSetterUpper import loadModelS

    datapoint = loadModelS(500)

    toPlot = ["temperatures", "pressures", "rhos"]

    plotSingleTimeDatapoint(datapoint, toPlot, log=True)

def testModelSVSCalmSunVSHybrid() -> None:
    # load model S data

    modelSFilename = "externalData/model_S_new.dat"
    surfaceTemperature = np.loadtxt(modelSFilename, skiprows=1, usecols=1)[0]

    dlnP = 1e-1
    logSurfacePressure = np.log(np.loadtxt(modelSFilename, skiprows=1, usecols=2)[0])
    maxDepth = 160*c.Mm  # just some housenumero hehe
    surfaceZ = 0
    from stateEquationsPT import IdealGas

    calmSun = getCalmSunDatapoint(
        StateEq=IdealGas,
        dlnP=dlnP,
        lnSurfacePressure=logSurfacePressure,
        maxDepth=maxDepth,
        surfaceTemperature=surfaceTemperature,
        surfaceZ=surfaceZ,
    )
    from stateEquationsPT import IdealGasWithModelSNablaAd

    calmSunHybrid = getCalmSunDatapoint(
        StateEq=IdealGasWithModelSNablaAd,
        dlnP=dlnP,
        lnSurfacePressure=logSurfacePressure,
        maxDepth=maxDepth,
        surfaceTemperature=surfaceTemperature,
        surfaceZ=surfaceZ,
    )

    toPlot = ["temperatures", "pressures"]
    from dataVizualizer import plotSingleTimeDatapoint
    from initialConditionsSetterUpper import loadModelS

    modelS = loadModelS(500)

    axs = plotSingleTimeDatapoint(modelS, toPlot, pltshow=False, label="model S", log= False)
    axs = plotSingleTimeDatapoint(
        calmSunHybrid,
        toPlot,
        pltshow=False,
        label="Ideal gas with model S ∇ad",
        axs=axs,
        log = False
    )
    plotSingleTimeDatapoint(calmSun, toPlot, axs=axs, label="Ideal gas", log = True)
    plt.legend()

def testModelSBasedIdealGas() -> None:
    resolition = 100

    modelS = loadModelS()
    modelSPressure = modelS.pressures
    modelSTemperature = modelS.temperatures
    from stateEquationsPT import IdealGasWithModelSNablaAd

    temperatures = np.logspace(
        np.log10(modelSTemperature[0]), np.log10(modelSTemperature[-1]), num=resolition
    )
    pressures = np.logspace(
        np.log10(modelSPressure[0]), np.log10(modelSPressure[-1]), num=resolition
    )

    TMesh, PMesh = np.meshgrid(temperatures, pressures)
    nablaAdMesh = IdealGasWithModelSNablaAd.adiabaticLogGradient(TMesh, PMesh)

    plt.pcolormesh(TMesh, PMesh, nablaAdMesh, shading="auto")

    plt.loglog(modelSTemperature, modelSPressure, "ok", label="input point")
    plt.xlabel("Temperature [K]")
    plt.ylabel("Pressure [Pa]")
    plt.legend()
    plt.colorbar()
    plt.show()

def testModelSHvsIdealGasH() -> None:
    modelS = loadModelS()
    modelSPressure = modelS.pressures
    modelSTemperature = modelS.temperatures
    modelSZs = modelS.zs
    from gravity import g
    modelSZs = np.loadtxt("externalData/H_p.dat", usecols=0)
    modelSZs = modelSZs[0] - modelSZs

    gravities = np.array(g(modelSZs))
    from stateEquationsPT import IdealGasWithModelSNablaAd

    idealHs = IdealGasWithModelSNablaAd.pressureScaleHeight(
        modelSTemperature, modelSPressure, gravities
    )
    modelSHs = np.loadtxt("externalData/H_p.dat", usecols=1)

    print(gravities[0])

    plt.loglog(modelSZs, idealHs, label="ideal")
    plt.loglog(modelSZs, modelSHs, label="from model")
    plt.xlabel("z [m]")
    plt.ylabel("H [m]")
    plt.legend()
    plt.show()

def compareCalmVsHybridToOldData() -> None:
    
    modelSFilename = "externalData/model_S_new.dat"
    surfaceTemperature = np.loadtxt(modelSFilename, skiprows=1, usecols=1)[0]

    dlnP = 1
    logSurfacePressure = np.log(np.loadtxt(modelSFilename, skiprows=1, usecols=2)[0])
    maxDepth = 30*c.Mm  # just some housenumero hehe
    surfaceZ = 0
    from stateEquationsPT import IdealGas

    calmSun = getCalmSunDatapoint(
        StateEq=IdealGas,
        dlnP=dlnP,
        lnSurfacePressure=logSurfacePressure,
        maxDepth=maxDepth,
        surfaceTemperature=surfaceTemperature,
        surfaceZ=surfaceZ,
    )
    from stateEquationsPT import IdealGasWithModelSNablaAd

    calmSunHybrid = getCalmSunDatapoint(
        StateEq=IdealGasWithModelSNablaAd,
        dlnP=dlnP,
        lnSurfacePressure=logSurfacePressure,
        maxDepth=maxDepth,
        surfaceTemperature=surfaceTemperature,
        surfaceZ=surfaceZ,
    )

    toPlot = ["temperatures", "pressures"]
    from dataVizualizer import plotSingleTimeDatapoint
    from initialConditionsSetterUpper import loadModelS

    modelS = loadModelS(500)

    axs = plotSingleTimeDatapoint(modelS, toPlot, pltshow=False, label="model S", log= False)
    axs = plotSingleTimeDatapoint(
        calmSunHybrid,
        toPlot,
        pltshow=False,
        label="Ideal gas with model S ∇ad",
        axs=axs,
        log = False
    )
    axs = plotSingleTimeDatapoint(calmSun, toPlot, axs=axs, label="Ideal gas", log = False, pltshow=False)

    oldZs = np.loadtxt("externalData/oldData", usecols=1, skiprows = 1)*c.cm/c.Mm
    oldTemperatures = np.loadtxt("externalData/oldData", usecols=2, skiprows = 1)
    oldPressures = np.loadtxt("externalData/oldData", usecols=3, skiprows = 1)*c.barye

    axs['temperatures'].loglog(oldZs, oldTemperatures, label="old data")
    axs['pressures'].loglog(oldZs, oldPressures, label="old data")
    plt.legend()

    plt.show()

def testMESAEOSvsModelSdensity() -> None:
    modelS = loadModelS()
    modelSPressure = modelS.pressures
    modelSTemperature = modelS.temperatures
    modelSZs = modelS.zs
    modelSDensity = modelS.derivedQuantities["rhos"]

    simpleSolarAbundances = {"h1" : 0.7, "he4" : 0.3}

    from stateEquationsPT import MESAEOS
    mesaDensities = MESAEOS.density(modelSTemperature, modelSPressure)

    print(f"pressure = {modelSPressure[0]} Pa \n temperature = {modelSTemperature[0]} K \n density = {modelSDensity[0]} kg/m^3 \n mesaDensity = {MESAEOS.density(modelSTemperature[0], modelSPressure[0])} kg/m^3")

    plt.loglog(modelSZs, modelSDensity, label="model S")
    plt.loglog(modelSZs, mesaDensities, label="MESA")
    plt.xlabel("z [m]")
    plt.ylabel("density [kg/m^3]")
    plt.legend()
    plt.show()

def testMESAEOSvsIdealGas() -> None:
    modelS = loadModelS()
    modelSPressure = modelS.pressures
    modelSTemperature = modelS.temperatures

    from stateEquationsPT import MESAEOS
    mesaDensities = MESAEOS.density(modelSTemperature, modelSPressure)
    mesaMu = MESAEOS.meanMolecularWeight(modelSTemperature, modelSPressure)
    mesaNablaAd = MESAEOS.adiabaticLogGradient(modelSTemperature, modelSPressure)

    from stateEquationsPT import IdealGas
    idealDensities = IdealGas.density(modelSTemperature, modelSPressure)
    idealMu = IdealGas.meanMolecularWeight(modelSTemperature, modelSPressure)
    idealNablaAd = IdealGas.adiabaticLogGradient(modelSTemperature, modelSPressure)

    # now plot them in three separate figures

    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].loglog(modelSTemperature, mesaDensities, label="MESA")
    axs[0].loglog(modelSTemperature, idealDensities, label="ideal gas")
    axs[0].set_ylabel("density [kg/m^3]")
    axs[0].legend()

    axs[1].loglog(modelSTemperature, mesaMu, label="MESA")
    axs[1].loglog(modelSTemperature, idealMu, label="ideal gas")
    axs[1].set_ylabel("mu [1]")
    axs[1].legend()

    axs[2].loglog(modelSTemperature, mesaNablaAd, label="MESA")
    axs[2].loglog(modelSTemperature, idealNablaAd, label="ideal gas")
    axs[2].set_ylabel("nabla ad")
    axs[2].set_xlabel("temperature [K]")
    axs[2].legend()

    plt.show()

def testMESAEOSvsModelSnablaAd() -> None:
    modelS = loadModelS()
    modelSPressure = modelS.pressures
    modelSTemperature = modelS.temperatures
    modelSZs = modelS.zs
    modelSDensity = modelS.derivedQuantities["rhos"]
    modelSNablaAds = modelS.derivedQuantities["nablaads"]

    from stateEquationsPT import MESAEOS
    mesaDensities = MESAEOS.density(modelSTemperature, modelSPressure)
    mesaNablaAd = MESAEOS.adiabaticLogGradient(modelSTemperature, modelSPressure)

    plt.plot(modelSZs, modelSNablaAds, label="model S")
    plt.plot(modelSZs, mesaNablaAd, label="MESA")
    plt.xlabel("z [m]")
    plt.ylabel("nabla ad")
    plt.legend()
    plt.show()

def testModelSVsCalmSunWithSvandaRad() -> None:
    modelS = loadModelS()
    modelSZs = modelS.zs
    modelSPressure = modelS.pressures
    modelSTemperature = modelS.temperatures

    dlnP = 1e-1
    maxDepth = 20*c.Mm
    logSurfacePressure = np.log(modelSPressure[0])
    surfaceTemperature = modelSTemperature[0]
    surfaceZ = 0

    from stateEquationsPT import IdealGasWithSvandasNablaRads
    calmSun = getCalmSunDatapoint(
        StateEq=IdealGasWithSvandasNablaRads,
        dlnP=dlnP,
        lnSurfacePressure=logSurfacePressure,
        maxDepth=maxDepth,
        surfaceTemperature=surfaceTemperature,
        surfaceZ=surfaceZ,
    )

    calmSunPressure = calmSun.pressures
    calmSunZs = calmSun.zs
    calmSunTemperature = calmSun.temperatures

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].loglog(modelSZs, modelSTemperature, label="model S")
    axs[0].loglog(calmSunZs, calmSunTemperature, label="ideal gas")
    axs[0].set_ylabel("temperature [K]")
    axs[0].legend()

    axs[1].loglog(modelSZs, modelSPressure, label="model S")
    axs[1].loglog(calmSunZs, calmSunPressure, label="ideal gas")
    axs[1].set_ylabel("pressure [Pa]")
    axs[1].set_xlabel("z [m]")
    axs[1].legend()

    plt.show()

def testModelSIdealNablaRadvsSvandasNablaRad() -> None:
    modelS = loadModelS()
    modelSZs = modelS.zs
    modelSPressure = modelS.pressures
    modelSTemperature = modelS.temperatures

    externalZs, externalNablarads = np.loadtxt("debuggingReferenceFromSvanda/nablas.dat", usecols = (0,1), skiprows = 1, unpack=True)

    from stateEquationsPT import IdealGasWithSvandasNablaRads
    from gravity import g
    gravAcss = np.array(g(modelSZs))
    nablaRads = IdealGasWithSvandasNablaRads.radiativeLogGradient(modelSTemperature, modelSPressure, gravAcss)

    from stateEquationsPT import IdealGas
    idealNablaRads = IdealGas.radiativeLogGradient(modelSTemperature, modelSPressure, gravAcss)

    plt.loglog(modelSZs, nablaRads, label="ideal with Svanda equation")
    plt.loglog(modelSZs, idealNablaRads, label="ideal gas")
    plt.loglog(externalZs, externalNablarads, label="Svanda data")
    plt.xlabel("z [m]")
    plt.ylabel("nabla rad")
    plt.legend()
    plt.show()

def testCalmSunWithMESAvsModelS() -> None:
    modelS = loadModelS()
    modelSZs = modelS.zs
    modelSPressure = modelS.pressures
    modelSTemperature = modelS.temperatures

    dlnP = 1e-1
    maxDepth = 20*c.Mm
    logSurfacePressure = np.log(modelSPressure[0])
    surfaceTemperature = modelSTemperature[0]
    surfaceZ = 0

    from stateEquationsPT import IdealGasWithSvandasNablaRads
    calmSun = getCalmSunDatapoint(
        StateEq=IdealGasWithSvandasNablaRads,
        dlnP=dlnP,
        lnSurfacePressure=logSurfacePressure,
        maxDepth=maxDepth,
        surfaceTemperature=surfaceTemperature,
        surfaceZ=surfaceZ,
    )

    calmSunPressure = calmSun.pressures
    calmSunZs = calmSun.zs
    calmSunTemperature = calmSun.temperatures

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].loglog(modelSZs, modelSTemperature, label="model S")
    axs[0].loglog(calmSunZs, calmSunTemperature, label="MESA")
    axs[0].set_ylabel("temperature [K]")
    axs[0].legend()

    axs[1].loglog(modelSZs, modelSPressure, label="model S")
    axs[1].loglog(calmSunZs, calmSunPressure, label="MESA")
    axs[1].set_ylabel("pressure [Pa]")
    axs[1].set_xlabel("z [m]")
    axs[1].legend()

    plt.show()

def testModelSOpacityVSMesaOpacity() -> None:
    modelS = loadModelS()
    modelSZs = modelS.zs
    modelSTemperature = modelS.temperatures
    modelSDensity = modelS.derivedQuantities["rhos"]
    modelSKappa = modelS.derivedQuantities["kappas"]

    from mesa2Py.kappaFromMesa import getMesaOpacity
    mesaOutput = getMesaOpacity(modelSDensity, modelSTemperature)

    mesaKappa = [output.tolist().kappa for output in mesaOutput]

    plt.loglog(modelSZs, modelSKappa, label="model S")
    plt.loglog(modelSZs, mesaKappa, label="MESA")
    plt.xlabel("z [m]")
    plt.ylabel("opacity [m^2/kg]")
    plt.legend()
    plt.show()

def testModelSOpacityVsMesaOpacityDirectly() -> None:

    modelS = loadModelS()
    modelSZs = modelS.zs
    modelSTemperature = modelS.temperatures
    modelSDensity = modelS.derivedQuantities["rhos"]
    modelSKappa = modelS.derivedQuantities["kappas"]

    from mesa2Py.kappaFromMesa import getMESAOpacityRhoT

    mesaKappa = getMESAOpacityRhoT(modelSDensity, modelSTemperature)

    plt.loglog(modelSZs, modelSKappa, label="model S")
    plt.loglog(modelSZs, mesaKappa, label="MESA")
    plt.xlabel("z [m]")
    plt.ylabel("opacity [m^2/kg]")
    plt.legend()
    plt.show()
    

def main():
    testModelSOpacityVsMesaOpacityDirectly()

if __name__ == "__main__":
    main()
