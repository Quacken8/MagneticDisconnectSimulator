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

    axs = plotSingleTimeDatapoint(modelS, toPlot, pltshow=False, label="model S")
    axs = plotSingleTimeDatapoint(
        calmSunHybrid,
        toPlot,
        pltshow=False,
        label="Ideal gas with model S ∇ad",
        axs=axs,
    )
    plotSingleTimeDatapoint(calmSun, toPlot, axs=axs, label="Ideal gas")
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


def main():

    
    testModelSVSCalmSunVSHybrid()
    input()


if __name__ == "__main__":
    main()
