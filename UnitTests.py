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
    # get calm sun datapoint
    # get model S datapoint
    # compare them
    # have a look at it with your peepers
    pass

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

    dlnP = 1e-3
    logSurfacePressure = np.log(np.loadtxt(modelSFilename, skiprows=1, usecols=2)[0])
    maxDepth = 160  # Mm just some housenumero hehe
    from stateEquationsPT import IdealGas
    calmSun = getCalmSunDatapoint(
        StateEq=IdealGas,
        dlnP=dlnP,
        logSurfacePressure=logSurfacePressure,
        maxDepth=maxDepth,
        surfaceTemperature=surfaceTemperature,
    )
    from stateEquationsPT import IdealGasWithModelSNablaAd
    calmSunHybrid = getCalmSunDatapoint(
        StateEq=IdealGasWithModelSNablaAd,
        dlnP=dlnP,
        logSurfacePressure=logSurfacePressure,
        maxDepth=maxDepth,
        surfaceTemperature=surfaceTemperature,
    )

    toPlot = ["temperatures", "pressures", "rhos"]
    from dataVizualizer import plotSingleTimeDatapoint
    from initialConditionsSetterUpper import loadModelS

    modelS = loadModelS(500)

    axs = plotSingleTimeDatapoint(modelS, toPlot, pltshow=False, label="model S")
    axs = plotSingleTimeDatapoint(calmSunHybrid, toPlot, pltshow=False, label="Ideal gas with model S ∇ad", axs=axs)
    plotSingleTimeDatapoint(calmSun, toPlot, axs=axs, label="Ideal gas")
    plt.legend()

def testModelSBasedIdealGas() -> None:
    resolition = 100

    modelS = loadModelS()
    modelSPressure = modelS.pressures
    modelSTemperature = modelS.temperatures
    from stateEquationsPT import IdealGasWithModelSNablaAd
    
    temperatures = np.logspace(np.log10(modelSTemperature[0]), np.log10(modelSTemperature[-1]), num = resolition)
    pressures = np.logspace(np.log10(modelSPressure[0]), np.log10(modelSPressure[-1]), num = resolition)

    TMesh, PMesh = np.meshgrid(temperatures, pressures)
    nablaAdMesh = IdealGasWithModelSNablaAd.adiabaticLogGradient(TMesh, PMesh)

    plt.pcolormesh(TMesh, PMesh, nablaAdMesh, shading="auto")
    
    plt.loglog(modelSTemperature, modelSPressure, "ok", label="input point")
    plt.xlabel("Temperature [K]")
    plt.ylabel("Pressure [Pa]")
    plt.legend()
    plt.colorbar()
    plt.show()

def main():

    testDataStructureSaveLoad()
    testCalmSunBasedOnModelSData()

    print("Tests passed :)")
    testModelSVSCalmSunVSHybrid()



if __name__ == "__main__":
    main()
