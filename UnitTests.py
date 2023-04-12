#!/usr/bin/env python3
from dataStructure import (
    Data,
    SingleTimeDatapoint,
    createDataFromFolder,
    dictionaryOfVariables,
)
import numpy as np
from initialConditionsSetterUpper import mockupDataSetterUpper, loadModelS
from stateEquations import IdealGas
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
    modelTs, modelPs, modelNablas = np.loadtxt(
        pathToModelS, skiprows=1, usecols=(1, 2, 6)
    ).T
    from stateEquations import IdealGas

    idealNablas = IdealGas.adiabaticLogGradient(temperature=modelTs, pressure=modelPs)

    indeces = np.arange(len(modelNablas))
    plt.scatter(indeces, modelNablas, label="model")
    plt.scatter(indeces, idealNablas, label="ideal")
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

def testModelSVSCalmSun() -> None:
    
    # load model S data

    modelSFilename = "externalData/model_S_new.dat"
    surfaceTemperature = np.loadtxt(modelSFilename, skiprows=1, usecols=1)[0]

    dlnP = 0.001
    logSurfacePressure = np.log(np.loadtxt(modelSFilename, skiprows=1, usecols=2)[0])
    maxDepth = 12  # Mm just some housenumero hehe
    convectiveAlpha = 0.3  # value of 0.3 comes from Schüssler Rempel 2018 section 3.2, harmanec brož (stavba a vývoj hvězd) speak of alpha = 2 in section 1.3
    calmSun = getCalmSunDatapoint(
        dlnP=dlnP,
        logSurfacePressure=logSurfacePressure,
        maxDepth=maxDepth,
        surfaceTemperature=surfaceTemperature,
        convectiveAlpha=convectiveAlpha,
    )

    toPlot = ["temperatures", "pressures", "rhos"]
    from dataVizualizer import plotSingleTimeDatapoint
    from initialConditionsSetterUpper import loadModelS

    modelS = loadModelS(500)

    axs = plotSingleTimeDatapoint(modelS, toPlot, pltshow=False, label="model S")
    plotSingleTimeDatapoint(calmSun, toPlot, axs=axs, label="calmSun")

def main():
    testDataStructureSaveLoad()
    testCalmSunBasedOnModelSData()

    print("Tests passed :)")
    #testModelSDensity()
    #testAdiabaticGradientBasedOnModelS()
    testModelSVSCalmSun()



if __name__ == "__main__":
    main()
