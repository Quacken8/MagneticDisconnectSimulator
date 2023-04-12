#!/usr/bin/env python3
from dataStructure import Data, SingleTimeDatapoint, createDataFromFolder, dictionaryOfVariables
import numpy as np
from initialConditionsSetterUpper import mockupDataSetterUpper
#from solvers import oldYSolver
from initialConditionsSetterUpper import modelSLoader
import constants as c
import os
from matplotlib import pyplot as plt

pathToModelS = "externalData/model_S_new.dat"

def TestDataStructureSaveLoad() -> None:
    datapoint = mockupDataSetterUpper(zLength=17)

    data = Data(finalT=3*c.hour, numberOfTSteps=4)

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

        for key, saved, loaded in zip(savedVariables.keys(), savedVariables.values(), loadedVariables.values()):
            try:
                assert np.allclose(saved, loaded)
            except TypeError:
                pass


def TestCalmSunBasedOnModelSData() -> None:
    # get calm sun datapoint
    # get model S datapoint
    # compare them
    # have a look at it with your peepers
    pass

def TestAdiabaticGradientBasedOnModelS() -> None:
    modelTs, modelPs, modelNablas = np.loadtxt(pathToModelS, skiprows=1, usecols = (1, 2, 6)).T
    from stateEquations import IdealGas
    idealNablas = IdealGas.adiabaticLogGradient(temperature = modelTs, pressure = modelPs)
    
    indeces = np.arange(len(modelNablas))
    plt.scatter(indeces, modelNablas, label = "model")
    plt.scatter(indeces, idealNablas, label = "ideal")
    plt.legend()
    plt.show()


def main():
    TestDataStructureSaveLoad()
    TestCalmSunBasedOnModelSData()

    print("Tests passed :)")

    TestAdiabaticGradientBasedOnModelS()

if __name__ == "__main__":
    main()
