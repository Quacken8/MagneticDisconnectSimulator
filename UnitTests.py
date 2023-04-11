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

        savedVariables = dictionaryOfVariables(data.values[i])
        loadedVariables = dictionaryOfVariables(loadedData.values[i])

        for key, saved, loaded in zip(savedVariables.keys(), savedVariables.values(), loadedVariables.values()):
            assert np.array_equal(saved, loaded)


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
    differencesOfNablas = np.abs(modelNablas - idealNablas)
    print(differencesOfNablas.max)


def main():
    TestDataStructureSaveLoad()
    TestCalmSunBasedOnModelSData()

    print("Tests passed :)")


    T = 7938.461188
    P = 14465.32707

    rootFinderConstantPart = (
        -2.5 * np.log10(T)
        + (13.53 * 5040) / T
        + 0.48
        + np.log10(c.massFractionOfHydrogen)
        + np.log10(P*c.barye)    # FIXME AAAAAAAAAAAA IS THERE A BARYE (CGS UNIT FOR PRESSURE) HERE OR NOT AAAA
        + np.log10(c.meanMolecularWeight * c.gram)
    )
    def toFindRoot(x):
        return np.log10(x * x / (1 - x * x)) + rootFinderConstantPart
    xs = np.linspace(0,1, num = 10000)
    plt.plot(xs, toFindRoot(xs))
    plt.show()
    from stateEquations import IdealGas

    x = IdealGas.degreeOfIonization(T, P)

    print(x)

    TestAdiabaticGradientBasedOnModelS()

if __name__ == "__main__":
    main()
