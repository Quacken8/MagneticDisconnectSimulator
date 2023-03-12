#!/usr/bin/env python3
from dataStructure import Data, SingleTimeDatapoint, createDataFromFolder, dictionaryOfVariables
import numpy as np
from initialConditionsSetterUpper import mockupDataSetterUpper
from solvers import oldYSolver
import constants as c
import os


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


def TestDiffEqBasedOnModelSData() -> None:
    #TODO - implement this
    # load model S data
    zs = 
    p_outer
    # run the diff eq
    oldYSolver(zs, innerPs, outerPs, totalMagneticFlux, yGuess)

    # have a look at it with your peepers


def main():
    TestDataStructureSaveLoad()
    TestDiffEqBasedOnModelSData()

    print("Tests passed :)")


if __name__ == "__main__":
    main()
