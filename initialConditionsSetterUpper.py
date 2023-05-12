#!/usr/bin/env python3
import numpy as np
from dataStructure import SingleTimeDatapoint, Data, subsampleArray
import constants as c


def getInitialConditions(
    numberOfZSteps: int,
    maxDepth: float,
    minDepth: float,
    surfaceTemperature: float = 3500, # value used by Schüssler and Rempel 2005
) -> SingleTimeDatapoint:
    """
    returns a datapoint that corresponds to flux tube at t=0 with linear magnetic field but otherwise model S
    """
    maxDepth *= c.Mm
    minDepth *= c.Mm
    zs = np.linspace(minDepth, maxDepth, num=numberOfZSteps)

    modelS = loadModelS()

    initialPs = np.interp(zs, modelS.zs, modelS.pressures)
    initialTs = np.interp(zs, modelS.zs, modelS.temperatures)
    initialBs = 0

    raise NotImplementedError()


def mockupDataSetterUpper(zLength: int = 10) -> SingleTimeDatapoint:
    """
    mockup data setter upper that just makes bunch of weird datapoints instead of the pressures and other datapoints of length zLength
    """
    ones = np.arange(zLength)
    maxdepth = 4
    zs = np.linspace(0, maxdepth, num=zLength)

    toReturn = SingleTimeDatapoint(
        temperatures=ones,
        pressures=ones * 10,
        B_0s=ones * 100,
        F_rads=ones * 1000,
        F_cons=ones * 10000,
        entropies=ones * 2,
        nablaads=ones * 4,
        deltas=ones * 6,
        zs=zs,
        rhos=ones * 7,
        cps=ones * 3,
        cvs=ones * 11,
    )
    return toReturn


def loadModelS(length: int | None = None) -> SingleTimeDatapoint:
    """
    loads model S into a single time datapoint of length length. If left at None, load the whole array
    """
    pathToModelS = "externalData/model_S_new.dat"
    unitSymbolBracket = "["
    header = np.loadtxt(pathToModelS, dtype=str, max_rows=1)
    variableNames = []
    for word in header:
        variableNames.append(word.split(unitSymbolBracket)[0])

    allLoadedData = np.loadtxt(pathToModelS, skiprows=1)
    if length is None:
        length = len(allLoadedData[:, 0]) + 1
    variablesDictionary = {}
    for i, variableName in enumerate(variableNames):
        variablesDictionary[variableName] = subsampleArray(allLoadedData[:, i], length)
        if variableName == "r":
            variablesDictionary["zs"] = (
                variablesDictionary[variableName][0] - variablesDictionary[variableName]
            )
            variablesDictionary.pop("r")

    datapoint = SingleTimeDatapoint(**variablesDictionary)

    return datapoint


def main():
    pass


if __name__ == "__main__":
    main()
