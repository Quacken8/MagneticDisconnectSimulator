#! usr/bin/env python3
import numpy as np
from dataHandling.dataStructure import SingleTimeDatapoint, subsampleArray

def loadModelS(length: int | None = None) -> SingleTimeDatapoint:
    """
    loads model S into a single time datapoint of length length. If left at None, load the whole array
    """
    pathToModelS = "dataHandling/model_S_new.dat"
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
