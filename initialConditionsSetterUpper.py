#!/usr/bin/env python3
import numpy as np
from dataStructure import SingleTimeDatapoint, Data, subsampleArray
import constants as c


def getInitialConditions(numberOfZSteps: int, maxDepth: float) -> SingleTimeDatapoint:
    """

    """
    maxDepth *= c.Mm
    raise NotImplementedError()


def mockupDataSetterUpper(zLength: int = 10) -> SingleTimeDatapoint:
    """
    mockup data setter upper that just makes bunch of weird datapoints instead of the pressures and other datapoints of length zLength
    """
    ones = np.arange(zLength)
    maxdepth = 4
    zs = np.linspace(0, maxdepth, num=zLength)

    toReturn = SingleTimeDatapoint(temperatures=ones, pressures=ones*10, B_0s=ones*100, F_rads=ones*1000, F_cons=ones *
                                   10000, entropies=ones*2, nablaAds=ones*4, deltas=ones*6, zs=zs, rhos=ones*7, cps=ones*3, cvs=ones*11)
    return toReturn


def modelSLoader(length: int) -> SingleTimeDatapoint:
    """
    loads model S into a single time datapoint of length length
    """
    pathToModelS = "externalData/model_S_new.dat"
    unitSymbolBracket = '['
    header = np.loadtxt(pathToModelS, dtype=str, max_rows=1)
    variableNames = []
    for word in header:
        variableNames.append(word.split(unitSymbolBracket)[0])
    
    allLoadedData = np.loadtxt(pathToModelS, skiprows=1)
    variablesDictionary = {}
    for i, variableName in enumerate(variableNames):
        variablesDictionary[variableName] = subsampleArray(allLoadedData[:,i], length)
        if variableName == 'r':
            variablesDictionary['zs'] = variablesDictionary[variableName][0] - variablesDictionary[variableName]
            variablesDictionary.pop('r')


    datapoint = SingleTimeDatapoint(**variablesDictionary)

    return datapoint

def main():
    """debugging"""
    datapoint = modelSLoader(5)
    for key, value in datapoint.allVariables.items():
        print(key, value)

if __name__ == "__main__":
    main()