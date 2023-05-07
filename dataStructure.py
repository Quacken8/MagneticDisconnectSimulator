#!/usr/bin/env python3
import numpy as np
import constants as c
import os
import logging
L = logging.getLogger(__name__)


def subsampleArray(array: np.ndarray, desiredNumberOfElements: int) -> np.ndarray:
    """
    returns a subsampled 1D array that has the desiredNumberOfElements
    """
    if np.ndim(array) != 1:
        raise ValueError("array must be 1D")
    if len(array) < desiredNumberOfElements:
        return array
    else:
        indices = np.linspace(0, len(array) - 1, desiredNumberOfElements, endpoint=True, dtype=int)
        return array[indices]


def dictionaryOfVariables(A: object) -> dict:
    """makes dictionary of variableName value pairs of a class"""
    dic = {
        key: value
        for key, value in A.__dict__.items()
        if not key.startswith("__") and not callable(key)
    }

    return dic


"""
this dictionary is used to put units next to values that come up in the simulatuin. It is expected that the units went through UX parse, i.e. units as hours and megameters are used for readibility
"""
unitsDictionary = {
    "times": "h",
    "zs": "m",
    "hs": "m",
    "numberofzsteps": "1",
    "maxdepth": "Mm",
    "temperatures": "K",
    "pressures": "Pa",
    "rhos": "kg/m^3",
    "b_0s": "T",
    "f_rads": "W/m^2",
    "f_cons": "W/m^2",
    "entropies": "J/K",
    "nablaads": "1",
    "nablarads": "1",
    "nablatots": "1",
    "cps": "J/(kg.K)",
    "cvs": "J/(kg.K)",
    "deltas": "1",
    "kappas": "m^2/kg",
    "gamma1s": "1",
}


class SingleTimeDatapoint:
    """
    class that saves an instanteanous state of the simulation. Similar to Data class, but only for one t
    all datapoints are expected in SI

    NumberOfZStepsPower: the number of steps in z direction will be 2^k + 1 where k is the NumberOfZStepsPower
    """

    def __init__(
        self,
        temperatures: np.ndarray,
        pressures: np.ndarray,
        zs: np.ndarray,
        **kwargs,
    ) -> None:
        self.zs = zs
        self.numberOfZSteps = len(zs)
        self.temperatures = temperatures
        self.pressures = pressures
        fundamentalVariables = dictionaryOfVariables(self)
        self.derivedQuantities = {}

        for key, value in kwargs.items():
            self.derivedQuantities[key] = value
        self.maxDepth = zs[-1]

        self.allVariables = (
            fundamentalVariables | self.derivedQuantities
        )  # that '|' is just yummy python syntax sugar merging two dicts

        # check whether all arrays have the same length
        for variableName, variableValue in self.allVariables.items():
            try:
                if len(variableValue) != self.numberOfZSteps:
                    raise ValueError(
                        f"Some of your variables ({variableName}) have the wrong length ({len(variableValue)})"
                    )
            except TypeError:  # so scalars dont get tested for length
                pass

        # check whether we have units for all variables

        for variableName in self.allVariables.keys():
            try:
                unitsDictionary[variableName.lower()]
            except KeyError:
                raise ValueError(f"We don't have units for {variableName} defined!")


class Data:
    """
    Class for handling all data. Is initialized with the whole size for speed reasons, might be a problem later lol
    ----
    VARIABLES
    ----
    Ts, 2D np array of doubles: temperatures in Kelvin. First index stands for z index (i.e. depth), second for t index (i.e. time)
    pressures, 2D np array of doubles: temperatures in Kelvin. First index stands for z index (i.e. depth), second for t index (i.e. time)
    """

    def __init__(self, finalT: float, numberOfTSteps: int, startT: float = 0):
        if startT == finalT:
            raise ValueError("your start T equals end T")
        self.finalT = finalT
        self.times = np.linspace(start=startT, stop=self.finalT, num=numberOfTSteps)
        self.dt = self.times[1]  # TODO - uniform timestep might not be the smartest
        L.warn("Assuming uniform timestep when creating data")
        self.numberOfTSteps = numberOfTSteps

        self.datapoints = np.empty(self.numberOfTSteps, dtype=SingleTimeDatapoint)
        self.occupiedElements = 0

    def addDatapointAtIndex(self, datapoint: SingleTimeDatapoint, index: int) -> None:
        """
        adds SingleTimeDatapoint to the data at index
        """
        if index > self.occupiedElements:
            raise ValueError(
                "You're trying to add a datapoint to an index while a previous one isn't filled"
            )
        self.datapoints[index] = datapoint
        if index == self.occupiedElements:
            self.occupiedElements += 1

    def appendDatapoint(self, datapoint: SingleTimeDatapoint) -> None:
        """
        appends datapoint at the end of the data cube
        """

        self.addDatapointAtIndex(datapoint=datapoint, index=self.occupiedElements)

    def saveToFolder(self, outputFolderName: str, rewriteFolder=False) -> None:
        """
        creates folder which contains csv files of the simulation
        """
        if rewriteFolder:
            os.system(f"rm -r {outputFolderName}")
        os.mkdir(outputFolderName)

        superDictionary = (
            {}
        )  # this dictionary is just like the "allVariables" dictionary of the single time datapoint but these contain cubes of data
        superDictionary.update({"times": self.times})
        firstDatapoint = self.datapoints[0]
        superDictionary.update(firstDatapoint.allVariables)
        for datapoint in self.datapoints[1:]:
            for variableName, variableArray in datapoint.allVariables.items():
                # to variableName key in superDictionary append variableArray
                superDictionary[variableName] = np.vstack(
                    (superDictionary[variableName], variableArray)
                )

        # now that all the data is in the one dict, save it
        for variableName, variableArray in superDictionary.items():
            filename = f"{outputFolderName}/{variableName}.csv"

            # unit conversion
            unitConversionFactor = 1
            if "times" in variableName:
                unitConversionFactor = 1 / c.hour
            if "zs" in variableName:
                unitConversionFactor = 1 / c.Mm

            # headers
            if np.ndim(variableArray) == 0:
                continue
            if np.ndim(variableArray) == 1:
                header = f"{variableName} [{unitsDictionary[variableName.lower()]}]"
            elif np.ndim(variableArray) == 2:
                header = f"{variableName} [{unitsDictionary[variableName.lower()]}], rows index depth, columns index time"
            else:
                raise ValueError(
                    f"Weird dimension of {variableName} array ({np.ndim(variableArray)})"
                )

            # and save
            np.savetxt(
                filename,
                variableArray.T * unitConversionFactor,
                header=header,
                delimiter=",",
            )


def createDataFromFolder(foldername: str) -> Data:
    """
    creates a data cube from a folder
    """

    times = np.loadtxt(f"{foldername}/times.csv", skiprows=1, delimiter=",") * c.hour
    toReturn = Data(finalT=times[-1], numberOfTSteps=len(times), startT=times[0])

    loadedVariables = {}
    folder = os.listdir(foldername)
    folder.remove("times.csv")
    for file in folder:
        variableName = file[:-4]
        unitConversionFactor = 1
        if "zs" in variableName:
            unitConversionFactor = c.Mm
        loadedVariables[variableName] = (
            np.loadtxt(f"{foldername}/{variableName}.csv", skiprows=1, delimiter=",")
            * unitConversionFactor
        ).T

    for i, _ in enumerate(times):
        thisTimesVariables = {}
        for key, value in loadedVariables.items():
            if np.ndim(value) == 0 or np.ndim(value) == 1:
                continue
            thisTimesVariables[key] = value[i, :]
        newDatapoint = SingleTimeDatapoint(**thisTimesVariables)
        toReturn.appendDatapoint(newDatapoint)

    return toReturn


def loadOneTimeDatapoint(foldername: str, time: float = 0.0) -> SingleTimeDatapoint:
    """
    loads a single time datapoint from the data folder. If no time is given, loads the first one
    """
    loadedDataCube = createDataFromFolder(foldername)
    timeIndex = np.argmin(np.abs(loadedDataCube.times - time))
    return loadedDataCube.datapoints[timeIndex]
