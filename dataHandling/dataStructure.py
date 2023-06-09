#!/usr/bin/env python3
import numpy as np
import constants as c
import os
import loggingConfig
import logging

L = loggingConfig.configureLogging(logging.INFO, __name__)

def padTillNotJagged(array, padValue=np.nan) -> np.ndarray:
    """
    pads a jagged array with padValue until it is not jagged anymore
    """
    maxLength = max([len(subarray) for subarray in array])
    for i, subarray in enumerate(array):
        if len(subarray) < maxLength:
            missingLength = maxLength - len(subarray)
            newarray = np.pad(
                subarray,
                (0, missingLength),
                mode="constant",
                constant_values=padValue,
            )
            array[i] = newarray
    return np.array(array)

def subsampleArray(array: np.ndarray, desiredNumberOfElements: int) -> np.ndarray:
    """
    returns a subsampled 1D array that has the desiredNumberOfElements
    """
    if np.ndim(array) != 1:
        raise ValueError("array must be 1D")
    if len(array) < desiredNumberOfElements:
        return array
    else:
        indices = np.linspace(
            0, len(array) - 1, desiredNumberOfElements, endpoint=True, dtype=int
        )
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
    "maxdepth": "m",
    "temperatures": "K",
    "pressures": "Pa",
    "rhos": "kg/m^3",
    "bs": "T",
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
    """

    def __init__(
        self,
        temperatures: np.ndarray,
        pressures: np.ndarray,
        zs: np.ndarray,
        bs: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        self.zs = zs
        self.numberOfZSteps = len(zs)
        self.temperatures = temperatures

        self.pressures = pressures
        if bs is None:
            bs = np.zeros_like(zs)
        self.bs = bs

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
                L.warning(f"We don't have units for {variableName} defined!")

    def regularizeGrid(self, nSteps: int | None = None) -> None:
        """turns current z grid into an equidistant one by linearly interpolating all variables onto it. If nSteps is given, the grid will have nSteps elements, otherwise it will use the current grid size

        Args:
            nSteps (int | None, optional): How many elements the new grid should have. Defaults to None, i.e. same number of elements as before the regularization.
        """

        if nSteps is None:
            nSteps = self.numberOfZSteps

        newZs = np.linspace(self.zs[0], self.zs[-1], nSteps)
        self.temperatures = np.interp(newZs, self.zs, self.temperatures)
        self.pressures = np.interp(newZs, self.zs, self.pressures)

        for key, value in self.derivedQuantities.items():
            self.derivedQuantities[key] = np.interp(newZs, self.zs, value)

        self.zs = newZs

    def saveToFolder(self, folderName, rewrite: bool = False):
        """
        saves all variables to a folder
        """
        L.info(f"Saving to {folderName}")
        if rewrite:
            os.system(f"rm -r {folderName}")
        os.mkdir(folderName)
        for variableName, variableValue in self.allVariables.items():
            if np.ndim(variableValue) == 0:
                variableValue = np.array([variableValue])
            np.savetxt(
                f"{folderName}/{variableName}.csv",
                variableValue,
                header=f"{variableName} [{unitsDictionary[variableName.lower()]}]",
                delimiter=",",
            )


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
        self.finalT = finalT
        self.times = np.linspace(start=startT, stop=self.finalT, num=numberOfTSteps)
        L.info("Assuming uniform timestep when creating data")
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
        
        # since times are separate from the datapoints, they get saved separately
        np.savetxt(
            f"{outputFolderName}/times.csv",
            self.times,
            header=f"t [{unitsDictionary['times']}]",
            delimiter=",",
        )

        firstDatapoint = self.datapoints[0]
        superDictionary.update(firstDatapoint.allVariables)
        # since each time step has different zs the super dictionary will be jagged
        # therefore we can't use np arrays and have to convert to lists
        for key, value in superDictionary.items():
            superDictionary[key] = [value]
        for datapoint in self.datapoints[1:]:
            if datapoint is None:
                continue
            for variableName, variableArray in datapoint.allVariables.items():
                # to variableName key in superDictionary append variableArray
                superDictionary[variableName].append(variableArray)

        # now that all the data is in the one dict, save it
        for variableName, variableArray in superDictionary.items():
            filename = f"{outputFolderName}/{variableName}.csv"

            # now make the jagged array into a rectangular one by padding with nans
            try:
                if np.ndim(variableArray) == 0:
                    # if it's just a scalar we don't need it saved
                    continue
                if len(variableArray) == 1:
                    # there is literally only one element in the array, so it can't be jagged
                    variableArray = variableArray[0]
            except ValueError:
                # this error happens when the array is jagged
                variableArray = padTillNotJagged(variableArray)

            # headers
            if np.ndim(variableArray) == 1:
                header = f"{variableName} [{unitsDictionary[variableName.lower()]}]"
            elif np.ndim(variableArray) == 2:
                header = f"{variableName} [{unitsDictionary[variableName.lower()]}]; rows index depth, columns index time"
            else:
                raise ValueError(
                    f"Weird dimension of {variableName} array ({np.ndim(variableArray)})"
                )

            np.savetxt(
                filename,
                np.array(variableArray).T,
                header=header,
                delimiter=",",
            )
            L.info(f"Saved data into {filename}")


def createDataFromFolder(foldername: str) -> Data:
    """
    creates a data cube from a folder
    """

    try:
        times = (
            np.loadtxt(f"{foldername}/times.csv", skiprows=1, delimiter=",")
        )
    except FileNotFoundError:
        L.info("No times.csv file found, assuming only one time datapoint")
        times = np.array([0.0])
    toReturn = Data(finalT=times[-1], numberOfTSteps=len(times), startT=times[0])

    loadedVariables = {}
    folder = os.listdir(foldername)
    try:
        folder.remove("times.csv")
    except ValueError:
        # times.csv is not in the folder, so we don't care
        pass
    for file in folder:
        variableName = file[:-4]
        if variableName == "numberOfZSteps":
            continue
        loadedVariables[variableName] = (
            np.loadtxt(f"{foldername}/{variableName}.csv", skiprows=1, delimiter=",")
        ).T

    for i, _ in enumerate(times):
        thisTimesVariables = {}
        for key, value in loadedVariables.items():
            if np.ndim(value) == 0 or np.ndim(value) == 1:
                thisTimesVariables[key] = value
            else:
                try:
                    thisTimesVariables[key] = value[i, :]
                    thisTimesVariables[key] = thisTimesVariables[key][~np.isnan(thisTimesVariables[key])]
                    pass
                except (
                    IndexError
                ):  # TODO wow this is dumb, dont do it this way pls, use numpy
                    break
        else:
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
