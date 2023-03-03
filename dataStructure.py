#!/usr/bin/env python3
import numpy as np
import constants as c
import os

class SingleTimeDatapoint():
    """
    class that saves an instanteanous state of the simulation. Similar to Data class, but only for one t
    all datapoints are expected in SI

    NumberOfZStepsPower: the number of steps in z direction will be 2^k + 1 where k is the NumberOfZStepsPower
    """

    def __init__(self, temperatures, pressures, B_0s, F_rads, F_cons, maxDepth: float, numberOfZStepsPower: int,) -> None:

        self.numberOfZSteps = 2**numberOfZStepsPower + 1
        self.maxDepth = maxDepth
        self.zs = np.linspace(start=0, stop=self.maxDepth,num=self.numberOfZSteps)
        self.dz = self.zs[1]

        self.temperatures = temperatures
        self.pressures = pressures
        self.B_0s = B_0s
        self.F_rads = F_rads
        self.F_cons = F_cons

class Data():
    """
    Class for handling all data. Is initialized with the whole size for speed reasons, might be a problem later lol
    ----
    VARIABLES
    ----
    Ts, 2D np array of doubles: temperatures in Kelvin. First index stands for z index (i.e. depth), second for t index (i.e. time)
    pressures, 2D np array of doubles: temperatures in TBD. First index stands for z index (i.e. depth), second for t index (i.e. time)
    """

    def __init__(self, finalT : float, numberOfTSteps : int):

        self.finalT = finalT
        self.times = np.linspace(start = 0, stop = self.finalT, num = numberOfTSteps)
        self.dt = self.times[1]
        self.numberOfTSteps = self.times.size

        self.values = np.empty(self.numberOfTSteps, dtype=SingleTimeDatapoint)
        self.occupiedElements = 0


    def addDatapointAtIndex(self, datapoint : SingleTimeDatapoint, index : int) -> None:
        """
        adds SingleTimeDatapoint to the data at index
        """
        if index > self.occupiedElements: raise ValueError("You're trying to add a datapoint to an index while a previous one isn't filled")
        self.values[index] = datapoint
        if index == self.occupiedElements:
            self.occupiedElements += 1
    
    def appendDatapoint(self, datapoint : SingleTimeDatapoint) -> None:
        """
        appends datapoint at the end of the data cube
        """

        self.addDatapointAtIndex(datapoint=datapoint, index=self.occupiedElements)


    def saveToFolder(self, outputFolderName : str, rewriteFolder = False) -> None:
        """
        creates folder which contains csv files of the simulation
        """
        if rewriteFolder: os.system(f"rm -r {outputFolderName}")
        os.mkdir(outputFolderName)

        temperatureFilename = f"{outputFolderName}/Temperature.csv"
        pressureFilename = f"{outputFolderName}/Pressure.csv"
        B_0Filename = f"{outputFolderName}/B_0.csv"
        timeFilename = f"{outputFolderName}/Time.csv"
        depthFilename = f"{outputFolderName}/Depth.csv"
        F_radFilename = f"{outputFolderName}/F_rad.csv"
        F_conFilename = f"{outputFolderName}/F_con.csv"

        np.savetxt(timeFilename, self.times/c.hour, header="time [h]", delimiter=",")

        tosaveZs = self.values[0].zs
        np.savetxt(depthFilename, tosaveZs/c.Mm, header="depth [Mm]", delimiter=",")

        numberOfZSteps = len(tosaveZs)
        toSaveTemperatures = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSavePressures = np.zeros(
            (self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSaveB_0s = np.zeros(
            (self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSaveF_rads = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSaveF_cons = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)

        for i, datapoint in enumerate(self.values):
            toSaveTemperatures[i] = datapoint.temperatures
            toSavePressures[i] = datapoint.pressures
            toSaveB_0s[i] = datapoint.B_0s
            toSaveF_rads[i] = datapoint.F_rads
            toSaveF_cons[i] = datapoint.F_cons

        np.savetxt(temperatureFilename, toSaveTemperatures.T, header="temperature [K], rows index depth, columns index time", delimiter=",")
        np.savetxt(pressureFilename, toSavePressures.T, header="pressure [Pa], rows index depth, columns index time", delimiter=",")
        np.savetxt(B_0Filename, toSaveB_0s.T/c.Gauss, header="B_0 [Gauss], rows index depth, columns index time", delimiter=",")
        np.savetxt(F_conFilename, toSaveF_cons.T, header="F_con [???], rows index depth, columns index time", delimiter=",")
        np.savetxt(F_radFilename, toSaveF_rads.T,header="F_rad [???], rows index depth, columns index time", delimiter=",")
        
        
