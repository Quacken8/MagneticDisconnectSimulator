import numpy as np
from constants import *
import os

class SingleTimeDatapoint():
    """
    class that saves an instanteanous state of the simulation. Similar to Data class, but only for one T
    all datapoints are expected in SI
    """

    def __init__(self, temperatures, pressures, B_0s) -> None:
        self.temperatures = temperatures
        self.pressures = pressures
        self.B_0s = B_0s

class Data():
    """
    Class for handling all data. Is initialized with the whole size for speed reasons, might be a problem later lol
    ----
    VARIABLES
    ----
    Ts, 2D np array of doubles: temperatures in Kelvin. First index stands for z index (i.e. depth), second for t index (i.e. time)
    pressures, 2D np array of doubles: temperatures in TBD. First index stands for z index (i.e. depth), second for t index (i.e. time)
    """

    def __init__(self, maxDepth : float, dz : float, finalT : float, dt : float):

        self.finalT = finalT
        self.dt = dt
        self.times = np.arange(start = 0, stop = self.finalT, step = self.dt)
        self.numberOfTSteps = self.times.size
        
        self.maxDepth = maxDepth
        self.dz = dz
        self.zs = np.arange(start=0, stop=self.maxDepth, step=self.dz)
        self.numberOfZSteps = self.zs.size

        self.values = np.empty(self.numberOfTSteps, dtype=SingleTimeDatapoint)

        self.pressures = np.zeros((self.numberOfZSteps, self.numberOfTSteps))
        self.temperatures = np.zeros((self.numberOfZSteps, self.numberOfTSteps))
        self.B_0s = np.zeros((self.numberOfZSteps, self.numberOfTSteps))

    def addDatapointAtIndex(self, datapoint : SingleTimeDatapoint, index : int):
        """
        adds SingleTimeDatapoint to the data at index
        """
        self.values[index] = datapoint

    def saveToFolder(self, outputFolderName):
        """
        creates folder which contains csv files of the simulation
        """
        os.mkdir(outputFolderName)

        temperatureFilename = f"{outputFolderName}/Temperature.csv"
        pressureFilename = f"{outputFolderName}/Pressure.csv"
        B_0Filename = f"{outputFolderName}/B_0.csv"
        timeFilename = f"{outputFolderName}/Time.csv"
        depthFilename = f"{outputFolderName}/Depth.csv"

        np.savetxt(timeFilename, self.times/Chour, header="time [h]", delimiter=",")
        np.savetxt(depthFilename, self.zs/CMm, header="depth [Mm]", delimiter=",")

        np.savetxt(temperatureFilename, self.temperatures, header="temperature [K], rows index depth, columns index time", delimiter=",")
        np.savetxt(pressureFilename, self.pressures, header="pressure [TBDDD], rows index depth, columns index time", delimiter=",")
        np.savetxt(B_0Filename, self.B_0s/CGauss, header="B_0 [Gauss], rows index depth, columns index time", delimiter=",")
        

