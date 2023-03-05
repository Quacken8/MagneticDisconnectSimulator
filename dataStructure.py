#!/usr/bin/env python3
import numpy as np
import constants as c
import os

def dictionaryOfVariables(A: object) -> dict:
    """makes dictionary of name value pairs of a class"""
    dic = {key: value for key, value in A.__dict__.items(
    ) if not key.startswith('__') and not callable(key)}

    return dic

"""
this dictionary is used to put units next to values that come up in the simulatuin. It is expected that the units went through UX parse, i.e. units as hours and megameters are used for readibility
"""
unitsDictionary = {
    "time" : "h",
    "depth": "Mm",
    "numberOfZSteps": "1",
    "maxDepth": "Mm",
    "temperatures": "K",
    "pressures": "Pa",
    "rhos": "kg/m^3",
    "B_0s": "T",
    "F_rads": "W/m^2",
    "F_cons": "W/m^2",
    "entropies": "J/K",
    "nablaAds": "log(Pa)/m", # TBD this aint correct lol
    "cps": "J/(kg.K)",
    "cvs": "J/(kg.K)",
    "deltas": "" # the fuck is delta TBD
}


class SingleTimeDatapoint():
    """
    class that saves an instanteanous state of the simulation. Similar to Data class, but only for one t
    all datapoints are expected in SI

    NumberOfZStepsPower: the number of steps in z direction will be 2^k + 1 where k is the NumberOfZStepsPower
    """

    def __init__(self, temperatures: np.ndarray, pressures: np.ndarray, B_0s: np.ndarray, F_rads: np.ndarray, F_cons: np.ndarray, zs: np.ndarray, rhos: np.ndarray, entropies: np.ndarray, nablaAds: np.ndarray, cps: np.ndarray, cvs: np.ndarray, deltas: np.ndarray) -> None:

        self.zs = zs
        self.numberOfZSteps = len(zs)
        self.maxDepth = zs[-1]
        self.temperatures = temperatures
        self.pressures = pressures
        self.rhos = rhos
        self.B_0s = B_0s
        self.F_rads = F_rads
        self.F_cons = F_cons
        self.entropies = entropies
        self.nablaAds = nablaAds
        self.cps = cps
        self.cvs = cvs
        self.deltas = deltas


class Data():
    """
    Class for handling all data. Is initialized with the whole size for speed reasons, might be a problem later lol
    ----
    VARIABLES
    ----
    Ts, 2D np array of doubles: temperatures in Kelvin. First index stands for z index (i.e. depth), second for t index (i.e. time)
    pressures, 2D np array of doubles: temperatures in TBD. First index stands for z index (i.e. depth), second for t index (i.e. time)
    """

    def __init__(self, finalT : float, numberOfTSteps : int, startT:float = 0):
        if startT == finalT: raise ValueError("your start T equals end T")
        self.finalT = finalT
        self.times = np.linspace(start=startT, stop=self.finalT, num=numberOfTSteps)
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
        densityFilename = f"{outputFolderName}/Density.csv"
        B_0Filename = f"{outputFolderName}/B_0.csv"
        timeFilename = f"{outputFolderName}/Time.csv"
        depthFilename = f"{outputFolderName}/Depth.csv"
        F_radFilename = f"{outputFolderName}/F_rad.csv"
        F_conFilename = f"{outputFolderName}/F_con.csv"
        entropyFilename = f"{outputFolderName}/Entropy.csv"
        nablaAdFilename = f"{outputFolderName}/nablaAd.csv"
        cpFilename = f"{outputFolderName}/cp.csv"
        cvFilename = f"{outputFolderName}/cv.csv"
        deltaFilename = f"{outputFolderName}/Delta.csv"

        np.savetxt(timeFilename, self.times/c.hour, header=f"time [{unitsDictionary['time']}]", delimiter=",")

        tosaveZs = self.values[0].zs
        np.savetxt(depthFilename, tosaveZs/c.Mm, header=f"depth [{unitsDictionary['depth']}]", delimiter=",")

        numberOfZSteps = len(tosaveZs)
        toSaveTemperatures = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSavePressures = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSaveDensities = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSaveB_0s = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSaveF_rads = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSaveF_cons = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSaveEntropies = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSaveNablaAds = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSavecps = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSavecvs = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        toSavedeltas = np.zeros((self.numberOfTSteps, numberOfZSteps), dtype=float)
        

        for i, datapoint in enumerate(self.values):
            toSaveTemperatures[i] = datapoint.temperatures
            toSavePressures[i] = datapoint.pressures
            toSaveDensities[i] = datapoint.rhos
            toSaveB_0s[i] = datapoint.B_0s
            toSaveF_rads[i] = datapoint.F_rads
            toSaveF_cons[i] = datapoint.F_cons
            toSaveEntropies[i] = datapoint.entropies
            toSaveNablaAds[i] = datapoint.nablaAds
            toSavecps[i] = datapoint.cps
            toSavecvs[i] = datapoint.cvs
            toSavedeltas[i] = datapoint.deltas
            

        np.savetxt(temperatureFilename, toSaveTemperatures.T, header=f"temperature [{unitsDictionary['temperatures']}], rows index depth, columns index time", delimiter=",")
        np.savetxt(pressureFilename, toSavePressures.T, header=f"pressure [{unitsDictionary['pressures']}], rows index depth, columns index time", delimiter=",")
        np.savetxt(densityFilename, toSaveDensities.T, header=f"density [{unitsDictionary['rhos']}], rows index depth, columns index time", delimiter=",")
        np.savetxt(B_0Filename, toSaveB_0s.T/c.Gauss, header=f"B_0 [{unitsDictionary['B_0s']}], rows index depth, columns index time", delimiter=",")
        np.savetxt(F_conFilename, toSaveF_cons.T, header=f"F_con [{unitsDictionary['F_cons']}], rows index depth, columns index time", delimiter=",")
        np.savetxt(F_radFilename, toSaveF_rads.T, header=f"F_rad [{unitsDictionary['F_rads']}], rows index depth, columns index time", delimiter=",")
        np.savetxt(entropyFilename, toSaveEntropies.T, header=f"S [{unitsDictionary['entropies']}], rows index depth, columns index time", delimiter=",")
        np.savetxt(nablaAdFilename, toSaveNablaAds.T, header=f"nablaAd [{unitsDictionary['nablaAds']}], rows index depth, columns index time", delimiter=",")
        np.savetxt(cpFilename, toSavecps.T, header=f"cp [{unitsDictionary['cps']}], rows index depth, columns index time", delimiter=",")
        np.savetxt(cvFilename, toSavecvs.T, header=f"cv [{unitsDictionary['cvs']}], rows index depth, columns index time", delimiter=",")
        np.savetxt(deltaFilename, toSavedeltas.T, header=f"Delta [{unitsDictionary['deltas']}], rows index depth, columns index time", delimiter=",")
        

def createDataFromFolder(foldername: str) -> Data:
    """
    creates a data cube from a folder
    """

    times = np.loadtxt(f"{foldername}/Time.csv", skiprows=1, delimiter = ",")*c.hour

    toReturn = Data(finalT=times[-1], numberOfTSteps=len(times), startT=times[0])

    zs = np.loadtxt(f"{foldername}/Depth.csv")*c.Mm

    Temperatures = np.loadtxt(f"{foldername}/Temperature.csv", skiprows=1, delimiter=",")
    Pressures = np.loadtxt(f"{foldername}/Pressure.csv", skiprows=1, delimiter=",")
    Densities = np.loadtxt(f"{foldername}/Density.csv", skiprows=1, delimiter=",")
    B_0s = np.loadtxt(f"{foldername}/B_0.csv", skiprows=1, delimiter=",") * c.Gauss
    F_cons = np.loadtxt(f"{foldername}/F_con.csv", skiprows=1, delimiter=",")
    F_rads = np.loadtxt(f"{foldername}/F_rad.csv", skiprows=1, delimiter=",")
    Entropies = np.loadtxt(f"{foldername}/Entropy.csv", skiprows=1, delimiter=",")
    NablaAds = np.loadtxt(f"{foldername}/nablaAd.csv", skiprows=1, delimiter=",")
    cps = np.loadtxt(f"{foldername}/cp.csv", skiprows=1, delimiter=",")
    cvs = np.loadtxt(f"{foldername}/cv.csv", skiprows=1, delimiter=",")
    deltas = np.loadtxt(f"{foldername}/Delta.csv", skiprows=1, delimiter=",")


    for i, _ in enumerate(times):
        newDatapoint = SingleTimeDatapoint(
            zs = zs[:],
            temperatures=Temperatures[:, i],
            pressures=Pressures[:, i],
            rhos=Densities[:, i],
            B_0s=B_0s[:, i],
            F_cons=F_cons[:, i],
            F_rads=F_rads[:, i],
            entropies=Entropies[:, i],
            nablaAds=NablaAds[:, i],
            cps=cps[:, i],
            cvs=cvs[:, i],
            deltas=deltas[:, i]
        )
        toReturn.appendDatapoint(newDatapoint)

    return toReturn