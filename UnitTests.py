#!/usr/bin/env python3
from dataStructure import Data, SingleTimeDatapoint
import numpy as np
from initialConditionsSetterUpper import mockupDataSetterUpper
from constants import *
import os


def TestDataStractureSave() -> None:
    # create mockup set of datapoints

    datapoint = mockupDataSetterUpper(zLengthPower= 4) # expected 2**4+1 = 17 datapoints

    data = Data(finalT=3*Chour, numberOfTSteps=4)

    data.appendDatapoint(datapoint)
    data.appendDatapoint(datapoint)
    data.appendDatapoint(datapoint)
    data.appendDatapoint(datapoint)

    foldername = "SaveTest"
    data.saveToFolder(foldername)

    loadedbs = np.loadtxt(foldername+"/B_0.csv", delimiter = ",", skiprows = 1)
    loadedTemperatures = np.loadtxt(foldername+"/Temperature.csv", delimiter=",", skiprows=1)
    loadedPressures = np.loadtxt(foldername+"/Pressure.csv", delimiter = ",", skiprows = 1)
    loadedF_rads = np.loadtxt(foldername+"/F_rad.csv", delimiter = ",", skiprows = 1)
    loadedF_cons = np.loadtxt(foldername+"/F_con.csv", delimiter=",", skiprows=1)
    loadedTimes = np.loadtxt(foldername+"/Time.csv",
                             delimiter=",", skiprows=1)
    loadedDepths = np.loadtxt(foldername+"/Depth.csv", delimiter=",", skiprows=1)

    os.system(f"rm -r {foldername}")
    

    ### expected values
    bs = np.array([[1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.],
                   [1000000.,1000000., 1000000., 1000000.]])
    pressures = np.array([[10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.],
                          [10., 10., 10., 10.]])
    temperatures = np.array([[1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.],
                             [1., 1., 1., 1.]])
    F_cons = np.array([[10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.],
                       [10000.,10000., 10000., 10000.]])
    F_rads = np.array([[1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.],
                       [1000., 1000., 1000., 1000.]])
    depths = np.array([0.00e+00, 2.50e-07, 5.00e-07, 7.50e-07, 1.00e-06, 1.25e-06, 1.50e-06, 1.75e-06,
                      2.00e-06, 2.25e-06, 2.50e-06, 2.75e-06, 3.00e-06, 3.25e-06, 3.50e-06, 3.75e-06, 4.00e-06])
    times = np.array([0., 1., 2., 3.])

    assert np.array_equal(loadedTemperatures, temperatures)
    assert np.array_equal(loadedbs, bs)
    assert np.array_equal(loadedDepths, depths)
    assert np.array_equal(loadedF_cons, F_cons)
    assert np.array_equal(loadedF_rads, F_rads)
    assert np.array_equal(loadedPressures, pressures)
    assert np.array_equal(loadedTimes, times)


def main():
    TestDataStractureSave()

    print("Tests passed :)")

if __name__ == "__main__":
    main()
