#!/usr/bin/env python3
import numpy as np
from dataStructure import SingleTimeDatapoint, Data
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
    pathToModelS = "model_S_raw.dat"
    zeros = np.zeros(length)  # FIXME used for all unknown cols from model S
    zs = np.loadtxt(pathToModelS, skiprows=1, usecols=0)
    lengthOfSModel = len(zs)
    skippingIndex = lengthOfSModel//length

    zs = zs[::skippingIndex][:length]
    Ts = np.loadtxt(pathToModelS, skiprows=1,
                    usecols=1)[::skippingIndex][:length]
    Ps = np.loadtxt(pathToModelS, skiprows=1,
                    usecols=2)[::skippingIndex][:length]
    rhos = np.loadtxt(pathToModelS, skiprows=1, usecols=3)[
        ::skippingIndex][:length]
    datapoint = SingleTimeDatapoint(temperatures=Ts, pressures=Ps, zs=zs, rhos=rhos, B_0s=zeros,
                                    F_rads=zeros, F_cons=zeros, entropies=zeros, nablaAds=zeros, cps=zeros, cvs=zeros, deltas=zeros)

    return datapoint
