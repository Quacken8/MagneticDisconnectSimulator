#!/usr/bin/env python3
import numpy as np
from dataStructure import SingleTimeDatapoint
import constants as c


def getInitialConditions(zLengthPower: int, maxDepth: float) -> SingleTimeDatapoint:
    """
    
    """
    maxDepth *= c.Mm
    numberOfZSteps = 2**zLengthPower + 1
    raise NotImplementedError()

def mockupDataSetterUpper(zLengthPower:int = 10) -> SingleTimeDatapoint:
    """
    mockup data setter upper that just makes bunch of weird powers of ten instead of the pressures and other datapoints of length 1+2**zLengthPower
    """
    ones = np.ones(2**zLengthPower + 1)
    maxdepth = 4
    toReturn = SingleTimeDatapoint(temperatures = ones, pressures = ones*10, B_0s = ones*100, F_rads = ones*1000, F_cons = ones*10000, maxDepth=maxdepth, numberOfZStepsPower= zLengthPower)
    return toReturn