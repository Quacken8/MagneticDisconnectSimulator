import numpy as np
from dataStructure import SingleTimeDatapoint

def mockupDataSetterUpper(zLengthPower:int = 10) -> SingleTimeDatapoint:
    ones = np.ones(2**zLengthPower + 1)
    maxdepth = 4
    toReturn = SingleTimeDatapoint(temperatures = ones, pressures = ones*10, B_0s = ones*100, F_rads = ones*1000, F_cons = ones*10000, maxDepth=maxdepth, numberOfZStepsPower= zLengthPower)
    return toReturn