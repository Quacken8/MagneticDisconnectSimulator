import numpy as np
from dataStructure import SingleTimeDatapoint

def mockupDataSetterUpper(zLength:int = 10) -> SingleTimeDatapoint:
    ones = np.ones(zLength)
    toReturn = SingleTimeDatapoint(ones, ones*10, ones*100)
    return toReturn