#!/usr/bin/env python3
from dataStructure import Data, SingleTimeDatapoint
import numpy as np
from initialConditionsSetterUpper import mockupDataSetterUpper

datapoint = mockupDataSetterUpper(4)

data = Data(4, 1, 6, 2)

data.addDatapointAtIndex(datapoint, 0)