from dataStructure import Data, SingleTimeDatapoint
import numpy as np

maxDepth = 10
dz = 1

finalT = 5
dt = 1


data = Data(maxDepth=maxDepth, dz = dz, finalT=finalT, dt=dt)

emptyArray = np.ones(10)

for i in range(5):
    datapoint = SingleTimeDatapoint(emptyArray, emptyArray*10, emptyArray*100)

data.saveToFolder("dataUnitTest")