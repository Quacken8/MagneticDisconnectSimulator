#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from dataStructure import Data, SingleTimeDatapoint, dictionaryOfVariables, unitsDictionary
import numpy.typing as npt
import constants as c
        

def plotSingleTimeDatapoint(datapoint: SingleTimeDatapoint, toPlot: npt.ArrayLike, pltshow: bool = True) -> None:

    zs = datapoint.zs/c.Mm
    variables = dictionaryOfVariables(datapoint)

    for plot in toPlot:
        fig, ax = plt.subplots()
        dataToPlot = variables[plot]
        ax.plot(zs, dataToPlot, legend=plot)
        ax.set_xlabel(f"z [{unitsDictionary['depth']}]")
        ax.set_ylabel(f"{plot} [{unitsDictionary[plot]}]")

    plt.legend()
    if pltshow: plt.show()