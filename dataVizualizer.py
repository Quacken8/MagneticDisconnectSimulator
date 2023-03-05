#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from dataStructure import Data, SingleTimeDatapoint, dictionaryOfVariables, unitsDictionary
import numpy.typing as npt
import constants as c
from typing import Iterable
        

def plotSingleTimeDatapoint(datapoint: SingleTimeDatapoint, toPlot: Iterable[str], pltshow: bool = True) -> None:

    zs = datapoint.zs/c.Mm

    variables = dictionaryOfVariables(datapoint)

    for plot in toPlot:
        if plot[-1] != 's': plot += 's'
        plot = plot.lower()
        fig, ax = plt.subplots()
        dataToPlot = variables[plot]
        ax.plot(zs, dataToPlot, label=plot)
        ax.set_xlabel(f"z [{unitsDictionary['depth']}]")
        ax.set_ylabel(f"{plot} [{unitsDictionary[plot]}]")

    plt.legend()
    if pltshow: plt.show()

def main():
    """
    test program
    """
    from initialConditionsSetterUpper import mockupDataSetterUpper
    datapoint = mockupDataSetterUpper(5)

    toPlot = ["temperatures", "pressures"]

    plotSingleTimeDatapoint(datapoint, toPlot)

if __name__ == "__main__":
    main()