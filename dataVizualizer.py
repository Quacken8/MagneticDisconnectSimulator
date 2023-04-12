#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from dataStructure import (
    Data,
    SingleTimeDatapoint,
    dictionaryOfVariables,
    unitsDictionary,
)
import numpy.typing as npt
import constants as c
from typing import Iterable


def plotSingleTimeDatapoint(
    datapoint: SingleTimeDatapoint,
    toPlot: Iterable[str],
    pltshow: bool = True,
    log=False,
) -> None:
    zs = datapoint.zs / c.Mm

    variables = dictionaryOfVariables(datapoint)

    for plot in toPlot:
        if plot[-1] != "s":
            plot += "s"
        plot = plot.lower()
        fig, ax = plt.subplots()
        dataToPlot = variables[plot]
        if log:
            ax.loglog(zs, dataToPlot, label=plot)
        else:
            ax.plot(zs, dataToPlot, label=plot)
        ax.set_xlabel(f"z [{unitsDictionary['depth']}]")
        ax.set_ylabel(f"{plot} [{unitsDictionary[plot]}]")
        plt.legend()

    if pltshow:
        plt.show()


def main():
    """
    test program
    """
    from initialConditionsSetterUpper import modelSLoader

    datapoint = modelSLoader(500)

    toPlot = ["temperatures", "pressures", "rhos"]

    plotSingleTimeDatapoint(datapoint, toPlot, log=True)


if __name__ == "__main__":
    main()
