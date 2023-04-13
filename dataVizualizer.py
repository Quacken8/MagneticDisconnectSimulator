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
from typing import Iterable, Dict
    
def plotSingleTimeDatapoint(
    datapoint: SingleTimeDatapoint,
    toPlot: Iterable[str],
    axs: Dict[str, plt.Axes] | None = None,
    pltshow: bool = True,
    log: bool=True,
    title: str = "",
    label: str = "",
) -> Dict[str, plt.Axes]:
    """
    Plots a single time datapoint into provided axes (if they are provided) and returns the axes.
    """
    
    title += " "
    zs = datapoint.zs / c.Mm
    if axs is None:
        axs = {}

    variables = dictionaryOfVariables(datapoint)

    for plot in toPlot:
        if plot[-1] != "s":
            plot += "s"
        plot = plot.lower()
        if not plot in axs.keys():
            fig, ax = plt.subplots()
            axs[plot] = ax
        else:
            ax = axs[plot]
        dataToPlot = variables[plot]
        if log:
            ax.loglog(zs, dataToPlot, label=plot + " " + label)
        else:
            ax.plot(zs, dataToPlot, label=plot + " " + label)
        ax.set_xlabel(f"z [{unitsDictionary['zs']}]")
        ax.set_ylabel(f"{plot} [{unitsDictionary[plot]}]")
        ax.set_title(f"{title}{plot} vs depth")
        ax.legend()

    if pltshow:
        plt.show()
    return axs

def main():
    pass


if __name__ == "__main__":
    main()
