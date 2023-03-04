#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from dataStructure import Data, SingleTimeDatapoint
import numpy.typing as npt
import constants as c

def plotSingleTimeDatapoint(datapoint: SingleTimeDatapoint, toPlot: npt.ArrayLike, pltshow: bool = True) -> None:
    raise NotImplementedError()
    zs = datapoint.zs/c.Mm

    for plot in toPlot:
        fig, ax = plt.subplots()
        ax.plot(zs, datapoint.plot, legend = plot)
        ax.set_xlabel("z [Mm]")
        ax.set_ylabel(f"{plot} [{plot.units}]")

    plt.legend()
    if pltshow: plt.show()