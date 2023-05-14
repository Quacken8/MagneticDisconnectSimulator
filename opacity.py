#!/usr/bin/env python3
"""
interpolates opacity from a table generated by MESA, in particular using pyMesa
"""

import constants as c
import numpy as np
from mesa2Py import kappaFromMesa as kapMes
from dataHandling.modelS import loadModelS


def mesaOpacity(
    pressure: np.ndarray, temperature: np.ndarray, massFractions: dict | None = None
) -> np.ndarray:
    """
    returns opacity from mesa
    ---
    inputs are in SI units
    massFractions is a dictionary of the form {"element": massFraction}
    where element is a string like h1, he4, c12, o16, etc.
    see constants.py for a list of all known chemical elements
    """
    if massFractions is None:
        massFractions = c.solarAbundances
    fullOpacityResults = kapMes.getMesaOpacity(pressure, temperature, massFractions)
    try:
        opacity = np.vectorize(lambda result: result.kappa)(fullOpacityResults)
    except AttributeError:
        opacity = np.vectorize(lambda result: result.item().kappa)(fullOpacityResults) # FIXME ughhh
    return opacity


# region modelS opacity
modelS = loadModelS()
modelTs, modelPs, modelKappas = modelS.temperatures, modelS.pressures, modelS.derivedQuantities["kappas"]
del modelS
# given the structure of the data, it's much better for the nearest interpolation algorithm to work on logs and then exp it later
logmodelTs, logmodelPs, logmodelKappas = (
    np.log(modelTs),
    np.log(modelPs),
    np.log(modelKappas),
)
from scipy.interpolate import NearestNDInterpolator

inteprloatedKappas = NearestNDInterpolator(
    list(zip(logmodelTs, logmodelPs)), logmodelKappas
)


def modelSNearestOpacity(
    pressure: np.ndarray, temperature: np.ndarray
) -> np.ndarray:
    """just interpolates using nearest neighbour from the model S kappas"""
    return np.exp(inteprloatedKappas(np.log(temperature), np.log(pressure)))


# endregion


def main():
    """debugging function for this file"""
    from dataHandling.modelS import loadModelS

    modelS = loadModelS()

    modelSRhos = modelS.derivedQuantities["rhos"]
    modelSTs = modelS.temperatures
    modelSKappas = modelS.derivedQuantities["kappas"]

    mesaKappas = mesaOpacity(modelSRhos, modelSTs)

    zs = modelS.zs

    import matplotlib.pyplot as plt

    plt.loglog(zs, modelSKappas, label="modelS")
    plt.loglog(zs, mesaKappas, label="mesa")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
