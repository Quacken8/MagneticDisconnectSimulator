#!/usr/bin/env python3
"""
interpolates opacity from a table generated by MESA, in particular using pyMesa
"""

import constants as c
import numpy as np
from mesa2Py import kappaFromMesa as kapMes


def mesaOpacity(
    density: float, temperature: float, massFractions: dict | None = None
) -> float:
    """
    returns opacity from mesa
    ---
    inputs are in SI units
    massFractions is a dictionary of the form {"element": massFraction}
    where element is a string like h1, he4, c12, o16, etc.
    see constants.py for a list of all known chemical elements # FIXME this is a cool idea, put the solar compsition in constants.py
    """

    fullOpacityResult = kapMes.getMESAOpacity(density, temperature, massFractions)
    opacity = fullOpacityResult.kappa
    return opacity


# region modelS opacity
modelSPath = "externalData/model_S_new.dat"
modelTs, modelPs, modelKappas = np.loadtxt(modelSPath, skiprows=1, usecols=(1, 2, 4)).T
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
    temperature: float | np.ndarray, pressure: float | np.ndarray
) -> float | np.ndarray:
    """just interpolates using nearest neighbour from the model S kappas"""
    return np.exp(inteprloatedKappas(np.log(temperature), np.log(pressure)))


# endregion


def main():
    """debugging function for this file"""
    from initialConditionsSetterUpper import loadModelS

    modelS = loadModelS()

    modelSRhos = modelS.derivedQuantities["rhos"]
    modelSTs = modelS.temperatures
    modelSKappas = modelS.derivedQuantities["kappas"]

    mesaKappas = mesaOpacity(modelSTs, modelSRhos)

    zs = modelS.zs

    import matplotlib.pyplot as plt

    plt.loglog(zs, modelSKappas, label="modelS")
    plt.loglog(zs, mesaKappas, label="mesa")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
