#!/usr/bin/env python3
from dataStructure import (
    Data,
    SingleTimeDatapoint,
    createDataFromFolder,
    dictionaryOfVariables,
)
import numpy as np
from initialConditionsSetterUpper import mockupDataSetterUpper, loadModelS
from stateEquationsPT import IdealGas
from sunSolvers.calmSun import getCalmSunDatapoint
from dataVizualizer import plotSingleTimeDatapoint
from sunSolvers.pressureSolvers import (
    integrateHydrostaticEquilibriumAndTemperatureGradient,
)
import time

# from solvers import oldYSolver
from initialConditionsSetterUpper import loadModelS
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import constants as c
import os
from matplotlib import pyplot as plt
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)
L = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

pathToModelS = "externalData/model_S_new.dat"
modelS = loadModelS()


def testCalmSunVsModelS():
    modelSPressures = modelS.pressures
    modelSTemperatures = modelS.temperatures
    modelSZs = modelS.zs

    from scipy.interpolate import interp1d

    surfaceZ = 0 * c.Mm
    surfaceT = interp1d(modelSZs, modelSTemperatures)(surfaceZ)
    surfaceP = interp1d(modelSZs, modelSPressures)(surfaceZ)

    dlnp = 1e-2
    maxDepth = 160 * c.Mm
    convectiveAlpha = 0.3  # schussler rempel

    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity, modelSNearestOpacity

    import time

    now = time.time()
    calmSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=dlnp,
        lnSurfacePressure=np.log(surfaceP),
        surfaceTemperature=surfaceT,
        surfaceZ=surfaceZ,
        maxDepth=maxDepth,
        opacityFunction=mesaOpacity,
        guessTheZRange=True,
    )
    calmSunWithModelSOpacity = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=dlnp,
        lnSurfacePressure=np.log(surfaceP),
        surfaceTemperature=surfaceT,
        surfaceZ=surfaceZ,
        maxDepth=maxDepth,
        opacityFunction=modelSNearestOpacity,
        guessTheZRange=True,
    )
    print("time elapsed: ", time.time() - now)
    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(
        calmSun, toPlot, pltshow=False, label="Calm Sun with MESA kappa", log=False
    )
    axs = plotSingleTimeDatapoint(
        calmSunWithModelSOpacity,
        toPlot,
        axs=axs,
        pltshow=False,
        label="Calm Sun with model S kappa",
        log=False,
    )
    plotSingleTimeDatapoint(modelS, toPlot, axs=axs, label="Model S", log=True)
    plt.legend()
    plt.show()


def testIDLOutput():
    modelSPressures = modelS.pressures
    modelSTemperatures = modelS.temperatures
    modelSZs = modelS.zs
    from opacity import mesaOpacity
    from stateEquationsPT import MESAEOS
    from gravity import g, massBelowZ

    zs, p, T, rho, kapa, nablaRad, H = np.loadtxt(
        "debuggingReferenceFromSvanda/idlOutput.dat", unpack=True, skiprows=1
    )
    myRho = MESAEOS.density(modelSTemperatures, modelSPressures)
    myKapa = mesaOpacity(modelSPressures, modelSTemperatures)
    myGs = g(modelSZs)
    myM_r = massBelowZ(modelSZs)
    myH = MESAEOS.pressureScaleHeight(modelSTemperatures, modelSPressures, myGs)
    myNablaRad = MESAEOS.radiativeLogGradient(
        modelSTemperatures, modelSPressures, myM_r, myKapa
    )

    svandaGs = g(zs)
    svandaM_r = massBelowZ(zs)
    myHFromSvanda = MESAEOS.pressureScaleHeight(T, p, svandaGs)
    myKappaFromSvanda = mesaOpacity(p, T)
    myNablaRadFromSvanda = MESAEOS.radiativeLogGradient(
        T, p, svandaM_r, myKappaFromSvanda
    )

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].loglog(modelSZs, myRho, label="MESA rho (z modelu S)")
    axs[0, 0].loglog(zs, rho, label="IDL rho")
    axs[0, 0].set_title("rho")
    axs[0, 0].set_xlim(5e5, 20e6)
    axs[0, 0].legend()

    axs[0, 1].loglog(modelSZs, modelSPressures, label="model S P")
    axs[0, 1].loglog(zs, p, label="IDL P")
    axs[0, 1].set_title("P")
    axs[0, 1].set_xlim(5e5, 20e6)
    axs[0, 1].legend()

    axs[0, 2].loglog(modelSZs, modelSTemperatures, label="model S T")
    axs[0, 2].loglog(zs, T, label="IDL T")
    axs[0, 2].set_title("T")
    axs[0, 2].set_xlim(5e5, 20e6)
    axs[0, 2].legend()

    axs[1, 0].loglog(modelSZs, myKapa, label="MESA kappa")
    axs[1, 0].loglog(zs, kapa, label="IDL kapa")
    axs[1, 0].set_title("kapa")
    axs[1, 0].set_xlim(5e5, 20e6)
    axs[1, 0].legend()

    axs[1, 1].loglog(modelSZs, myNablaRad, label="MESA ∇rad z model S kappa")
    axs[1, 1].loglog(zs, nablaRad, label="IDL ∇rad")
    axs[1, 1].set_title("nablaRad")
    axs[1, 1].set_xlim(5e5, 20e6)
    axs[1, 1].legend()

    axs[1, 2].loglog(modelSZs, myH, label="MESA H z modelz S")
    axs[1, 2].loglog(zs, H, label="IDL H")
    axs[1, 2].set_title("H")
    axs[1, 2].set_xlim(5e5, 20e6)
    axs[1, 2].legend()

    plt.show()


def testNewIDLOutput():
    zsS, pressureS, temperatureS, kappaS = (
        modelS.zs,
        modelS.pressures,
        modelS.temperatures,
        modelS.derivedQuantities["kappas"],
    )
    idlZs, idlP, idlT, idlKappa = np.loadtxt(
        "debuggingReferenceFromSvanda/anotherIdlOutput.dat", unpack=True, skiprows=1
    )

    from opacity import modelSNearestOpacity, mesaOpacity

    mesaFromIDL = mesaOpacity(idlP, idlT)
    mesaFromS = mesaOpacity(pressureS, temperatureS)
    myKappaS = modelSNearestOpacity(pressureS, temperatureS)
    myKappaIDL = modelSNearestOpacity(idlP, idlT)

    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].loglog(zsS, mesaFromS, label="model S kappa from mesa", linewidth=10)
    axs[0].loglog(zsS, kappaS, label="model S kappa", linewidth=5)
    axs[0].loglog(zsS, myKappaS, label="model S kappa from my interpolation")
    axs[0].loglog(idlZs, mesaFromIDL, label="IDL kappa from mesa", linewidth=10)
    axs[0].loglog(idlZs, idlKappa, label="IDL kapa", linewidth=5)
    axs[0].loglog(idlZs, myKappaIDL, label="IDL kappa from my interpolation")
    axs[0].set_title("kapa")
    axs[0].legend()

    axs[1].loglog(zsS, pressureS, label="modelS")
    axs[1].loglog(idlZs, idlP, label="IDL")
    axs[1].set_title("P")
    axs[1].legend()

    axs[2].loglog(zsS, temperatureS, label="modelS")
    axs[2].loglog(idlZs, idlT, label="IDL")
    axs[2].set_title("T")
    axs[2].legend()

    plt.show()


def testCalmSunVsIDLS():
    idlZs, idlPs, idlTs = np.loadtxt(
        "debuggingReferenceFromSvanda/anotherIdlOutput.dat",
        unpack=True,
        skiprows=1,
        usecols=(0, 1, 2),
    )

    modelSPressures = idlPs
    modelSTemperatures = idlTs
    modelSZs = idlZs

    idlDatapoint = SingleTimeDatapoint(idlTs, idlPs, idlZs)

    from scipy.interpolate import interp1d

    surfaceZ = 0 * c.Mm
    surfaceT = interp1d(modelSZs, modelSTemperatures)(surfaceZ)
    surfaceP = interp1d(modelSZs, modelSPressures)(surfaceZ)

    dlnp = 1e-3
    maxDepth = 160 * c.Mm

    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity, modelSNearestOpacity

    import time

    now = time.time()
    calmSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=dlnp,
        lnSurfacePressure=np.log(surfaceP),
        surfaceTemperature=surfaceT,
        surfaceZ=surfaceZ,
        maxDepth=maxDepth,
        opacityFunction=modelSNearestOpacity,
        guessTheZRange=True,
    )
    print("time elapsed: ", time.time() - now)
    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(
        calmSun, toPlot, pltshow=False, label="Calm Sun with MESA kappa", log=False
    )
    # axs = plotSingleTimeDatapoint(calmSunWithModelSOpacity, toPlot, axs = axs, pltshow=False, label="Calm Sun with model S kappa", log = False)
    plotSingleTimeDatapoint(idlDatapoint, toPlot, axs=axs, label="Model S", log=True)
    plt.legend()
    plt.show()


def testCalmSunBottomUp():
    modelSPressures = modelS.pressures
    modelSTemperatures = modelS.temperatures
    modelSZs = modelS.zs

    from scipy.interpolate import interp1d

    surfaceZ = 160 * c.Mm
    surfaceT = interp1d(modelSZs, modelSTemperatures)(surfaceZ)
    surfaceP = interp1d(modelSZs, modelSPressures)(surfaceZ)

    dlnp = -1e-3
    maxDepth = 0.1 * c.Mm

    from stateEquationsPT import MESAEOS
    from opacity import modelSNearestOpacity

    import time

    now = time.time()
    calmSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=dlnp,
        lnSurfacePressure=np.log(surfaceP),
        surfaceTemperature=surfaceT,
        surfaceZ=surfaceZ,
        maxDepth=maxDepth,
        opacityFunction=modelSNearestOpacity,
        guessTheZRange=True,
    )
    print("time elapsed: ", time.time() - now)
    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(
        calmSun, toPlot, pltshow=False, label="Calm Sun with MESA kappa", log=False
    )
    plotSingleTimeDatapoint(modelS, toPlot, axs=axs, label="Model S", log=True)
    plt.legend()
    plt.show()


def testFiniteDifferences():
    from sunSolvers.temperatureSolvers import secondCentralDifferencesMatrix

    N = 100
    xs = np.sort(np.random.random(N)) * 2 * np.pi

    secondDif = secondCentralDifferencesMatrix(xs)

    f = np.sin(xs)
    expectedDf = -np.sin(xs)
    aprox = secondDif.dot(f)

    df = np.gradient(f, xs)
    ddf = np.gradient(df, xs)
    numpyVersion = ddf

    import matplotlib.pyplot as plt

    plt.plot(xs, expectedDf, label="exact solution of d² sin x/dx²")
    plt.scatter(xs, aprox, marker=".", label="second differences δ² sin x/δx²", c="red")  # type: ignore
    plt.scatter(xs, numpyVersion, marker=".", label="numpy version of second differences", c="black")  # type: ignore
    plt.ylim(-3, 3)
    plt.legend()
    plt.show()


def testModelSKappavsMESAKappa():
    ps = modelS.pressures
    ts = modelS.temperatures
    kappas = modelS.derivedQuantities["kappas"]
    zs = modelS.zs
    from opacity import mesaOpacity, modelSNearestOpacity

    mesaOpacities = mesaOpacity(ps, ts)
    interpolatedOpacities = modelSNearestOpacity(ps, ts)

    plt.loglog(zs, kappas, label="model S kappa")
    plt.loglog(
        zs, interpolatedOpacities, label="interpolated model S kappa", linestyle="--"
    )
    plt.loglog(zs, mesaOpacities, label="mesa kappa")
    plt.legend()
    plt.show()


def testIDLAsDatapoint():
    zs, ps, ts, rhos, kappas, hs, nablaRads, nablaAds, nablaTots = np.loadtxt(
        "debuggingReferenceFromSvanda/idlOutput.dat", unpack=True, skiprows=1
    )

    idlDatapoint = SingleTimeDatapoint(
        ts,
        ps,
        zs,
        rhos=rhos,
        kappas=kappas,
        hs=hs,
        nablarads=nablaRads,
        nablaads=nablaAds,
        nablatots=nablaTots,
    )

    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity, modelSNearestOpacity
    from gravity import g, massBelowZ

    convectiveAlpha = 0.3
    calmSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=1e-2,
        lnSurfacePressure=np.log(ps[0]),
        surfaceTemperature=ts[0],
        surfaceZ=zs[0],
        maxDepth=160 * c.Mm,
        opacityFunction=modelSNearestOpacity,
        guessTheZRange=True,
    )
    calmSun.derivedQuantities["rhos"] = MESAEOS.density(
        calmSun.temperatures, calmSun.pressures
    )
    calmSun.derivedQuantities["kappas"] = modelSNearestOpacity(
        calmSun.pressures, calmSun.temperatures
    )
    calmSun.derivedQuantities["hs"] = MESAEOS.pressureScaleHeight(
        calmSun.temperatures, calmSun.pressures, g(calmSun.zs)
    )
    calmSun.derivedQuantities["nablarads"] = MESAEOS.radiativeLogGradient(
        calmSun.temperatures,
        calmSun.pressures,
        massBelowZ(calmSun.zs),
        calmSun.derivedQuantities["kappas"],
    )
    calmSun.derivedQuantities["nablaads"] = MESAEOS.adiabaticLogGradient(
        calmSun.temperatures, calmSun.pressures
    )
    calmSun.derivedQuantities["nablatots"] = np.minimum(
        calmSun.derivedQuantities["nablarads"], calmSun.derivedQuantities["nablaads"]
    )

    calmSunWithMesaOpacity = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=1e-2,
        lnSurfacePressure=np.log(ps[0]),
        surfaceTemperature=ts[0],
        surfaceZ=zs[0],
        maxDepth=160 * c.Mm,
        opacityFunction=mesaOpacity,
        guessTheZRange=True,
    )
    calmSunWithMesaOpacity.derivedQuantities["rhos"] = MESAEOS.density(
        calmSunWithMesaOpacity.temperatures, calmSunWithMesaOpacity.pressures
    )
    calmSunWithMesaOpacity.derivedQuantities["kappas"] = mesaOpacity(
        calmSunWithMesaOpacity.pressures, calmSunWithMesaOpacity.temperatures
    )
    calmSunWithMesaOpacity.derivedQuantities["hs"] = MESAEOS.pressureScaleHeight(
        calmSunWithMesaOpacity.temperatures,
        calmSunWithMesaOpacity.pressures,
        g(calmSunWithMesaOpacity.zs),
    )
    calmSunWithMesaOpacity.derivedQuantities[
        "nablarads"
    ] = MESAEOS.radiativeLogGradient(
        calmSunWithMesaOpacity.temperatures,
        calmSunWithMesaOpacity.pressures,
        massBelowZ(calmSunWithMesaOpacity.zs),
        calmSunWithMesaOpacity.derivedQuantities["kappas"],
    )
    calmSunWithMesaOpacity.derivedQuantities["nablaads"] = MESAEOS.adiabaticLogGradient(
        calmSunWithMesaOpacity.temperatures, calmSunWithMesaOpacity.pressures
    )
    calmSunWithMesaOpacity.derivedQuantities["nablatots"] = np.minimum(
        calmSunWithMesaOpacity.derivedQuantities["nablarads"],
        calmSunWithMesaOpacity.derivedQuantities["nablaads"],
    )

    toPlot = [
        "temperatures",
        "pressures",
        "rhos",
        "kappas",
        "Hs",
        "nablaRads",
        "nablaAds",
        "nablaTots",
    ]

    axs = plotSingleTimeDatapoint(
        calmSun,
        toPlot,
        pltshow=False,
        label="Calm Sun with MESA EOS model S kappa",
        log=True,
    )
    axs = plotSingleTimeDatapoint(
        calmSunWithMesaOpacity,
        toPlot,
        axs=axs,
        pltshow=False,
        label="Calm Sun with MESA EOS mesa kappa",
        log=True,
        linestyle="--",
    )
    axs = plotSingleTimeDatapoint(
        idlDatapoint, toPlot, pltshow=False, axs=axs, label="IDL", log=True
    )
    axs = plotSingleTimeDatapoint(
        modelS,
        toPlot,
        axs=axs,
        pltshow=False,
        label="model S",
        log=True,
        linestyle="--",
    )

    for ax in axs.values():
        ax.set_xlim(8e-1, 26)
        ax.autoscale_view(tight=True)

    plt.show()

    modSData = Data(1, 1)
    modSData.appendDatapoint(modelS)
    modSData.saveToFolder("debuggingReferenceFromSvanda/modelSData")

    idlData = Data(1, 1)
    idlData.appendDatapoint(idlDatapoint)
    idlData.saveToFolder("debuggingReferenceFromSvanda/idlData")

    calmSunData = Data(1, 1)
    calmSunData.appendDatapoint(calmSun)
    calmSunData.saveToFolder("debuggingReferenceFromSvanda/calmSunData")

    calmSunWithMesaOpacityData = Data(1, 1)
    calmSunWithMesaOpacityData.appendDatapoint(calmSunWithMesaOpacity)
    calmSunWithMesaOpacityData.saveToFolder(
        "debuggingReferenceFromSvanda/calmSunWithMesaOpacityData"
    )


def testBottomUpVsTopDown():
    surfaceZ = 1 * c.Mm
    surfaceP = interp1d(modelS.zs, modelS.pressures)(surfaceZ)
    surfaceT = interp1d(modelS.zs, modelS.temperatures)(surfaceZ)

    dLnp = 1e-1
    from stateEquationsPT import MESAEOS
    from opacity import modelSNearestOpacity

    topDownSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=dLnp,
        lnSurfacePressure=np.log(surfaceP),
        surfaceTemperature=surfaceT,
        surfaceZ=surfaceZ,
        maxDepth=30 * c.Mm,
        opacityFunction=modelSNearestOpacity,
        guessTheZRange=True,
    )

    topZ = topDownSun.zs[-1]
    topP = topDownSun.pressures[-1]
    topT = topDownSun.temperatures[-1]

    bottomUpSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=-dLnp,
        lnSurfacePressure=np.log(topP),
        surfaceTemperature=topT,
        surfaceZ=topZ,
        maxDepth=surfaceZ,
        opacityFunction=modelSNearestOpacity,
        guessTheZRange=True,
    )

    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(
        topDownSun, toPlot, pltshow=False, label="top down", log=True
    )
    axs = plotSingleTimeDatapoint(
        bottomUpSun,
        toPlot,
        axs=axs,
        pltshow=False,
        label="bottom up",
        log=True,
        linestyle="--",
    )
    plt.show()


def plotCalmSunZs():
    modelSZs = modelS.zs
    modelSPressures = modelS.pressures
    modelSTemperatures = modelS.temperatures
    from scipy.interpolate import interp1d

    surfaceZ = 0 * c.Mm
    surfaceT = interp1d(modelSZs, modelSTemperatures)(surfaceZ)
    surfaceP = interp1d(modelSZs, modelSPressures)(surfaceZ)

    dlnp = 1e-3
    maxDepth = 16 * c.Mm

    from stateEquationsPT import MESAEOS
    from opacity import modelSNearestOpacity

    import time

    now = time.time()
    calmSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=dlnp,
        lnSurfacePressure=np.log(surfaceP),
        surfaceTemperature=surfaceT,
        surfaceZ=surfaceZ,
        maxDepth=maxDepth,
        opacityFunction=modelSNearestOpacity,
        guessTheZRange=True,
    )
    print("time elapsed: ", time.time() - now)

    indeces = np.arange(len(calmSun.zs))
    plt.plot(indeces, calmSun.zs)
    plt.show()


def benchmarkDifferentLinearInterpolations():
    tests = 100000
    points = 1000
    xs = np.linspace(1, 100, points)
    expxs = np.logspace(0, 2, points)

    ys = np.sin(expxs)

    from scipy.interpolate import interp1d
    from scipy.interpolate import UnivariateSpline

    now = time.time()
    for _ in range(tests):
        interp1d(expxs, ys)(xs)
    print("interp1d: ", time.time() - now)

    now = time.time()
    for _ in range(tests):
        UnivariateSpline(expxs, ys, k=1)(xs)
    print("UnivariateSpline: ", time.time() - now)

    now = time.time()
    for _ in range(tests):
        np.interp(xs, expxs, ys)
    print("numpy: ", time.time() - now)


def bechmarkAndTestDifferentQuadratures():
    nPoints = 2**10 + 1
    xs = np.logspace(0, np.log10(np.pi), nPoints)
    ys = np.sin(xs)
    expectedResult = 2
    numberOfTests = 10000

    from scipy.integrate import trapz, simps, romb, quad

    result = 0
    now = time.time()
    for _ in range(numberOfTests):
        if np.isclose(xs[0] - xs[1], xs[1] - xs[2]):
            pass
        result = trapz(ys, xs)
    print(
        f"trapz: \t{(time.time()-now):.3e} error: \t{np.abs(result-expectedResult):.3e}"
    )

    result = 0
    now = time.time()
    for _ in range(numberOfTests):
        if np.isclose(xs[0] - xs[1], xs[1] - xs[2]):
            pass
        result = simps(ys, xs)
    print(
        f"simps: \t{(time.time()-now):.3e} error: \t{np.abs(result-expectedResult):.3e}"
    )

    result = 0
    now = time.time()
    for _ in range(numberOfTests):
        if np.isclose(xs[0] - xs[1], xs[1] - xs[2]):
            pass
        result = np.trapz(ys, xs)
    print(f"np: \t{(time.time()-now):.3e} error: \t{np.abs(result-expectedResult):.3e}")


def benchamrkLinearInterpolationsWithRandomAccesses():
    nPoints = 2**10 + 1
    xs = np.logspace(0, np.log10(np.pi), nPoints)
    ys = np.sin(xs)
    numberOfTests = 1000
    numberOfAccesses = 1000

    from scipy.interpolate import interp1d
    from scipy.interpolate import UnivariateSpline

    accesses = np.random.random(numberOfAccesses) * (np.pi - 1) + 1

    now = time.time()
    for _ in range(numberOfTests):
        interp = interp1d(xs, ys)
        for access in accesses:
            interp(access)
    print(f"interp1d: \t{(time.time()-now):.3e}")

    now = time.time()
    for _ in range(numberOfTests):
        interp = UnivariateSpline(xs, ys, k=1)
        for access in accesses:
            interp(access)
    print(f"UnivariateSpline: \t{(time.time()-now):.3e}")

    now = time.time()
    for _ in range(numberOfTests):
        for access in accesses:
            np.interp(access, xs, ys)
    print(f"numpy: \t{(time.time()-now):.3e}")


def main():
    benchamrkLinearInterpolationsWithRandomAccesses()


if __name__ == "__main__":
    main()
