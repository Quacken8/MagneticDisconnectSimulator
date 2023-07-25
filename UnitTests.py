#!/usr/bin/env python3
from turtle import title
from matplotlib.backend_bases import ToolContainerBase
from matplotlib.scale import LogScale
from py import log
from sympy import li
from dataHandling.dataStructure import (
    Data,
    SingleTimeDatapoint,
    createDataFromFolder,
    dictionaryOfVariables,
)
import numpy as np
from dataHandling.initialConditionsSetterUpper import mockupDataSetterUpper
from dataHandling.modelS import interpolatedF_con, loadModelS
from opacity import mesaOpacity
from stateEquationsPT import MESAEOS, IdealGas
from sunSolvers.calmSun import getCalmSunDatapoint
from dataHandling.dataVizualizer import plotSingleTimeDatapoint, plotData
from sunSolvers.pressureSolvers import (
    integrateAdiabaticHydrostaticEquilibrium,
    integrateHydrostaticEquilibriumAndTemperatureGradient,
)
import time

# from solvers import oldYSolver
from dataHandling.modelS import loadModelS
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import constants as c
import os
from matplotlib import lines, pyplot as plt
import loggingConfig
import logging

L = loggingConfig.configureLogging(logging.DEBUG, __name__)


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
    )
    calmSunWithModelSOpacity = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=dlnp,
        lnSurfacePressure=np.log(surfaceP),
        surfaceTemperature=surfaceT,
        surfaceZ=surfaceZ,
        maxDepth=maxDepth,
        opacityFunction=modelSNearestOpacity,
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
    from sunSolvers.handySolverStuff import secondCentralDifferencesMatrix

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

    now = time.time()
    calmSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=1e-2,
        lnSurfacePressure=np.log(ps[0]),
        surfaceTemperature=ts[0],
        surfaceZ=zs[0],
        maxDepth=160 * c.Mm,
        opacityFunction=modelSNearestOpacity,
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
    print("time elapsed: ", time.time() - now)

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


def getModelSEntropyFromMesa():
    Ts = modelS.temperatures
    Ps = modelS.pressures
    from stateEquationsPT import MESAEOS

    entropies = MESAEOS.entropy(Ts, Ps)
    print(entropies)


def testBartaInitialConditions():
    from dataHandling.initialConditionsSetterUpper import getBartaInit

    p0ratio = 1
    maxDepth = 16 * c.Mm
    surfaceDepth = 0 * c.Mm
    dlnP = 1e-3
    initialModel = getBartaInit(p0ratio, maxDepth, surfaceDepth, dlnP)
    from stateEquationsPT import MESAEOS

    initialModel.derivedQuantities["entropies"] = MESAEOS.entropy(
        initialModel.temperatures, initialModel.pressures
    )
    toPlot = ["temperatures", "pressures", "entropies", "bs"]
    axs = plotSingleTimeDatapoint(
        initialModel, toPlot, pltshow=False, label="Barta initial model", log=False
    )
    plotSingleTimeDatapoint(modelS, toPlot, axs=axs, label="Model S", log=True)


def testAdiabaticHydrostaticEquilibrium():
    from stateEquationsPT import MESAEOS
    from sunSolvers.pressureSolvers import (
        integrateHydrostaticEquilibriumAndTemperatureGradient,
    )
    from opacity import mesaOpacity

    surfaceZ = 0 * c.Mm
    maxDepth = 160 * c.Mm
    surfaceT = np.interp(surfaceZ, modelS.zs, modelS.temperatures).item()
    surfaceP = np.interp(surfaceZ, modelS.zs, modelS.pressures).item()
    dlnP = 1e-2
    initialModel = integrateAdiabaticHydrostaticEquilibrium(
        StateEq=MESAEOS,
        dlnP=dlnP,
        lnBoundaryPressure=np.log(surfaceP),
        boundaryTemperature=surfaceT,
        initialZ=surfaceZ,
        finalZ=maxDepth,
    )
    initialModel.derivedQuantities["entropies"] = MESAEOS.entropy(
        initialModel.temperatures, initialModel.pressures
    )

    nonAdiabaticiInitialModel = integrateHydrostaticEquilibriumAndTemperatureGradient(
        StateEq=MESAEOS,
        dlnP=dlnP,
        lnBoundaryPressure=np.log(surfaceP),
        boundaryTemperature=surfaceT,
        initialZ=surfaceZ,
        finalZ=maxDepth,
        opacityFunction=mesaOpacity,
    )
    nonAdiabaticiInitialModel.derivedQuantities["entropies"] = MESAEOS.entropy(
        nonAdiabaticiInitialModel.temperatures, nonAdiabaticiInitialModel.pressures
    )
    calmSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        dlnP=dlnP,
        lnSurfacePressure=np.log(surfaceP),
        surfaceTemperature=surfaceT,
        surfaceZ=surfaceZ,
        maxDepth=maxDepth,
        opacityFunction=mesaOpacity,
    )
    calmSun.derivedQuantities["entropies"] = MESAEOS.entropy(
        calmSun.temperatures, calmSun.pressures
    )

    toPlot = ["entropies", "temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(
        modelS, toPlot, pltshow=False, label="Model S", log=False
    )
    axs = plotSingleTimeDatapoint(
        calmSun, toPlot, axs=axs, label="Calm Sun", log=False, pltshow=False
    )
    axs = plotSingleTimeDatapoint(
        nonAdiabaticiInitialModel,
        toPlot,
        axs=axs,
        label="Non adiabatic Hydrostatic Equilibrium",
        log=False,
        pltshow=False,
        linestyle=":",
    )
    plotSingleTimeDatapoint(
        initialModel,
        toPlot,
        pltshow=True,
        axs=axs,
        label="Adiabatic Hydrostatic Equilibrium",
        log=False,
        linestyle="--",
    )


def testHydrostaticEquilibrium():
    Ts = modelS.temperatures
    zs = modelS.zs

    zMax = 160 * c.Mm
    zMin = 0 * c.Mm
    dlnP = 1e-2
    pBottom = np.interp(zMin, zs, modelS.pressures).item()

    from sunSolvers.pressureSolvers import integrateHydrostaticEquilibrium
    from stateEquationsPT import MESAEOS

    initialModel = integrateHydrostaticEquilibrium(
        StateEq=MESAEOS,
        dlnP=dlnP,
        lnBoundaryPressure=np.log(pBottom),
        referenceTs=Ts,
        referenceZs=zs,
        initialZ=zMin,
        finalZ=zMax,
    )

    toPlot = ["pressures"]
    axs = plotSingleTimeDatapoint(
        modelS, toPlot, pltshow=False, label="Model S", log=False
    )
    plotSingleTimeDatapoint(
        initialModel,
        toPlot,
        pltshow=True,
        axs=axs,
        label="Hydrostatic Equilibrium",
        log=True,
        linestyle="--",
    )


def testYSolver():
    # we define some sensible function
    # run it throuh the pressure solver
    # and try to put the result back into the function
    # probably using np derivatives

    from sunSolvers.magneticSolvers import oldYSolver, rightHandSideOfYEq
    from dataHandling.boundaryConditions import getBottomB, getTopB

    # here we define function on which we work
    nodes = 10
    zs = np.linspace(0, 10, nodes)
    innerPs = np.linspace(1e6, 1e10, nodes)
    outerPs = np.logspace(6, 12, nodes)
    totalMagneticFlux = 1e3
    bottomB, topB = (
        getBottomB(bottomPressure=innerPs[-1], externalPressure=outerPs[-1]),
        getTopB(),
    )
    yGuess = np.linspace(np.sqrt(topB), np.sqrt(bottomB), nodes)
    tolerance = 1e-3

    lhs = np.gradient(np.gradient(yGuess, zs), zs)
    rhs = rightHandSideOfYEq(
        y=yGuess,
        innerP=innerPs,
        outerP=outerPs,
        totalMagneticFlux=totalMagneticFlux,
    )
    print("original error: ", np.max(np.abs(lhs - rhs)))

    solution = oldYSolver(
        zs=zs,
        innerPs=innerPs,
        outerPs=outerPs,
        totalMagneticFlux=totalMagneticFlux,
        yGuess=yGuess,
        tolerance=tolerance,
    )

    lhs = np.gradient(np.gradient(solution, zs), zs)

    rhs = rightHandSideOfYEq(
        y=solution,
        innerP=innerPs,
        outerP=outerPs,
        totalMagneticFlux=totalMagneticFlux,
    )

    print("max error: ", np.max(np.abs(lhs - rhs)))
    plt.plot(zs, solution, label="sol")
    plt.legend()
    plt.show()


def testIntegrateMagneticField():
    # we define some sensible function
    # run it throuh the pressure solver
    # and try to put the result back into the function
    # probably using np derivatives

    from sunSolvers.magneticSolvers import integrateMagneticEquation, rightHandSideOfYEq
    from dataHandling.boundaryConditions import getBottomB, getTopB

    # here we define function on which we work
    nodes = 1000
    zs = np.linspace(0, 10, nodes)
    innerPs = np.linspace(1e6, 1e10, nodes)
    outerPs = np.logspace(7, 12, nodes)
    totalMagneticFlux = 1e5
    bottomB, topB = (
        getBottomB(bottomPressure=innerPs[-1], externalPressure=outerPs[-1]),
        getTopB(),
    )
    yGuess = np.linspace(np.sqrt(topB), np.sqrt(bottomB), nodes)[::-1]
    tolerance = 1e-3

    solution = integrateMagneticEquation(
        zs=zs,
        innerPs=innerPs,
        outerPs=outerPs,
        totalMagneticFlux=totalMagneticFlux,
        yGuess=yGuess,
        tolerance=tolerance,
    )

    rhs = rightHandSideOfYEq(
        y=solution,
        innerP=innerPs,
        outerP=outerPs,
        totalMagneticFlux=totalMagneticFlux,
    )
    lhs = np.gradient(np.gradient(solution, zs), zs)
    print("max error: ", np.max(np.abs(lhs - rhs)))

    plt.loglog(zs, lhs, label="lhs")
    plt.loglog(zs, rhs, label="rhs", linestyle="--")
    plt.legend()
    plt.show()


def testTemperatureSolver():
    # we define some sensible function
    # run it throuh the pressure solver
    # and try to put the result back into the function
    # probably using np derivatives

    from sunSolvers.temperatureSolvers import oldTSolver, rightHandSideOfTEq
    from stateEquationsPT import MESAEOS
    from opacity import mesaOpacity
    from gravity import g, massBelowZ
    from scipy.interpolate import interp1d

    initialState = modelS
    oldTs = initialState.temperatures
    dt = 1e-3
    surfaceTemperature = initialState.temperatures[0]
    convectiveAlpha = 1

    newTs = oldTSolver(
        currentState=initialState,
        dt=dt,
        StateEq=MESAEOS,
        opacityFunction=mesaOpacity,
        surfaceTemperature=surfaceTemperature,
        convectiveAlpha=convectiveAlpha,
    )

    dTdt = (newTs - oldTs) / dt
    rhs = rightHandSideOfTEq(
        convectiveAlpha=convectiveAlpha,
        zs=initialState.zs,
        temperatures=initialState.temperatures,
        pressures=initialState.pressures,
        opacityFunction=mesaOpacity,
        StateEq=MESAEOS,
    )

    plt.plot(initialState.zs, dTdt, label="dTdt")
    plt.plot(initialState.zs, rhs, label="rhs")
    plt.legend()
    plt.show()

    raise NotImplementedError()


def lookAtInterruptedData():
    data = createDataFromFolder("interruptedRun")
    toPlot = ["bs"]
    plotSingleTimeDatapoint(data.datapoints[0], toPlot, pltshow=True, log=False)


def plotPressureAndTempModelS():
    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(
        modelS, toPlot, pltshow=False, label="Model S", log=False
    )
    plt.show()


def plotModelSCpsVsMESA():
    from stateEquationsPT import MESAEOS

    mesaCps = MESAEOS.Cp(modelS.temperatures, modelS.pressures)
    modelSCps = modelS.derivedQuantities["cps"]
    plt.loglog(modelS.zs, modelSCps, label="model S")
    plt.loglog(modelS.zs, mesaCps, label="mesa")
    plt.legend()
    plt.show()


def compareCalmSunFromFileWithModelS():
    from dataHandling.dataStructure import loadOneTimeDatapoint

    calmSun = loadOneTimeDatapoint("calmSun")

    toPlot = ["temperatures", "pressures"]
    axs = plotSingleTimeDatapoint(
        calmSun, toPlot, pltshow=False, label="Calm Sun", log=True
    )
    axs = plotSingleTimeDatapoint(
        modelS, toPlot, axs=axs, pltshow=False, label="Model S", log=True
    )
    plt.show()


def plotInterruptedRun():
    data = createDataFromFolder("interruptedRun")
    toPlot = ["temperatures", "pressures", "bs"]

    axs = plotData(data, toPlot, pltshow=False, title="interrupted run", log=True)
    plt.show()


def compareTotalGradients():
    totalGradNp = np.gradient(np.log(modelS.temperatures), np.log(modelS.pressures))
    import gravity
    totalGradMesa = MESAEOS.actualGradient(
        temperature=modelS.temperatures[:-1],
        pressure=modelS.pressures[:-1],
        massBelowZ=gravity.massBelowZ(modelS.zs[:-1]),
        opacity=modelS.derivedQuantities["kappas"][:-1],
    )

    plt.plot(modelS.zs, totalGradNp, label="np")
    plt.plot(modelS.zs[:-1], totalGradMesa, label="mesa", linestyle="--")
    plt.legend()
    plt.show()

def modelSFconsVSMESAFcons():
    from gravity import g, massBelowZ
    from sunSolvers.handySolverStuff import centralDifferencesMatrix
    modelS.derivedQuantities["f_cons"] *= -1 
    toPlot = ["f_cons", "f_rads"]
    convectiveAlpha = 1

    myDatapoint = SingleTimeDatapoint(
        modelS.temperatures[:-3],
        modelS.pressures[:-3],
        modelS.zs[:-3],
    )
    mygs = g(myDatapoint.zs)
    myMasses = massBelowZ(myDatapoint.zs)
    myTgrad = centralDifferencesMatrix(myDatapoint.zs).dot(myDatapoint.temperatures)
    myOpacity = mesaOpacity(myDatapoint.pressures, myDatapoint.temperatures)
    myTotalGradient = np.gradient(np.log(myDatapoint.temperatures), np.log(myDatapoint.pressures))
    myDatapoint.derivedQuantities["f_cons"] = -MESAEOS.f_con(
        temperature=myDatapoint.temperatures,
        pressure=myDatapoint.pressures,
        convectiveAlpha=convectiveAlpha,
        gravitationalAcceleration=mygs,
        massBelowZ=myMasses,
        opacity=myOpacity,
    )
    myDatapoint.derivedQuantities["f_rads"] = MESAEOS.f_rad(
        temperature=myDatapoint.temperatures,
        pressure=myDatapoint.pressures,
        opacity=myOpacity,
        dTdz=myTgrad,
    )

    axs = plotSingleTimeDatapoint(
        modelS, toPlot, pltshow=False, label="Model S", log=True, linestyle="--"
    )
    axs = plotSingleTimeDatapoint(
        myDatapoint, toPlot, axs=axs, pltshow=False, label="MESA", log=True
    )
    plt.show()

def nablaTick(temperature, convectiveAlpha, c_p, density, opacity, Hp, gravitationalAcceleration, adiabaticGradient, actualGradient):
        # these are geometric parameters of convection used in Schüssler & Rempel 2005
    a = 0.125
    b = 0.5
    f = 1.5

    u = (
        1
        / (f * np.sqrt(a))
        * convectiveAlpha
        * convectiveAlpha
        * 12
        * c.SteffanBoltzmann
        * temperature
        * temperature
        * temperature
        / (c_p * density * opacity * Hp * Hp)
        * np.sqrt(Hp / gravitationalAcceleration)
    )
    # this is sometimes called ∇' (schussler & rempel 2005) or ∇_e (Lattanzio in thier class M4111 of 2009) where e stands for element of stellar matter
    # it, according to Schüssler & Rempel 2005, "reflects radiative energy exchange of the convective parcels"
    gradTick = (
        adiabaticGradient
        - 2 * u * u
        + 2 * u * np.sqrt(np.maximum(actualGradient - adiabaticGradient + u * u, 0))
    )
    return gradTick

def compareGradTicks():
    import gravity
    modelSGradTicks = modelS.derivedQuantities["nablaPrimes"]

    nablaRad = MESAEOS.radiativeLogGradient(
        temperature=modelS.temperatures[:-1],
        pressure=modelS.pressures[:-1],
        massBelowZ=gravity.massBelowZ(modelS.zs[:-1]),
        opacity=modelS.derivedQuantities["kappas"][:-1],
    )
    nablaRad = np.append(nablaRad, nablaRad[-1])

    mesaGradTicks = nablaTick(
        temperature=modelS.temperatures,
        convectiveAlpha=1.9904568,
        c_p=modelS.derivedQuantities["cps"],
        density=modelS.derivedQuantities["rhos"],
        opacity=modelS.derivedQuantities["kappas"],
        Hp=MESAEOS.pressureScaleHeight(modelS.temperatures, modelS.pressures, gravity.g(modelS.zs)),
        gravitationalAcceleration=gravity.g(modelS.zs),
        adiabaticGradient=modelS.derivedQuantities["nablaads"],
        actualGradient=np.minimum(modelS.derivedQuantities["nablaads"], nablaRad),
    )

    plt.plot(modelS.zs, modelSGradTicks, label="model S")
    plt.plot(modelS.zs, mesaGradTicks, label="mesa", linestyle="--")
    plt.legend()
    plt.show()

def testModelSMuvsMesaMu():
    modelSMus = modelS.derivedQuantities["mu_unis"]
    mesaMus = MESAEOS.meanMolecularWeight(modelS.temperatures, modelS.pressures)/(c.m_u*c.N_A)
    plt.loglog(modelS.zs/c.Mm, modelSMus, label="model S")
    plt.loglog(modelS.zs/c.Mm, mesaMus, label="mesa", linestyle="--")
    plt.ylabel(r"$\mu$")
    plt.xlabel("z [Mm]")
    plt.legend()
    plt.show()

def testMesaKappaVsModelSKappa():
    modelSKappas = modelS.derivedQuantities["kappas"]
    mesaKappas = mesaOpacity(modelS.pressures, modelS.temperatures)
    plt.loglog(modelS.zs/c.Mm, modelSKappas, label="model S")
    plt.loglog(modelS.zs/c.Mm, mesaKappas, label="mesa", linestyle="--")
    plt.ylabel(r"$\kappa$ [cm$^2$ g$^{-1}$]")
    plt.xlabel("z [Mm]")
    plt.legend()
    plt.show()

def testMesaNablaRadvsModelSNablaRad():
    import gravity
    modelSNablaRads = modelS.derivedQuantities["nablarads"]
    mesaNablaRads = MESAEOS.radiativeLogGradient(
        temperature=modelS.temperatures[:-1],
        pressure=modelS.pressures[:-1],
        massBelowZ=gravity.massBelowZ(modelS.zs[:-1]),
        opacity=modelS.derivedQuantities["kappas"][:-1],
    )
    mesaNablaRads = np.append(mesaNablaRads, mesaNablaRads[-1])
    plt.plot(modelS.zs, modelSNablaRads, label="model S")
    plt.plot(modelS.zs, mesaNablaRads, label="mesa", linestyle="--")
    plt.legend()
    plt.show()


def testMesaNablaRadsVsModelSNablaRads():
    import gravity
    modelSNablaRads = modelS.derivedQuantities["nabla_rads"]
    mesaNablaRads = MESAEOS.radiativeLogGradient(
        temperature=modelS.temperatures[:-1],
        pressure=modelS.pressures[:-1],
        massBelowZ=gravity.massBelowZ(modelS.zs[:-1]),
        opacity=modelS.derivedQuantities["kappas"][:-1],
    )
    mesaNablaRads = np.append(mesaNablaRads, mesaNablaRads[-1])
    modelSNablaAds = modelS.derivedQuantities["nablaads"]
    plt.loglog(modelS.zs, modelSNablaRads, label="model S")
    plt.loglog(modelS.zs, mesaNablaRads, label="mesa", linestyle="--")
    plt.loglog(modelS.zs, modelSNablaAds, label="model S adiabatic")
    plt.legend()
    plt.show()

def plotNablaTickVsNablaTot():
    import gravity

    mesaGradTicks = nablaTick(
        temperature=modelS.temperatures[:-1],
        convectiveAlpha=1.9904568,
        c_p=MESAEOS.Cp(modelS.temperatures, modelS.pressures)[:-1],
        density=MESAEOS.density(modelS.temperatures, modelS.pressures)[:-1],
        opacity=mesaOpacity(modelS.pressures, modelS.temperatures)[:-1],
        Hp=MESAEOS.pressureScaleHeight(modelS.temperatures[:-1], modelS.pressures[:-1], gravity.g(modelS.zs[:-1])),
        gravitationalAcceleration=gravity.g(modelS.zs[:-1]),
        adiabaticGradient=MESAEOS.adiabaticLogGradient(modelS.temperatures[:-1], modelS.pressures[:-1]),
        actualGradient=MESAEOS.actualGradient(modelS.temperatures[:-1], modelS.pressures[:-1], gravity.massBelowZ(modelS.zs[:-1]), mesaOpacity(modelS.pressures[:-1], modelS.temperatures[:-1]))
    )
    nablaTots = MESAEOS.actualGradient(
        temperature=modelS.temperatures[:-1],
        pressure=modelS.pressures[:-1],
        massBelowZ=gravity.massBelowZ(modelS.zs[:-1]),
        opacity=modelS.derivedQuantities["kappas"][:-1],
    )

    plt.plot(modelS.zs[:-1], mesaGradTicks, label="tick")
    plt.plot(modelS.zs[:-1], nablaTots, label="nabla tot", linestyle="--")
    plt.legend()
    plt.show()

def testInterpolatedFCons():
    modelTs, modelPs, modelFcons = modelS.temperatures, modelS.pressures, modelS.derivedQuantities["f_cons"]
    
    interpFs = interpolatedF_con(modelTs, modelPs)
    plt.plot(modelTs, modelFcons, label="model")
    plt.plot(modelTs, interpFs, label="interp", linestyle="--")
    plt.legend()
    plt.show()

def compareInitialConiditionsToModelS():
    from dataHandling.initialConditionsSetterUpper import getBartaInit

    p0ratio = 1
    maxDepth = 16 * c.Mm
    surfaceDepth = 0 * c.Mm
    dlnP = 1e-3
    initialModel = getBartaInit(p0ratio, maxDepth, surfaceDepth, dlnP)
    from stateEquationsPT import MESAEOS

    initialModel.derivedQuantities["entropies"] = MESAEOS.entropy(
        initialModel.temperatures, initialModel.pressures
    )
    toPlot = ["temperatures", "pressures", "entropies", "bs"]
    axs = plotSingleTimeDatapoint(
        initialModel, toPlot, pltshow=False, label="Barta initial model", log=False
    )
    plotSingleTimeDatapoint(modelS, toPlot, axs=axs, label="Model S", log=True)


def main():
    testBartaInitialConditions()
    pass
    pass


if __name__ == "__main__":
    L.debug("Starting tests")
    main()
