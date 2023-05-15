#!/usr/bin/env python3

import numpy as np
from dataHandling.initialConditionsSetterUpper import getInitialConditions, getBartaInit
from dataHandling.modelS import loadModelS
from dataHandling.dataStructure import Data, SingleTimeDatapoint
import constants as c
from sunSolvers.calmSun import getCalmSunDatapoint
from sunSolvers.temperatureSolvers import oldTSolver
from sunSolvers.pressureSolvers import integrateHydrostaticEquilibrium
from sunSolvers.magneticSolvers import oldYSolver
from stateEquationsPT import MESAEOS
from opacity import mesaOpacity


modelS = loadModelS()


def main(
    initialConditions: SingleTimeDatapoint,
    backgroundReference: SingleTimeDatapoint = modelS,
    finalT: float = 100,
    numberOfTSteps: int = 2**4,
    maxDepth: float = 100,
    outputFolderName: str = "output",
    dlnP: float = 1e-2,
    dt: float = 1e-2,
) -> None:
    """Simulates the evolution of a sunspot and saves the simulation into a folder


    Args:
        initialConditions (SingleTimeDatapoint): Initial conditions of the simulation. Surface temperature of this model will stay constant during the simulation # FIXME maybe the surface temperature should be separate from the initial conditions?
        backgroundReference (SingleTimeDatapoint, optional): Model on which to base calm Sun model. Should be properly deep. Defaults to modelS. # FIXME this is kinda lame I actually only need surface z, pressure and temperature
        finalT (float, optional): Total length of the simulation in hours. Defaults to 100.
        numberOfTSteps (int, optional): How many time steps to take from 0 to finalT. Defaults to 2**4.
        maxDepth (float, optional): Approximate maximum depth of the simulation in Mm. Defaults to 100. # TODO check that these dont get too weird, you havent rly made sure that the max pressure well corresponds to max Z
        outputFolderName (str, optional): Name of the folder in which to save the output of the simulation. Defaults to "output".
        dlnpP (float, optional): Step by which to integrate hydrostatic equilibrium in ln(Pa). Defaults to 1e-2.
        dt (float, optional): Step by which to integrate temperature in hours. Defaults to 1e-2.

    Raises:
        NotImplementedError: _description_
    """

    # turn user input to SI
    finalT *= c.hour
    maxDepth *= c.Mm

    # first prepare calm sun as a reference of background outside pressure
    calmSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        opacityFunction=mesaOpacity,
        dlnP=dlnP,
        lnSurfacePressure=np.log(backgroundReference.pressures[-1]),
        surfaceTemperature=backgroundReference.temperatures[-1],
        surfaceZ=backgroundReference.zs[-1],
        maxDepth=maxDepth,
    )

    externalPressures = calmSun.pressures[:]
    # only P_e is important from the background model
    # these *can* be be quite big, get rid of them
    del calmSun
    del backgroundReference

    # create empty data structure with only initial conditions
    data = Data(finalT=finalT, numberOfTSteps=numberOfTSteps)
    numberOfTSteps = data.numberOfTSteps
    currentState = initialConditions
    data.addDatapointAtIndex(currentState, 0)
    lastYs = np.sqrt(
        currentState.Bs
    )  # these will be used as initial guess for the magnetic equation
    surfaceTemperature = currentState.temperatures[
        0
    ]  # this will be held constant during the simulation

    time = 0
    while time < finalT:
        time += dt  # TODO maybe use non constant dt?

        # first integrate the temperature equation
        newTs = oldTSolver(
            currentState=currentState,
            StateEq=MESAEOS,
            dt=dt,
            opacityFunction=mesaOpacity,
            surfaceTemperature=surfaceTemperature,
        )
        # with new temperatures now get new bottom pressure from inflow of material
        boundaryPressure = 0

        # then integrate hydrostatic equilibrium from bottom to the top
        initialZ = currentState.zs[-1]
        finalZ = currentState.zs[0]

        newZs, newPs = integrateHydrostaticEquilibrium(
            temperatures=newTs,
            zs=currentState.zs,
            StateEq=MESAEOS,
            dlnP=dlnP,
            lnBoundaryPressure=np.log(boundaryPressure),
            initialZ=initialZ,
            finalZ=finalZ,
        )

        # finally solve the magnetic equation
        newYs = oldYSolver(
            newZs,
            newPs,
            externalPressures[:],
            totalMagneticFlux=1e20,
            yGuess=lastYs,
            tolerance=1e-6,
        )
        lastYs = newYs
        newBs = newYs * newYs

        # and save the new datapoint
        currentState = SingleTimeDatapoint(
            zs=newZs,
            temperatures=newTs,
            pressures=newPs,
            Bs=newBs,
        )

        data.appendDatapoint(currentState)

    data.saveToFolder(outputFolderName)
    raise NotImplementedError("this is not finished yet")
    visualizeData(data)


if __name__ == "__main__":
    maxDepth = 100  # depth in Mm
    minDepth = 1  # depth in Mm
    p0_ratio = 1  # ratio of initial pressure to the pressure at the top of the model S
    surfaceTemperature = 3500  # temperature in K
    numberOfZSteps = 100
    dlnP = 1e-2

    initialConditions = getBartaInit(p0_ratio, maxDepth, minDepth, dlnP=dlnP)

    finalT = 100  # final time in hours
    numberOfTSteps = 32  # number of time steps

    main(
        initialConditions,
        finalT=finalT,
        maxDepth=maxDepth,
        numberOfTSteps=numberOfTSteps,
        outputFolderName="FirstTestHaha",
    )
