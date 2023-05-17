#!/usr/bin/env python3

from hmac import new
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
from dataHandling.boundaryConditions import getAdjustedBottomPressure
from loggingConfig import configureLogging
import logging

L = configureLogging(logging.INFO, __name__)


modelS = loadModelS()


def main(
    initialConditions: SingleTimeDatapoint,
    backgroundReference: SingleTimeDatapoint = modelS,
    maxDepth: float = 100,
    upflowVelocity: float = 1e-3,
    totalMagneticFlux: float = 1e20,
    finalT: float = 100,
    numberOfTSteps: int = 2**4,
    outputFolderName: str = "output",
    dlnP: float = 1e-2,
    convectiveAlpha: float = 0.3,
) -> None:
    """Simulates the evolution of a sunspot and saves the simulation into a folder


    Args:
        initialConditions (SingleTimeDatapoint): Initial conditions of the simulation. Surface temperature of this model will stay constant during the simulation # FIXME maybe the surface temperature should be separate from the initial conditions?
        backgroundReference (SingleTimeDatapoint, optional): Model on which to base calm Sun model. Should be properly deep. Defaults to modelS. # FIXME this is kinda lame I actually only need surface z, pressure and temperature
        maxDepth (float, optional): Approximate maximum depth of the simulation in Mm. Defaults to 100. # TODO check that these dont get too weird, you havent rly made sure that the max pressure well corresponds to max Z
        upflowVelocity (float, optional): Velocity of the upflow in Mm/h. Defaults to 1e-3. # FIXME random ass values
        totalMagneticFlux (float, optional): Total magnetic flux in Gauss. Defaults to 1e20. # FIXME random ass values
        finalT (float, optional): Total length of the simulation in hours. Defaults to 100.
        numberOfTSteps (int, optional): How many time steps to take from 0 to finalT. Defaults to 2**4.
        outputFolderName (str, optional): Name of the folder in which to save the output of the simulation. Defaults to "output".
        dlnpP (float, optional): Step by which to integrate hydrostatic equilibrium in ln(Pa). Defaults to 1e-2.

    Raises:
        NotImplementedError: _description_
    """

    # turn user input to SI
    finalT *= c.hour
    maxDepth *= c.Mm
    L.info("Prepairing background reference...")
    # first prepare calm sun as a reference of background outside pressure
    calmMaxDepth = maxDepth * 1.3  # just a bit of padding to be sure
    calmMinDepth = -1e5  # TODO these may need adjusting

    calmSun = getCalmSunDatapoint(
        StateEq=MESAEOS,
        opacityFunction=mesaOpacity,
        dlnP=dlnP,
        lnSurfacePressure=np.log(
            np.interp(
                calmMinDepth, backgroundReference.zs, backgroundReference.pressures
            ).item()
        ),
        surfaceTemperature=np.interp(
            calmMinDepth, backgroundReference.zs, backgroundReference.temperatures
        ).item(),
        surfaceZ=calmMinDepth,
        maxDepth=calmMaxDepth,
    )

    externalzP = (calmSun.zs[:], calmSun.pressures[:])
    # only P_e is important from the background model
    # these *can* be be quite big, get rid of them
    del calmSun
    del backgroundReference

    # create empty data structure with only initial conditions
    data = Data(finalT=finalT, numberOfTSteps=numberOfTSteps)
    dt = finalT / numberOfTSteps
    currentState = initialConditions
    data.addDatapointAtIndex(currentState, 0)
    lastYs = np.sqrt(
        currentState.bs
    )  # these will be used as initial guess for the magnetic equation
    surfaceTemperature = currentState.temperatures[
        0
    ]  # this will be held constant during the simulation
    L.info("Starting simulation...")
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
            convectiveAlpha=convectiveAlpha,
        )
        # with new temperatures now get new bottom pressure from inflow of material
        bottomPe = np.interp(currentState.zs[-1], externalzP[0], externalzP[1]).item()

        boundaryPressure = getAdjustedBottomPressure(currentState=currentState, dt=dt, bottomExternalPressure=bottomPe, upflowVelocity=upflowVelocity, totalMagneticFlux=totalMagneticFlux)

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
        externalPressures = np.interp(newZs, externalzP[0], externalzP[1])
        newYs = oldYSolver(
            newZs,
            newPs,
            externalPressures,
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
            bs=newBs,
        )

        data.appendDatapoint(currentState)
    L.info(f"Simulation finished, saving results to folder {outputFolderName}")
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
