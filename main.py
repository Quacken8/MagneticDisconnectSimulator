#!/usr/bin/env python3

import numpy as np
from dataHandling import initialConditionsSetterUpper as init
from dataHandling import modelS
from dataHandling import dataStructure
import constants as c
from sunSolvers import calmSun
from sunSolvers import temperatureSolvers
from sunSolvers import pressureSolvers
from sunSolvers import magneticSolvers
from stateEquationsPT import MESAEOS
import opacity
from dataHandling import boundaryConditions
import loggingConfig
import logging

L = loggingConfig.configureLogging(logging.INFO, __name__)


modelS = modelS.loadModelS()


def main(
    initialConditions: dataStructure.SingleTimeDatapoint,
    backgroundReference: dataStructure.SingleTimeDatapoint = modelS,
    maxDepth: float = 100,
    upflowVelocity: float = 1e-3,
    totalMagneticFlux: float = 1e13,
    surfaceTemperature: float = 3500,
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
        surfaceTemperature (float, optional): Fixed surface temperature in K. Defaults to 3500 K # FIXME random ass values
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

    try:
        calmModel = dataStructure.loadOneTimeDatapoint("calmSun")
        L.info("Loaded calm Sun from a folder")
    except FileNotFoundError:
        L.info("Couldn't find calm Sun in a folder")
        calmModel = calmSun.getCalmSunDatapoint(
            StateEq=MESAEOS,
            opacityFunction=opacity.mesaOpacity,
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

    externalzP = (calmModel.zs[:], calmModel.pressures[:])
    # only P_e is important from the background model
    # these *can* be be quite big, get rid of them
    del calmModel
    del backgroundReference

    # create empty data structure with only initial conditions
    data = dataStructure.Data(finalT=finalT, numberOfTSteps=numberOfTSteps)
    try:  # this huge try block makes sure data gets saved in case of an error
        dt = finalT / numberOfTSteps
        currentState = initialConditions
        data.addDatapointAtIndex(currentState, 0)
        lastYs = np.sqrt(
            currentState.bs
        )  # these will be used as initial guess for the magnetic equation
        L.info("Starting simulation...")
        time = 0
        while time < finalT:
            time += dt  # TODO maybe use non constant dt?

            # first integrate the temperature equation
            newTs = temperatureSolvers.oldTSolver(
                currentState=currentState,
                dt=dt,
                StateEq=MESAEOS,
                opacityFunction=opacity.mesaOpacity,
                surfaceTemperature=surfaceTemperature,
                convectiveAlpha=convectiveAlpha,
            )
            # with new temperatures now get new bottom pressure from inflow of material
            bottomPe = np.interp(
                currentState.zs[-1], externalzP[0], externalzP[1]
            ).item()

            newBottomP = boundaryConditions.getAdjustedBottomPressure(
                currentState=currentState,
                dt=dt,
                dlnP=dlnP,
                bottomExternalPressure=bottomPe,
                upflowVelocity=upflowVelocity,
                totalMagneticFlux=totalMagneticFlux,
                newTs=newTs,
            )

            # then integrate hydrostatic equilibrium from bottom to the top
            initialZ = currentState.zs[-1]
            finalZ = currentState.zs[0]
            oldPs = currentState.pressures
            currentState = pressureSolvers.integrateHydrostaticEquilibrium(
                referenceTs=newTs,
                referenceZs=currentState.zs,
                StateEq=MESAEOS,
                dlnP=dlnP,
                lnBoundaryPressure=np.log(newBottomP),
                initialZ=initialZ,
                finalZ=finalZ,
                interpolableYs=lastYs,
            )
            newZs = currentState.zs
            newPs = currentState.pressures
            lastYs = np.sqrt(currentState.bs)  # FIXME this may be a bottlenect

            # finally solve the magnetic equation
            externalPressures = np.interp(newZs, externalzP[0], externalzP[1])
            newYs = magneticSolvers.integrateMagneticEquation(
                newZs,
                newPs,
                externalPressures,
                totalMagneticFlux=totalMagneticFlux,
                yGuess=lastYs,
                tolerance=1e-6,
            )
            lastYs = newYs
            newBs = newYs * newYs
            currentState.bs = newBs
            currentState.allVariables["bs"] = newBs

            # and save the new datapoint

            data.appendDatapoint(currentState)
        L.info(f"Simulation finished, saving results to folder {outputFolderName}")
        data.saveToFolder(outputFolderName)
        raise NotImplementedError("this is not finished yet")
    except Exception as e:
        data.saveToFolder("interruptedRun", rewriteFolder=True)
        raise e
    visualizeData(data)


if __name__ == "__main__":
    maxDepth = 10  # depth in Mm
    minDepth = 1  # depth in Mm
    p0_ratio = 1  # ratio of initial pressure to the pressure at the top of the model S
    surfaceTemperature = 3500  # temperature in K FIXME ur not even using thiiiiiiiiis
    dlnP = 1e-2

    initialConditions = init.getBartaInit(p0_ratio, maxDepth, minDepth, dlnP=dlnP)

    finalT = 1  # final time in hours
    numberOfTSteps = 32  # number of time steps

    main(
        initialConditions,
        finalT=finalT,
        maxDepth=maxDepth,
        numberOfTSteps=numberOfTSteps,
        outputFolderName="FirstTestHaha",
    )
