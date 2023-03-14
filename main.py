#!/usr/bin/env python3

import numpy as np
from initialConditionsSetterUpper import getInitialConditions
from dataVizualizer import visualizeData
from dataStructure import Data, SingleTimeDatapoint
import constants as c
from calmSun import getCalmSunDatapoint

# Choise of solvers:
from solvers import firstOrderTSolver as getNewTs
from solvers import oldYSolver as getNewYs

# EVERYTHING IN SI!!!!

# ok, here's the deal
# we solving the differential equation based on the paper Schüssler Rempel 2018
# Phi/2pi y d^2y/dz^2 = y^4 - 8 pi (p_e -p_i)
# y = sqrt(B_0(z))
# at the bottom they always assume that B = sqrt(8 pi (p_e - p_i))

# they in particular choose some weird iteration btw
# p_e is fixed external pressure far away from the sunspot
# phi is ughhhh idk? is it fixed? cant be TBD!!

# p_i comes from hydrodynamic equilibrium dp/dz = g rho
# rho has to be optained from state eq? TBD

# we also solve temperature from energy transport
# for that we need opacity and some thingies for convection which also use opacity and rho

# each timestep there is mass flowing in at the bottom, that changes the pressure
# it needs to be adjusted using eq 15 (solved in the paper via newton method)
# velocity of the inflow which is in that eq also caculated form elsewhere, here the eq 17
# after adjustment it has to be updated (i e integrated) throughout the whole tube


def main(initialConditions, finalT=100, numberOfTSteps = 2**4, maxDepth=100, outputFolderName="output"):
    """
    integrates the whole thing
    ----
    PARAMETERS
    ----
    finalT, double: total length of the simulation in hours
    dt, double: length of the time step in hours
    maxDepth, double: total depth of the simulation in Mm
    dz, double: step in the depth direction in Mm
    """

    # turn user input to SI
    finalT *= c.hour
    maxDepth *= c.Mm

    # get the calm sun model
    calmSun = getCalmSunDatapoint()

    # create empty data structure with only initial conditions
    data = Data(finalT=finalT, numberOfTSteps=numberOfTSteps)
    dt = data.dt

    numberOfTSteps = data.numberOfTSteps
    data.addDatapointAtIndex(initialConditions, 0)

    timeIndex = 0 # time index, not actual time
    while (timeIndex < numberOfTSteps): # hey, did you know that in python for cycles are just while cycles that interanlly increment their made up indeces?

        currentState = data.values[timeIndex]

        newTs = getNewTs(currentState, dt)
        newPs = getNewPs(currentState, dt, upflowVelocity, totalMagneticFlux)
        newYs = getNewYs()
        newB_0s = newYs*newYs
        newF_cons = solvers.getNewF_cons()
        newF_rads = solvers.getNewF_rads()


        newDatapoint = SingleTimeDatapoint(temperatures = newTs, pressures=newPs, B_0s=newB_0s, F_cons=newF_cons, F_rads=newF_rads)

        timeIndex += 1
        data.addDatapointAtIndex(newDatapoint, timeIndex)

    data.saveToFolder(outputFolderName)
    visualizeData(data)



if __name__ == "__main__":

    maxDepth = 100  # depth in Mm
    numberOfZSteps = 100

    initialConditions = getInitialConditions(maxDepth = maxDepth, numberOfZSteps=numberOfZSteps)

    finalT = 100 # final time in hours
    numberOfTSteps = 32 # number of time steps

    main(initialConditions, finalT=finalT, maxDepth=maxDepth,
         numberOfTSteps=numberOfTSteps, outputFolderName="FirstTestHaha")
