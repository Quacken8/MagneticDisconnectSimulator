#!/usr/bin/env python3

import numpy as np
from initialConditionsSetterUpper import getInitialConditions
from dataVisualisator import visualizeData
from dataStructure import Data, SingleTimeDatapoint
import solvers
from constants import *

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

def main(initialConditions, finalT = 100, dt = 1e-1, maxDepth = 100, dz = 1e-2, outputFolderName = "output"):
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
    finalT *= Chour
    dt *= Chour
    maxDepth *= CMm
    dz *= CMm

    # create empty data structure with only initial conditions
    data = Data(finalT=finalT, dt=dt, maxDepth=maxDepth, dz=dz)
    numberOfTSteps = data.numberOfTSteps
    data.addDatapointAtIndex(initialConditions, 0)

    ti = 0 # time index, not actual time
    while (ti < numberOfTSteps): # hey, did you know that in python for cycles are just while cycles that interanlly increment their made up indeces?

        currentState = data.values[ti]

        newTs = solvers.getNewTs(currentState, dt)
        newPs = solvers.getNewPs()
        newYs = solvers.getNewYs()
        newB_0s = newYs*newYs
        newF_cons = solvers.getNewF_cons()
        newF_rads = solvers.getNewF_rads()


        newDatapoint = SingleTimeDatapoint(temperatures = newTs, pressures=newPs, B_0s=newB_0s, F_cons=newF_cons, F_rads=newF_rads)

        ti += 1
        data.addDatapointAtIndex(newDatapoint, ti)

    data.saveToFolder(outputFolderName)
    visualizeData(data)



if __name__ == "__main__":
    initialConditions = getInitialConditions()
    dt = 1e-3 # timestep in TBD units
    finalT = 100 # final time in TBD units

    maxDepth = 100 # depth in Mm
    dz = 1e-2   # step in depth in Mm

    main(initialConditions, finalT=finalT, dt=dt, maxDepth = maxDepth, dz = dz)