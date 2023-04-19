#!/usr/bin/env python3
"""
fundamental constants and some used units for the code with their units and source in the comments
"""
hour = 60 * 60  # seconds in an hour
Mm = 1e6  # meters in megameter Mm
cm = 1e-2  # meters in a centimeter
gram = 1e-3  # kilgrams in a gram
Gauss = 1e-4  # Teslas in one Gauss
barye = 1e1  # Pascals in one barye
erg = 1e1  # ergs in Joule
eV = 1.602176634e-19  # Joules in one eV [CODATA 2018]
BoltzmannConstant = 1.380649e-23  # J/K [CODATA 2018]
gasConstant = 8.314462618  # J/K/mol [CODATA 2018]
AH = 1.0078250322  # atomic weight of hydrogen atom [CIAAW 2000]
AHe = 4.002_603_2545  # atomic weight of helium atom [CIAAW 2000]
massFractionOfHydrogen = 0.6  # of solar matter
massFractionOfHelium = 0.4  # of solar matter
meanMolecularWeight = gram / (
    massFractionOfHydrogen / AH + massFractionOfHelium / AHe
)  # kg/mol of solar matter
mu0 = 1.25663706212e-6  # N/A^2 permeability of vacuum [CODATA 2018]
G = 6.67430e-11  # m^3/(kg s^2) gravitational constant [CODATA 2018]
solarLuminosity = 3.828e26  # W [IAU 2015]
speedOfLight = 299792458  # m/s [CODATA 2018]
SteffanBoltzmann = 5.670374419e-8  # W/(m^2 K^4) [CODATA 2018]
ionizationEnergyOfHydrogen = (
    13.59844 * eV
)  # TODO get better source [Lide, 1992, Ionization potentials of atoms and atomic ions]
