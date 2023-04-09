#!/usr/bin/env python3
import numpy as np
import constants as c
import warnings
from gravity import massBelowZ
from scipy.optimize import fsolve as scipyFSolve
warnings.warn("Ur using the model S opacity here")
from opacity import modelSNearestOpacity as opacity

class IdealGas:
    ## TODO it looks like most of these are useless? i mean the original code saves them but doesnt seem to use things like entropy or the weird delta
    @np.vectorize
    @staticmethod
    def density(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """
        returns density according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        #TODO -  is this real?
        mu = IdealGas.meanMolecularWeight(temperature, pressure)
        return mu *pressure / (temperature * c.BoltzmannConstant)

    @np.vectorize
    @staticmethod
    def convectiveLogGradient(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def entropy(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def adiabaticLogGradient(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        according to Denizer 1965 eq 10
        """
        x = IdealGas.degreeOfIonization(temperature, pressure)
        chi = c.ionizationEnergyOfHydrogen
        warnings.warn("Ur using ideal gas")
        toReturn = (2+x*(1-x)*(2.5+chi/(c.BoltzmannConstant*temperature)))/(5+x*(1-x)*(2.5+chi/(c.BoltzmannConstant*temperature))*(2.5+chi/(c.BoltzmannConstant*temperature)))
        return toReturn
    
    @staticmethod
    def meanMolecularWeight(temperature: np.ndarray, pressure: np.ndarray)-> np.ndarray:
        x = IdealGas.degreeOfIonization(temperature, pressure)
        toReturn = c.meanMolecularWeight/(1+x)
        return toReturn

    @staticmethod
    def radiativeLogGradient(
        temperature: np.ndarray, pressure: np.ndarray, gravitationalAcceleration: np.ndarray
    ) -> np.ndarray:   
        """returns radiative log gradient according to denizer 1965"""
        warnings.warn("Ur using ideal gas")
        density = IdealGas.density(temperature, pressure)
        H = IdealGas.pressureScaleHeight(temperature= temperature, pressure=pressure , gravitationalAcceleration=gravitationalAcceleration)
        toReturn = 3/16 * opacity(temperature, pressure) * density*H/(c.SteffanBoltzmann*temperature*temperature*temperature*temperature)
        return toReturn

    @staticmethod
    def cp(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """
        returns c_p according to denizer 1965
        """
        warnings.warn("Ur using ideal gas")
        mu = IdealGas.meanMolecularWeight(temperature, pressure)
        x = IdealGas.degreeOfIonization(temperature, pressure)
        chi = c.ionizationEnergyOfHydrogen
        toReturn = c.gasConstant/mu * (2.5 +0.5*x*(1-x)*(2.5+chi/(c.BoltzmannConstant*temperature))*(2.5+chi/(c.BoltzmannConstant*temperature)))
        return toReturn
    
    @staticmethod
    def degreeOfIonization(temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """
        returns degree of ionization according to denizer 1965
        """
        warnings.warn("Ur using ideal gas")

        rootFinderConstantPart = -2.5 * np.log10(temperature) + (13.53*5040)/temperature +0.48+np.log10(c.massFractionOfHydrogen) + np.log10(pressure) + np.log10(c.meanMolecularWeight)

        def toFindRoot(x):
            return np.log10(x*x/(1-x*x)) + rootFinderConstantPart
        
        sizeOfInput = np.array(temperature).size
        startingEstimate = 4.07e-4 * np.ones(sizeOfInput) # just some estimate based on ionization of photosphere (Gingerich Jager 1967)
        result = scipyFSolve(toFindRoot, startingEstimate)

        if result[2] != 1: raise RuntimeError("Search for degree of ionization wasnt succesful")
        toReturn = result[0]
        return toReturn



    @np.vectorize
    @staticmethod
    def cv(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def delta(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @staticmethod
    def pressureScaleHeight(temperature: np.ndarray, pressure: np.ndarray, gravitationalAcceleration: np.ndarray) -> np.ndarray:
        """
        pressure scale height according to denizer 1965 eq 8
        """
        mu = IdealGas.meanMolecularWeight(temperature, pressure)
        return c.gasConstant*temperature/(mu*gravitationalAcceleration)


def F_rad(
    temperature: np.ndarray, pressure: np.ndarray
) -> np.ndarray:
    """
    returns convectiveGradient according to ideal gas law
    """
    # TODO equation 9 from rempel schussler
    raise NotImplementedError()

def F_con(
    convectiveLength: float, temperature: np.ndarray, pressure: np.ndarray, meanMolecularWeight: np.ndarray, adiabaticGrad: np.ndarray, radiativeGrad: np.ndarray, pressureScaleHeight: np.ndarray, c_p: np.ndarray, gravitationalAcceleration: np.ndarray
) -> np.ndarray:
    """
    returns convectiveGradient according to ideal gas law
    """

    realGradient = np.minimum(radiativeGrad, adiabaticGrad) # TODO check

    kappa = opacity(temperature, pressure)

    # these are parameters of convection used in Schüssler & Rempel 2005
    a = 0.125   # TODO maybe precalculating these could be useful?
    b = 0.5
    f = 1.5

    # just some renaming for the sake of readibilty
    g = gravitationalAcceleration
    mu = meanMolecularWeight
    Hp = pressureScaleHeight
    lH = convectiveLength/Hp
    T3 = temperature*temperature*temperature
    density = IdealGas.density(temperature, pressure)

    u = 1/(f*np.sqrt(a)*lH*lH)*12*c.SteffanBoltzmann*T3/(c_p*density*kappa*Hp*Hp)*np.sqrt(Hp/g)
    gradTick = adiabaticGrad-2*u*u+2*u*np.sqrt(realGradient-adiabaticGrad+u*u)
    differenceOfGradients = realGradient-gradTick
    
    toReturn = -b*np.sqrt(a*c.gasConstant*lH/mu)*density*c_p*np.power(temperature*differenceOfGradients,1.5)
    # TODO equation 10 from rempel schussler
    raise NotImplementedError()
    return toReturn
