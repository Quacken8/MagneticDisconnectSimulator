#!/usr/bin/env python3
import numpy as np
import constants as c
import warnings
from gravity import massBelowZ
from scipy.optimize import fsolve as scipyFSolve


class MockupIdealGas:
    @np.vectorize
    @staticmethod
    def pressure(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        """
        returns density according to ideal gas law
        """
        warnings.warn("Ur using mockup ideal gas")

        return density * temperature / c.meanMolecularWeight * c.BoltzmannConstant

    @np.vectorize
    @staticmethod
    def entropy(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def cp(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def cv(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def delta(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def F_rad(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def F_con(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def convectiveLogGradient(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        raise NotImplementedError()
        return 0

    @np.vectorize
    @staticmethod
    def adiabaticLogGradient(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def radiativeLogGradient(
        temperature: float | np.ndarray,
        density: float | np.ndarray,
        z: float | np.ndarray,
    ) -> float | np.ndarray:
        P = MockupIdealGas.pressure(temperature=temperature, density=density)
        luminostiy = c.solarLuminosity
        kappa = opacity
        Mr = massBelowZ(z)
        return (
            3
            * kappa
            * P
            * luminostiy
            / (16 * np.pi * a * c.speedOfLight * c.G * Mr * temperature**4)
        )


class IdealGas:
    ## TODO it looks like most of these are useless? i mean the original code saves them but doesnt seem to use things like entropy or the weird delta
    @np.vectorize
    @staticmethod
    def pressure(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        """
        returns density according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")

        return density / temperature * c.meanMolecularWeight / c.BoltzmannConstant

    @np.vectorize
    @staticmethod
    def convectiveLogGradient(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def entropy(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def adiabaticLogGradient(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        according to Denizer 1965 eq 10
        """
        x = IdealGas.degreeOfIonization(temperature, density)
        chi = c.ionizationEnergyOfHydrogen
        warnings.warn("Ur using ideal gas")
        toReturn = (2+x*(1-x)*(2.5+chi/(c.BoltzmannConstant*temperature)))/(5+x*(1-x)*(2.5+chi/(c.BoltzmannConstant*temperature))*(2.5+chi/(c.BoltzmannConstant*temperature)))
        return toReturn
    
    @staticmethod
    def meanMolecularWeight(temperature: float | np.ndarray, density: float | np.ndarray)-> float | np.ndarray:
        x = IdealGas.degreeOfIonization(temperature, density)
        toReturn = c.meanMolecularWeight/(1+x)
        return toReturn

    @np.vectorize
    @staticmethod
    def radiativeLogGradient(
        temperature: float | np.ndarray, density: float | np.ndarray, gravitationalAcceleration: float | np.ndarray
    ) -> float | np.ndarray:   
        """returns radiative log gradient according to denizer 1965"""
        warnings.warn("Ur using ideal gas")
        meanMolecularWeight = IdealGas.meanMolecularWeight(temperature, density)
        H = IdealGas.pressureScaleHeight(temperature, gravitationalAcceleration, meanMolecularWeight)
        toReturn = 3/16 * opacity*density*H/(c.SteffanBoltzmann*temperature*temperature*temperature*temperature)
        return toReturn

    @np.vectorize
    @staticmethod
    def cp(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        """
        returns c_p according to denizer 1965
        """
        warnings.warn("Ur using ideal gas")
        mu = IdealGas.meanMolecularWeight(temperature, density)
        x = IdealGas.degreeOfIonization(temperature, density)
        chi = c.ionizationEnergyOfHydrogen
        toReturn = c.gasConstant/mu * (2.5 +0.5*x*(1-x)*(2.5+chi/(c.BoltzmannConstant*temperature))*(2.5+chi/(c.BoltzmannConstant*temperature)))
        return toReturn
    
    @staticmethod
    def degreeOfIonization(temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
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
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def delta(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @staticmethod
    def pressureScaleHeight(temperature: float | np.ndarray, gravitationalAcceleration: float | np.ndarray, meanMolecularWeight: float|np.ndarray) -> float | np.ndarray:
        """
        pressure scale height according to denizer 1965 eq 8
        """
        return c.gasConstant*temperature/(meanMolecularWeight*gravitationalAcceleration)


def F_rad(
    temperature: float | np.ndarray, density: float | np.ndarray
) -> float | np.ndarray:
    """
    returns convectiveGradient according to ideal gas law
    """
    # TODO equation 9 from rempel schussler
    raise NotImplementedError()

def F_con(
    temperature: float | np.ndarray, density: float | np.ndarray, mu: float | np.ndarray, cp: float | np.ndarray, adiabaticGrad: float | np.ndarray, radiativeGrad: float | np.ndarray
) -> float | np.ndarray:
    """
    returns convectiveGradient according to ideal gas law
    """

    realGradient = np.minimum(radiativeGrad, adiabaticGrad) # TODO check

    # these are parameters of convection used in Schüssler & Rempel 2005
    a = 0.125   # TODO maybe precalculating these could be useful?
    b = 0.5
    f = 1.5

    mixingLengthParam = l/Hp
    T3 = temperature*temperature*temperature

    u = 1/(f*np.sqrt(a)*mixingLengthParam*mixingLengthParam)*12*c.SteffanBoltzmann*T3/(c_p*density*kappa*Hp*Hp)*np.sqrt(Hp/g)
    gradTick = adiabaticGrad-2*u*u+2*u*np.sqrt(realGradient-adiabaticGrad+u*u)
    differenceOfGradients = realGradient-gradTick
    
    toReturn = -b*np.sqrt(a*c.gasConstant*l*/(mu*Hp))*density*c_p*np.pow(temperature*differenceOfGradients,1.5)
    # TODO equation 10 from rempel schussler
    raise NotImplementedError()
    return toReturn
