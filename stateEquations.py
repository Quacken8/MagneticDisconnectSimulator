#!/usr/bin/env python3
import numpy as np
import constants as c
import warnings
from gravity import massBelowZ


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
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def cp(
        temperature: float | np.ndarray, density: float | np.ndarray
    ) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

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
