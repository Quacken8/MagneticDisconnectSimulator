#!/usr/bin/env python3
import numpy as np
import constants as c
import warnings
from gravity import massBelowZ

class MockupIdealGas():
    @np.vectorize
    @staticmethod
    def pressure(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """
        returns density according to ideal gas law
        """
        warnings.warn("Ur using mockup ideal gas")

        return density*temperature/c.meanMolecularWeight*c.BoltzmannConstant
    
    @np.vectorize
    @staticmethod
    def entropy(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def cp(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def cv(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def delta(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        return 0
    
    @np.vectorize
    @staticmethod
    def F_rad(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        return 0
    
    @np.vectorize
    @staticmethod
    def F_con(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        return 0

    @np.vectorize
    @staticmethod
    def convectiveLogGradient(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError()
        return 0

    @np.vectorize
    @staticmethod
    def adiabaticLogGradient(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        return 0
    
    @np.vectorize
    @staticmethod
    def radiativeLogGradient(temperature: float | np.ndarray, density: float | np.ndarray, z: float | np.ndarray) -> float | np.ndarray:
        P = MockupIdealGas.pressure(temperature=temperature, density=density)
        luminostiy = c.solarLuminosity
        kappa = opacity
        Mr = massBelowZ(z)
        return 3*kappa*P*luminostiy/(16*np.pi*a*c.speedOfLight*c.G*Mr*temperature**4)

class IdealGas():
    ## TODO it looks like most of these are useless? i mean the original code saves them but doesnt seem to use things like entropy or the weird delta
    @np.vectorize
    @staticmethod
    def pressure(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """
        returns density according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")

        return density/temperature*c.meanMolecularWeight/c.BoltzmannConstant


    @np.vectorize
    @staticmethod
    def convectiveLogGradient(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()
    
    @np.vectorize
    @staticmethod
    def entropy(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()
    
    @np.vectorize
    @staticmethod
    def adiabaticLogGradient(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()
    
    @np.vectorize
    @staticmethod
    def cp(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()
    
    @np.vectorize
    @staticmethod
    def cv(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def delta(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def F_rad(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def F_con(temperature: float | np.ndarray, density: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()
    
