#!/usr/bin/env python3
import numpy as np
from constants import *
import warnings

class IdealGas():

    @np.vectorize
    @staticmethod
    def density(temperature: float | np.ndarray, pressure: float | np.ndarray) -> float | np.ndarray:
        """
        returns density according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError("What's the mean molecural weight u donkey, huh?")
        return pressure/temperature*c.MeanMolecularWeight/c.BoltzmannConstant


    @np.vectorize
    @staticmethod
    def convectiveGradient(temperature: float | np.ndarray, pressure: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()
    
    @np.vectorize
    @staticmethod
    def entropy(temperature: float | np.ndarray, pressure: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()
    
    @np.vectorize
    @staticmethod
    def adiabaticLogGradient(temperature: float | np.ndarray, pressure: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()
    
    @np.vectorize
    @staticmethod
    def cp(temperature: float | np.ndarray, pressure: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()
    
    @np.vectorize
    @staticmethod
    def cv(temperature: float | np.ndarray, pressure: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def delta(temperature: float | np.ndarray, pressure: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def F_rad(temperature: float | np.ndarray, pressure: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()

    @np.vectorize
    @staticmethod
    def F_con(temperature: float | np.ndarray, pressure: float | np.ndarray) -> float | np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        warnings.warn("Ur using ideal gas")
        raise NotImplementedError()
    
    

