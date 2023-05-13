#!/usr/bin/env python3
import numpy as np
import constants as c
import abc
from dataclasses import dataclass
from mesa2Py.eosFromMesa import getEosResult
import logging

L = logging.getLogger(__name__)


class StateEquationInterface(metaclass=abc.ABCMeta):
    """
    Interface for state equations classes
    """

    @staticmethod
    @abc.abstractmethod
    def density(
        temperature: float | np.ndarray, pressure: float | np.ndarray
    ) -> float | np.ndarray:
        """
        returns density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def convectiveLogGradient(
        temperature: np.ndarray,
        pressure: np.ndarray,
        opacity: np.ndarray,
        gravitationalAcceleration: np.ndarray,
        massBelowZ: np.ndarray,
        convectiveAlpha: float,
    ) -> np.ndarray:
        """
        returns convectiveGradient
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def adiabaticLogGradient(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """
        returns convectiveGradient
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def radiativeLogGradient(
        temperature: np.ndarray,
        pressure: np.ndarray,
        massBelowZ: np.ndarray,
        opacity: np.ndarray,
    ) -> np.ndarray:
        """returns radiative log gradient"""
        pass

    @staticmethod
    @abc.abstractmethod
    def meanMolecularWeight(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        "returns mean molecular weight"

    @staticmethod
    @abc.abstractmethod
    def cp(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """
        returns c_p
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def pressureScaleHeight(
        temperature: np.ndarray,
        pressure: np.ndarray,
        gravitationalAcceleration: np.ndarray,
    ) -> np.ndarray:
        """
        returns pressure scale height
        """
        pass


class IdealGas(StateEquationInterface):
    """
    State equations for ideal gas mostly based on Denizer 1965
    """

    @staticmethod
    def density(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """
        returns density according to ideal gas law
        """
        mu = IdealGas.meanMolecularWeight(temperature, pressure)
        return mu * pressure / (temperature * c.gasConstant)

    @staticmethod
    def convectiveLogGradient(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        """
        returns convectiveGradient according to ideal gas law
        """
        raise NotImplementedError()

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
        toReturn = (
            2 + x * (1 - x) * (2.5 + chi / (c.BoltzmannConstant * temperature))
        ) / (
            5
            + x
            * (1 - x)
            * (2.5 + chi / (c.BoltzmannConstant * temperature))
            * (2.5 + chi / (c.BoltzmannConstant * temperature))
        )
        return toReturn

    @staticmethod
    def meanMolecularWeight(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        x = IdealGas.degreeOfIonization(temperature, pressure)
        toReturn = c.meanMolecularWeight / (1 + x)
        return toReturn

    @staticmethod
    def radiativeLogGradient(
        temperature: np.ndarray,
        pressure: np.ndarray,
        massBelowZ: np.ndarray,
        opacity: np.ndarray,
    ) -> np.ndarray:
        """
        returns radiative log gradient according to Harmanec Broz 2011
        assumes constant luminosity, therefore is only applicable near Sun's surface
        """
        nablaRad = (
            3
            * opacity
            * pressure
            * c.L_sun
            / (
                16
                * np.pi
                * c.aRad
                * c.speedOfLight
                * c.G
                * massBelowZ
                * temperature
                * temperature
                * temperature
                * temperature
            )
        )

        return nablaRad

    @staticmethod
    def cp(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """
        returns c_p according to denizer 1965
        """
        mu = IdealGas.meanMolecularWeight(temperature, pressure)
        x = IdealGas.degreeOfIonization(temperature, pressure)
        chi = c.ionizationEnergyOfHydrogen
        toReturn = (
            c.gasConstant
            / mu
            * (
                2.5
                + 0.5
                * x
                * (1 - x)
                * (2.5 + chi / (c.BoltzmannConstant * temperature))
                * (2.5 + chi / (c.BoltzmannConstant * temperature))
            )
        )
        return toReturn

    @staticmethod
    def degreeOfIonization(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """
        returns degree of ionization according to denizer 1965, resp  Kippenhahn, Temesvary, and Biermann (1958)
        which states that log(x^2/(1-x^2)) + C(T, P) = 0
        solution to that equation is
        1/sqrt(1+10^C)
        """

        C = (
            -2.5 * np.log10(temperature)
            + (13.53 * 5040) / temperature
            + 0.48
            + np.log10(c.massFractionOfHydrogen)
            + np.log10(pressure * c.barye)
            + np.log10(c.meanMolecularWeight * c.gram)
        )

        tenPower = np.float_power(10, C)
        toReturn = 1 / np.sqrt(1 + tenPower)
        return toReturn

    @staticmethod
    def pressureScaleHeight(
        temperature: np.ndarray,
        pressure: np.ndarray,
        gravitationalAcceleration: np.ndarray,
    ) -> np.ndarray:
        """
        pressure scale height according to denizer 1965 eq 8
        """
        rho = IdealGas.density(temperature, pressure)
        return pressure / (rho * gravitationalAcceleration)


@dataclass
class MESACache:
    _cache = {}

    simpleSolarAbundances = {
        "h1": c.massFractionOfHydrogen,
        "he4": c.massFractionOfHelium,
    }

    def __init__(self):
        self._cache = {}

    def __getitem__(self, key):  # when you call MESACache[(temperature, pressure)]
        if key not in self._cache.keys():  # not found in cache, call Fortran
            temperature, pressure = key
            self._cache[(temperature, pressure)] = getEosResult(
                temperature, pressure, massFractions=c.solarAbundances, cgsOutput=False
            )
        return self._cache[key]

    def clear(self):
        self._cache = {}


class MESAEOS(StateEquationInterface):
    """
    Interface for state equations classes
    """

    _cache = MESACache()

    @staticmethod
    def cacheClear():
        MESAEOS._cache = {}

    @np.vectorize
    @staticmethod
    def fullOutput(temperature: float, pressure: float):
        """
        returns full output of MESA EOS
        """
        return MESAEOS._cache[(temperature, pressure)]

    @np.vectorize
    @staticmethod
    def density(temperature: float, pressure: float) -> float:
        """
        returns density
        """
        rho = MESAEOS._cache[(temperature, pressure)].results.rho
        return rho
    
    @np.vectorize
    @staticmethod
    def entropy(temperature: float, pressure: float) -> float:
        """
        returns entropy
        """
        lnEntropy = MESAEOS._cache[(temperature, pressure)].results.lnS
        return np.exp(lnEntropy)

    @np.vectorize
    @staticmethod
    def convectiveLogGradient(
        temperature: float,
        pressure: float,
        convectiveAlpha: float,
        opacity: float,
        gravitationalAcceleration: float,
        massBelowZ: float,
    ) -> float:
        """
        returns convectiveGradient
        """

        density = MESAEOS.density(temperature, pressure)
        Hp = MESAEOS.pressureScaleHeight(
            temperature, pressure, gravitationalAcceleration=gravitationalAcceleration
        )
        T3 = temperature * temperature * temperature
        c_p = MESAEOS.cp(temperature, pressure)
        nablaAd = MESAEOS.adiabaticLogGradient(temperature, pressure)
        nablaRad = MESAEOS.radiativeLogGradient(
            temperature, pressure, massBelowZ, opacity
        )

        # these are parameters of convection used in Schüssler & Rempel 2005
        a = 0.125
        b = 0.5
        f = 1.5

        u = (
            1
            / (f * np.sqrt(a) * convectiveAlpha * convectiveAlpha)
            * 12
            * c.SteffanBoltzmann
            * T3
            / (c_p * density * opacity * Hp * Hp)
            * np.sqrt(Hp / gravitationalAcceleration)
        )

        nablaConv = (
            (nablaRad - nablaAd + 2 * u * u)
            * (nablaRad - nablaAd + 2 * u * u)
            / 4
            / u
            / u
            + nablaAd
            - u * u
        )

        return nablaConv

    @np.vectorize
    @staticmethod
    def adiabaticLogGradient(temperature: float, pressure: float) -> float:
        """
        returns convectiveGradient
        """
        nablaAd = MESAEOS._cache[(temperature, pressure)].results.grad_ad
        return nablaAd

    @np.vectorize
    @staticmethod
    def meanMolecularWeight(temperature: float, pressure: float) -> float:
        "returns mean molecular weight"
        mu = MESAEOS._cache[(temperature, pressure)].results.mu
        return mu

    @np.vectorize
    @staticmethod
    def radiativeLogGradient(
        temperature: float,
        pressure: float,
        massBelowZ: float,
        opacity: float,
    ) -> float:
        """
        returns radiative log gradient according to Harmanec Broz 2011
        assumes constant luminosity, therefore is only applicable near Sun's surface
        """

        nablaRad = (
            3
            * opacity
            * pressure
            * c.L_sun
            / (
                16
                * np.pi
                * c.aRad
                * c.speedOfLight
                * c.G
                * massBelowZ
                * temperature
                * temperature
                * temperature
                * temperature
            )
        )
        return nablaRad

    @np.vectorize
    @staticmethod
    def cp(temperature: float, pressure: float) -> float:
        """
        returns c_p
        """
        Cp = MESAEOS._cache[(temperature, pressure)].results.Cp
        return Cp

    @np.vectorize
    @staticmethod
    def pressureScaleHeight(
        temperature: float,
        pressure: float,
        gravitationalAcceleration: float,
    ) -> float:
        """
        returns pressure scale height
        """
        rho = MESAEOS.density(temperature, pressure)
        return pressure / (rho * gravitationalAcceleration)


def F_rad(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """
    returns radiative flux according to ideal gas law
    """
    # TODO equation 9 from rempel schussler
    raise NotImplementedError()


def F_con(
    convectiveAlpha: float,
    temperature: np.ndarray,
    pressure: np.ndarray,
    meanMolecularWeight: np.ndarray,
    adiabaticGrad: np.ndarray,
    radiativeGrad: np.ndarray,
    pressureScaleHeight: np.ndarray,
    c_p: np.ndarray,
    gravitationalAcceleration: np.ndarray,
    opacity: np.ndarray,
) -> np.ndarray:
    """
    returns convectiveGradient according to ideal gas law
    uses the unitless parameter convectiveAlpha (see Schüssler Rempel 2018)
    """

    realGradient = np.minimum(radiativeGrad, adiabaticGrad)

    # these are parameters of convection used in Schüssler & Rempel 2005
    a = 0.125
    b = 0.5
    f = 1.5

    # just some renaming for the sake of readibilty
    g = gravitationalAcceleration
    mu = meanMolecularWeight
    Hp = pressureScaleHeight
    T3 = temperature * temperature * temperature
    density = IdealGas.density(temperature, pressure)

    u = (
        1
        / (f * np.sqrt(a) * convectiveAlpha * convectiveAlpha)
        * 12
        * c.SteffanBoltzmann
        * T3
        / (c_p * density * opacity * Hp * Hp)
        * np.sqrt(Hp / g)
    )
    gradTick = (
        adiabaticGrad
        - 2 * u * u
        + 2 * u * np.sqrt(realGradient - adiabaticGrad + u * u)
    )
    differenceOfGradients = realGradient - gradTick

    toReturn = (
        -b
        * np.sqrt(a * c.gasConstant * convectiveAlpha / mu)
        * density
        * c_p
        * np.power(temperature * differenceOfGradients, 1.5)
    )
    # TODO equation 10 from rempel schussler
    raise NotImplementedError()
    return toReturn


zs, nablas = np.loadtxt("debuggingReferenceFromSvanda/actualNabla.dat", unpack=True)
from scipy.interpolate import interp1d

interpolatedNablas = interp1d(zs, nablas, kind="linear")


def interpolatedNabla(z):
    return interpolatedNablas(z)
