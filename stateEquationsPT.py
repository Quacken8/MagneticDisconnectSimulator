#!/usr/bin/env python3
import numpy as np
import constants as c
import abc
import dataclasses
from mesa2Py import eosFromMesa
import loggingConfig
import logging

L = loggingConfig.configureLogging(logging.INFO, __name__)


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
    def actualGradient(
        temperature: np.ndarray,
        pressure: np.ndarray,
        massBelowZ: np.ndarray,
        opacity: np.ndarray,
    ) -> np.ndarray:
        """
        returns actual gradient
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def meanMolecularWeight(
        temperature: np.ndarray, pressure: np.ndarray
    ) -> np.ndarray:
        "returns mean molecular weight"

    @staticmethod
    @abc.abstractmethod
    def Cp(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
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

    @staticmethod
    @abc.abstractmethod
    def f_con(
        temperature: np.ndarray,
        pressure: np.ndarray,
        convectiveAlpha: float,
        opacity: np.ndarray,
        massBelowZ: np.ndarray,
        gravitationalAcceleration: np.ndarray,
    ) -> np.ndarray:
        """
        returns convective flux
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def f_rad(
        temperature: np.ndarray,
        pressure: np.ndarray,
        opacity: np.ndarray,
        Tgrad: np.ndarray,
    ) -> np.ndarray:
        """
        returns radiative flux
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
    def Cp(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def actualGradient(
        temperature: np.ndarray,
        pressure: np.ndarray,
        massBelowZ: np.ndarray,
        opacity: np.ndarray,
    ) -> np.ndarray:
        """
        returns actual gradient
        """
        nablaRad = IdealGas.radiativeLogGradient(
            temperature, pressure, massBelowZ, opacity
        )
        nablaAd = IdealGas.adiabaticLogGradient(temperature, pressure)

        return np.minimum(nablaRad, nablaAd)


@dataclasses.dataclass
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

            self._cache[(temperature, pressure)] = eosFromMesa.getEosResult(
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

    @staticmethod
    def actualGradient(
        temperature: np.ndarray,
        pressure: np.ndarray,
        massBelowZ: np.ndarray,
        opacity: np.ndarray,
    ) -> np.ndarray:
        """
        returns actual gradient
        """
        nablaRad = MESAEOS.radiativeLogGradient(
            temperature, pressure, massBelowZ, opacity
        )
        nablaAd = MESAEOS.adiabaticLogGradient(temperature, pressure)

        return np.minimum(nablaRad, nablaAd)

    @np.vectorize
    @staticmethod
    def Cp(temperature: float, pressure: float) -> float:
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

    @staticmethod
    def f_con(
        temperature: np.ndarray,
        pressure: np.ndarray,
        convectiveAlpha: float,
        opacity: np.ndarray,
        massBelowZ: np.ndarray,
        gravitationalAcceleration: np.ndarray,
    ) -> np.ndarray:
        actalGradient = MESAEOS.actualGradient(
            temperature=temperature,
            pressure=pressure,
            massBelowZ=massBelowZ,
            opacity=opacity,
        )
        adiabaticGradient = MESAEOS.adiabaticLogGradient(
            temperature=temperature, pressure=pressure
        )
        c_p = MESAEOS.Cp(temperature, pressure)
        rho = MESAEOS.density(temperature, pressure)
        mu = MESAEOS.meanMolecularWeight(temperature, pressure)
        Hp = MESAEOS.pressureScaleHeight(
            temperature=temperature,
            pressure=pressure,
            gravitationalAcceleration=gravitationalAcceleration,
        )

        toReturn = _f_con(
            convectiveAlpha=convectiveAlpha,
            temperature=temperature,
            density=rho,
            meanMolecularWeight=mu,
            c_p=c_p,
            actualGradient=actalGradient,
            adiabaticGradient=adiabaticGradient,
            Hp=Hp,
            gravitationalAcceleration=gravitationalAcceleration,
            opacity=opacity,
        )
        return toReturn

    @staticmethod
    def f_rad(
        temperature: np.ndarray,
        pressure: np.ndarray,
        opacity: np.ndarray,
        dTdz: np.ndarray,
    ) -> np.ndarray:
        """
        returns radiative flux


        """
        rho = MESAEOS.density(temperature, pressure)
        toReturn = _f_rad(
            temperature=temperature,
            opacity=opacity,
            dTdr=-dTdz,
            density=rho,
        )

        return toReturn


def _f_rad(
    temperature: np.ndarray,
    opacity: np.ndarray,
    dTdr: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """
    returns radiative flux based on state variables; used as a template for f_rad in other state equations
    """
    toReturn = (
        -16
        * c.SteffanBoltzmann
        * temperature
        * temperature
        * temperature
        / (3 * opacity * density)
        * dTdr
    )
    return toReturn


def _f_con(
    convectiveAlpha: float,
    temperature: np.ndarray,
    density: np.ndarray,
    meanMolecularWeight: np.ndarray,
    c_p: np.ndarray,
    actualGradient: np.ndarray,
    adiabaticGradient: np.ndarray,
    Hp: np.ndarray,
    gravitationalAcceleration: np.ndarray,
    opacity: np.ndarray,
) -> np.ndarray:
    """
    returns convectiveGradient based on state variables; used as a template for f_con in other state equations
    """

    # these are geometric parameters of convection used in Schüssler & Rempel 2005
    a = 0.125
    b = 0.5
    f = 1.5

    u = (
        1
        / (f * np.sqrt(a))
        * convectiveAlpha
        * convectiveAlpha
        * 12
        * c.SteffanBoltzmann
        * temperature
        * temperature
        * temperature
        / (c_p * density * opacity * Hp * Hp)
        * np.sqrt(Hp / gravitationalAcceleration)
    )
    # this is sometimes called ∇' (schussler & rempel 2005) or ∇_e (Lattanzio in thier class M4111 of 2009) where e stands for element of stellar matter
    # it, according to Schüssler & Rempel 2005, "reflects radiative energy exchange of the convective parcels"
    gradTick = (
        adiabaticGradient
        - 2 * u * u
        + 2 * u * np.sqrt(np.maximum(actualGradient - adiabaticGradient + u * u, 0))
    )
    # TODO these np.maximums probably shouldnt be there..? but we need them, getting negative values happens
    differenceOfGradients = np.maximum(actualGradient - gradTick, 0)
    toReturn = (
        -b
        * np.sqrt(a * c.gasConstant * convectiveAlpha / meanMolecularWeight)
        * density
        * c_p
        * np.power(temperature * differenceOfGradients, 1.5)
    )
    return toReturn
