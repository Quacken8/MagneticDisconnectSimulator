# This file calls the FORTRAN interpolation code
# and returns density, specific heat at constant
# pressure, adiabatic gradient and partial derivative
# of pressure by temperature; all in SI units as
# a function of T and P

from __future__ import division
from .EOSdata import eos
from numpy import zeros_like, atleast_1d

def gas_state(T, P, X=0.7, ztab=0.02, irad=0):
    T, P = atleast_1d(T, P)
    T = T.astype(float)
    P = P.astype(float)
    rho = zeros_like(T)
    chiT = zeros_like(T)
    chiRho = zeros_like(T)
    E = zeros_like(T)
    cv = zeros_like(T)
    nabla_ad_inv = zeros_like(T)
    gamma1 = zeros_like(T)
    S = zeros_like(T)
    for i in range(len(rho)):
        rho[i] = eos.rhoofp(X, ztab, T[i] * 1e-6, P[i] * 1e-11, irad)
        try:
            reteos = eos.esac(X, ztab, T[i] * 1e-6, rho[i], 9, irad)
            P_dump, E[i], S[i], cv[i], chiRho[i], chiT[i], gamma1[i], nabla_ad_inv[i] = reteos[:8]
        except:
            print T[i], P[i], rho[i]
        # chiRho ... dlogP/dlogRho at constant T
        # chiT ... dlogP/dlogT at constant rho
    rho = rho * 1e3
    E = E * 1e-8  # energy density [J/kg]
    S = S * 1e2  # entrophy [J/K]
    cv = cv * 1e2  # specific heat at constant V [J/(K.kg)]
    nabla_ad = 1. / nabla_ad_inv  # adiabatic gradient
    delta = chiT / chiRho
    cp = cv + P/(T*rho)*chiT**2/chiRho
    return {'rho': rho, 'cp': cp, 'cv': cv,
            'nabla_ad': nabla_ad, 'delta': delta, 'S': S}  # chybi tlakova skala
