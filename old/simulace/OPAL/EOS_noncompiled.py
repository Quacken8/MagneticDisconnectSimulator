# This file calls the FORTRAN interpolation code
# and returns density, specific heat at constant
# pressure, adiabatic gradient and partial derivative
# of pressure by temperature; all in SI units as
# a function of T and P

from .EOSdata import eos

def gas_state(T, P, X=0.7, ztab=0.02, irad=0):
	rho = eos.rhoofp(X, ztab, T*1e-6, P*1e-11, irad)
	reteos = eos.esac(X, ztab, T*1e-6, rho, 9, irad)
	rho = rho*1e3
	P, E, S, cv, chiRho, chiT, gamma1, nabla_ad_inv = reteos[:8]
	E = E*1e-8 #energy density [J/kg]
	S = S*1e2 #entrophy [J/K]
	cv = cv*1e2 #specific heat at constant V [J/(K.kg)]
	#chiRho ... dlogP/dlogRho at constant T
	#chiT ... dlogP/dlogT at constant rho
	nabla_ad = 1./nabla_ad_inv #adiabatic gradient
	delta = chiT/chiRho
	cp = cv+P*1e11/(T*rho)*chiT**2/chiRho
	return {'rho': rho, 'cp': cp, 'cv': cv,
		'nabla_ad': nabla_ad, 'delta': delta, 'S': S} # chybi tlakova skala
