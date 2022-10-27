from numpy import sqrt, exp, linspace
from pandas import read_csv
from scipy.constants import mu_0
from solvers import solve_T_eq, adjust_base_pressure, integrate_pressure, comp_mag
from OPAL.opacity import opac
from OPAL.EOS import gas_state
from mixing_length.fconv import fconv, frad
from constants import *
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

bcg_P = read_csv('data/static/bcg.dat', delimiter='\t', usecols=['P']).values.ravel()
mod_old = read_csv('data/simulation/casovy_vyvoj_600.0_sekund.csv')
mod = read_csv('data/simulation/problematic.csv')
tau = 100

gas_state(3500, 17000)['rho']

# [ 3500.14916961] 17981.0252613
# integrate_pressure(mod['P'].iloc[-1], mod['T'].values, mod['z'].values);
# adjust_base_pressure(mod, mod_old, bcg_P, tau);