from numpy import sqrt, exp, linspace
from pandas import read_csv
from scipy.constants import mu_0
from solvers import solve_T_eq, adjust_base_pressure, integrate_pressure, comp_mag
from OPAL.opacity import opac
from OPAL.EOS import gas_state
from mixing_length.fconv import fconv, frad
from initial_state import get_ini
from constants import *
from os.path import dirname, realpath, sep, pardir
import gc
import argparse

import matplotlib.pyplot as plt

import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

def B_bottom(P_e, P_i):
    return sqrt(2*mu_0*(P_e-P_i))

def plot_grid(mod, bcg_P, t, path):
    f, ax = plt.subplots(nrows=2, ncols=2)

    ax[0,0].plot(mod['z'], mod['T'])
    ax[0,0].set_ylabel('T')
    ax[0,0].set_xlabel('z')
    ax[0,1].plot(mod['z'], mod['fconv'])
    ax[0,1].set_ylabel('fconv')
    ax[0,1].set_xlabel('z')
    ax[1,0].plot(mod['z'], mod['y'])
    ax[1,0].set_ylabel('y')
    ax[1,0].set_xlabel('z')
    ax[1,1].plot(mod['z'], mod['P']-bcg_P)
    ax[1,1].set_ylabel('Pdiff')
    ax[1,1].set_xlabel('z')

    f.savefig(path+'{}.png'.format(t))
    print 'plot savet to:'
    print path+'{}.png'.format(t)

def main(upflow_velocity, tau, ini):
    t = 0.
    tau = tau

#    if t==0:
#        mod = read_csv('data/static/initial_state_T0_8319.dat', delimiter='\t')
#    else:
#        mod = read_csv('data/simulation/casovy_vyvoj_{}_sekund.csv'.format(t))
    mod = ini
    bcg_P = read_csv('data/static/bcg.dat', delimiter='\t', usecols=['P']).values.ravel()
    mod['opac'] = opac(mod['T'].values,mod['rho'].values)
    eos_results = gas_state(mod['T'].values,mod['P'].values)
    for col in eos_results.keys():
        mod[col] = eos_results[col]
    mod['fconv'] = fconv(0.3,mod)
    mod['frad'] = frad(mod)
    thinflux = (2*mu_0*(bcg_P-mod['P'].values))**0.25
    x = linspace(0,1,501)
    diff = sqrt(B_top)-thinflux[0]
    thinflux_approx = thinflux+diff*exp(-10*x)
    thinflux_approx[-1] = thinflux[-1]
    mod['y'] = comp_mag(mod['P'].values, bcg_P, thinflux_approx)
    name_str = 'data/simulation_v0_{velocity:.0f}/casovy_vyvoj_0_sekund.csv'.\
            format(velocity=upflow_velocity, time=t)
    mod.to_csv(name_str)
    mod_old = mod.copy()
    while True==True:
        t += tau
        # tau = 10
        print 'time: ', t

        mod_fix = mod.copy()

        mod['T'] = solve_T_eq(mod, tau)
        # print 'adjusting base pressure'
        P_bottom = adjust_base_pressure(mod, mod_old, bcg_P, tau, upflow_velocity)
        # print 'pressure difference: ', bcg_P[-1]-P_bottom
        # print 'min/max T: ', mod['T'].min(), mod['T'].max()
        # print 'min/max rho: ', mod['rho'].min(), mod['rho'].max()
        # print 'now going to integrate pressure'
        mod['P'] = integrate_pressure(P_bottom, mod['T'].values, mod['z'].values)
        # print 'ODE for pressure solved'
        # print 'pressure sum: ', mod['P'].sum()

        thinflux = (2*mu_0*(bcg_P-mod['P'].values))**0.25
        x = linspace(0,1,501)
        combined_approx = (1-x)*mod['y'].values+x*thinflux
        mod['y'] = comp_mag(mod['P'].values, bcg_P, combined_approx)

        eos_results = gas_state(mod['T'].values,mod['P'].values)
        for col in eos_results.keys():
            mod[col] = eos_results[col]
        mod['opac'] = opac(mod['T'].values,mod['rho'].values)
        mod['fconv'] = fconv(0.3,mod)
        mod_old = mod_fix

        name_str = 'data/simulation_v0_{velocity:.0f}/casovy_vyvoj_{time:.0f}_sekund.csv'.\
            format(velocity=upflow_velocity, time=t)
        mod.to_csv(name_str)

        path = 'data/simulation_v0_{velocity:.0f}/'.\
            format(velocity=upflow_velocity)
        plot_grid(mod, bcg_P, t, path)

        # for var, obj in locals().items():
        #     print var, sys.getsizeof(obj)
        gc.collect()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v0', type=float, default=100., dest='v0')
    parser.add_argument('-dt', type=float, default=10., dest='dt')
    parser.add_argument('-p0', type=float, default=0.2, dest='p0')
    args = parser.parse_args()
    
    ini = get_ini(args.p0)
    main(args.v0, args.dt, ini)
