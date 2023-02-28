from scipy.constants import sigma, mu_0
from scipy.sparse import diags as sparse_diags
from scipy.sparse.linalg import spsolve, inv
from scipy.optimize import newton, bisect, brentq
from scipy.interpolate import UnivariateSpline
from scipy.integrate import ode, simps
from scipy.ndimage import gaussian_filter
from pandas import DataFrame
from numpy import concatenate, sqrt, linspace, ones, zeros, array, pi, exp, log, abs, max, interp
from numpy.linalg import norm
import sys
import time

from OPAL.EOS import gas_state
from model_s.interp import g
from constants import *
from copy import copy
import pdb

def integrate_pressure(P_bottom, T_model, z, ret_rho=False):
    Tfunc = UnivariateSpline(z, T_model, s=0, k=1, ext='const')

    def f(logP, z_):
        P = exp(logP)
        try:
            T = Tfunc(z_)
        except:
            pdb.set_trace()
        # T = interp(z_, z, T_model)
        # TODO uvest omezeni do prace
        state = gas_state(T, P)
        H = P/(state['rho']*g(z_))
        return H

    r = ode(f).set_integrator('vode')
    y0 = array([ 12.5e6 ])
    r.set_initial_value(y0, log(P_bottom))
    dlogP = 0.01

    model = []
    while r.successful() and r.y[0]>=-1e4 and len(model)<=2000:
        z_ = r.y[0]
        logP = r.t
        P = exp(logP)

        if ret_rho==True:
            try:
                T = Tfunc(z_)
            except:
                pdb.set_trace()
            # T = interp(z_, z, T_model)
            rho = gas_state(T, P)['rho'][0]
            new_state = { 'z': z_, 'P': P, 'rho': rho }
            model.append(new_state)
        elif ret_rho==False:
            model.append( {'z': z_, 'P': P})
        else:
            sys.exit('ret_rho values must be boolean')

        r.integrate(logP-dlogP)

    model = DataFrame(model)
    Pfunc = UnivariateSpline(model['z'].values[::-1], model['P'].values[::-1], s=0, k=1, ext='const')
    if ret_rho==True:
        rhofunc = UnivariateSpline(model['z'][::-1], model['rho'][::-1], s=0, k=1, ext='const')
        return Pfunc(z), rhofunc(z)
    else:
        return Pfunc(z)


def solve_T_eq(mod, tau):
    z = mod['z'].values
    T = mod['T'].values
    T_out_down = 3500.
    T_out_up = 2*T[-1]-T[-2]
    P = mod['P'].values
    opac= mod['opac'].values
    rho = mod['rho'].values
    fconv = mod['fconv'].values
    cp = mod['cp'].values
    nabla_ad_bottom = mod['nabla_ad'].values[-1]
    length = z.size

    h = 25000.	#grid spacing

    B = 16*sigma*T**3/(3*opac*rho)
    # nize je derivace B pomoci centralnich diferenci
    # krajni body A jsou aproximovany jednostrannymi diferencemi
    A = concatenate( (
        [(B[1]-B[0])/h],
        (B[2:]-B[:-2])/(2*h),
        [(B[-1]-B[-2])/h]) )
    # H...tlakova skala na spodni hranici
    # potrebna pro okrajovou podminku
    H = P[-1]/(rho[-1]*g(z[-1]))

    # definice hlavni diagonaly a vedlejsich diagonal
    d_main = -2*B/h**2-rho*cp/tau
    #d_main[-1] = 1/h
    #d_main[0] = 1

    d_sub = -0.5*A/h+B/h**2
    b_out_down = d_sub[0]
    d_sub = d_sub[1:]
    #d_sub[-1] = -1/h

    d_sup = 0.5*A/h+B/h**2
    b_out_up = d_sup[-1]
    d_sup = d_sup[:-1]
    #d_sup[0] = 0

    # vektor prave strany:
    # derivace F_conv a zbytek
    b = concatenate( (
        #[3500.],
        [(fconv[1]-fconv[0])/h-(rho[0]*cp[0]*T[0]/tau)-b_out_down*T_out_down],
        (fconv[2:]-fconv[:-2])/(2*h)-(rho*cp*T/tau)[1:-1],
        [(fconv[-1]-fconv[-2])/h-(rho[-1]*cp[-1]*T[-1]/tau)-b_out_up*T_out_up] ) )

    data = [d_sub, d_main, d_sup]
    diags = [-1,0,1]

    M = sparse_diags(data, diags, shape=(length, length), format='csc')
    # TODO zkusit teplotu upravit tak, aby v ni nebyl hrbol
    return gaussian_filter(spsolve(M,b),0)

def A(y_,P,bcg_P):
    dP = bcg_P-P
    h = 25000

    eta = ( Phi/(pi)*(-h**-2) )*ones(len(y_))
    xi = Phi/(h**2*2*pi)*ones(len(y_))
    eta[0] = eta[-1] = 1
    xi_bottom = copy(xi[:-1])
    xi_bottom[-1] = 0
    xi_top = copy(xi[1:])
    xi_top[0] = 0

    b = y_**3-2*mu_0*dP/y_
    b[0] = y_[0]
    try:
        b[-1] = y_[-1]
    except:
        print y_

    data = [xi_bottom, eta, xi_top]
    diags = [-1, 0, 1]

    A = sparse_diags(data, diags, shape=(len(eta),len(eta)), format='csc')
    return A, b

def comp_mag(P, bcg_P, y0=None, tol=1e-5):
    print 'starting computation of mag. field'
    B_bottom = sqrt(2*mu_0*(bcg_P[-1]-P[-1]))
    if y0 is None:
        y0 = linspace(sqrt(B_top),sqrt(B_bottom),len(P))
    eps = 1

    A_, b_ = A(y0, P, bcg_P)
    A_inv = inv(A_)
    b_try = A_.dot(y0)
    i = 0

    err = norm(b_try-b_)
    while err>tol:
        diff = A_inv.dot(b_-b_try)
        diff[0] = diff[-1] = 0
        y0_new = y0+eps*diff
        A_new,b_new = A(y0_new, P, bcg_P)
        b_try_new = A_.dot(y0)
        if norm(b_try_new-b_new)<err:
            y0 = y0_new
            A_ = A_new
            b_ = b_new
            b_try = b_try_new
            # err = (abs(spsolve(A_,b_try-b_))).max()
            err = norm(b_-b_try)
            eps = eps*1.1
        else:
            eps = eps/2.
        i += 1
        print 'done in {} iterations'.format(i)
    return y0

def comp_mag_old(P, bcg_P, y_=None, tol=1e-6):

    B_bottom = sqrt(2*mu_0*(bcg_P[-1]-P[-1]))
    if y_ is None:
        y_ = linspace(sqrt(B_top),sqrt(B_bottom),len(P))

    dP = bcg_P-P
    h = 25000

    eta = ( Phi/(pi)*(-h**-2) )*ones(len(y_))
    xi = Phi/(h**2*2*pi)*ones(len(y_))
    eta[0] = eta[-1] = 1
    xi_bottom = copy(xi[:-1])
    xi_bottom[-1] = 0
    xi_top = copy(xi[1:])
    xi_top[0] = 0

    A_full, b = A(y_, P, bcg_P)

    while ((A_full.dot(y_)-b)**2).sum()>1e-6:
        b = y_**3-2*mu_0*dP/y_
        b[0] = y_[0]
        try:
            b[-1] = y_[-1]
        except:
            print y_

        data = [xi_bottom, xi_top]
        diags = [-1, 1]

        A_ = sparse_diags(data, diags, shape=(len(eta),len(eta)), format='csc')
        y_ = spsolve(A_full, b)
    return y_

def adjust_base_pressure(mod, mod_old, bcg_P, tau, uplow_velocity):
    def mass(P_bottom, mod, mod_old, P_bcg):
        start = time.time() 
        # print 'pressure difference: ', bcg_P[-1]-P_bottom
        y = mod['y'].values
        y_old = mod_old['y'].values
        P_bcg_bottom = P_bcg[-1]

        # nejprve ziska prubeh tlaku a hutoty pro novy odhad P_bottom
        # TODO zjistit, proc vypocet funguje behem odhadu tlaku a v casovem vyvoji ne. zmeni se neco?
        P, rho = integrate_pressure(P_bottom, mod['T'].values, mod['z'].values, ret_rho=True)
        # jako pocatecni odhad noveho magnetickeho pole je pouzita
        # linearni extrapolace puvodniho
        #
        # nejprve je extrapolovana hodnota na spodnim okraji, podle thin flux tube

        # print 'pressure computed {}'.format(time.time()-start)
        # print 'pressure sum: ', P.sum()

        if mod['y'].equals(mod_old['y']):
            thinflux = (2*mu_0*(P_bcg-P))**0.25
            x = linspace(0,1,len(P_bcg))
            diff = (sqrt(B_top)-thinflux[0])*exp(-10*x)
            diff[-1] = 0
            thinflux_adjusted = thinflux+diff

            y_init = thinflux_adjusted
            # print 'starting computation of y {}'.format(time.time()-start)
            # mod['y'] je modifikovano inplace (mod['y'] je pouze ukazatel)
        else:
            # print 'extrapolating y'
            ratio = (mod['y'].iloc[-1]-(2*mu_0*(P_bcg[-1]-P[-1]))**0.25)/(mod_old['y'].iloc[-1]-mod['y'].iloc[-1])
            y_init = mod['y'].values+ratio*(mod['y'].values-mod_old['y'].values)

        y = comp_mag(P, bcg_P, y_init)
        # print 'y computed {}'.format(time.time()-start)
#        x = rho/y**2
#        h = mod['z'].iloc[1]-mod['z'].iloc[0]
#        return h*Phi*(x[:-1]+x[1:]).sum()/2.
	return Phi*simps(rho/y**2,mod['z'].values)

    P_bottom_init = mod['P'].iloc[-1]
    initial_mass = mass(P_bottom_init, mod, mod_old, bcg_P)
    def func(P_bottom, mod, mod_old, bcg_P, tau):
        if P_bottom>bcg_P[-1]:
            raise ValueError('value of P_bottom is too high')
        y = mod['y'].values
        rho = mod['rho'].values
        ret_val = mass(P_bottom, mod, mod_old, bcg_P)-(initial_mass+uplow_velocity*tau*Phi*rho[-1]/y[-1]**2)
        # print 'ret_val computed: ', ret_val
        return ret_val

    P_bottom_extrap = 2*mod['P'].values[-1]-mod_old['P'].values[-1]
    try:
        return newton(lambda P: func(P, mod, mod_old, bcg_P, tau), P_bottom_extrap, tol=1e8)
    except ValueError:
        print 'using brent method'
        return brentq(lambda P: func(P, mod, mod_old, bcg_P, tau), P_bottom_init, bcg_P[-1], rtol=1e-6)
