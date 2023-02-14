from numpy import log, exp, array, concatenate, linspace, asarray
from pandas import read_csv, DataFrame
from scipy.integrate import ode
from scipy.optimize import bisect
from scipy.interpolate import UnivariateSpline

from OPAL.EOS import gas_state
from model_s.interp import g

def TofS(P0, S0, T0):
    return bisect(lambda T: S0-gas_state(T,P0)['S'], 3300, 15000)

def get_ini(p0_ratio):
    bcg = read_csv('data/static/bcg.dat', delimiter='\t')

    # prakticky jediny volitelny parametr inicialniho modelu
    # je tlak u povrchu
    P_top = p0_ratio*bcg['P'].values[0]
    S_bottom = bcg['S'].values[-1]
    T_top = TofS(P_top, S_bottom, bcg['T'].values[0])
    print(T_top)

    def f(logP, y):
        logT, z = y
        T = exp(logT)
        P = exp(logP)
        state = gas_state(T, P)
        rho = state['rho']
        nabla_ad = state['nabla_ad']
        H = P/(rho*g(z))
        return array([ nabla_ad, H ])

    r = ode(f).set_integrator('dopri5')
    y0 = array([ log(T_top), 0 ])
    r.set_initial_value(y0, log(P_top))
    dlogP = 0.01

    model = []
    while r.successful() and r.y[1]<=12.7e6:
        logT = r.y[0]
        z = r.y[1]
        logP = r.t

        T = exp(logT)
        P = exp(logP)
        gs = gas_state(T,P)
        new_state = concatenate(( [z], [T], [P],
                      gs['nabla_ad'], gs['S'], gs['rho'], gs['delta'],
                      gs['cp'], gs['cv']))
        model.append(new_state)
        r.integrate(logP+dlogP)

    model = asarray(model)
    z_model = model[:,0]
    T_model = model[:,1]
    P_model = model[:,2]
    nabla_ad_model = model[:,3]
    S_model = model[:,4]
    rho_model = model[:,5]
    delta_model = model[:,6]
    cp_model = model[:,7]
    cv_model = model[:,8]

    z = linspace(0, 12.5e6, 501)
    T = UnivariateSpline(z_model, T_model, s=0, k=1)(z)
    P = UnivariateSpline(z_model, P_model, s=0, k=1)(z)
    nabla_ad = UnivariateSpline(z_model, nabla_ad_model, s=0, k=1)(z)
    S = UnivariateSpline(z_model, S_model, s=0, k=1)(z)
    rho = UnivariateSpline(z_model, rho_model, s=0, k=1)(z)
    delta = UnivariateSpline(z_model, delta_model, s=0, k=1)(z)
    cp = UnivariateSpline(z_model, cp_model, s=0, k=1)(z)
    cv = UnivariateSpline(z_model, cv_model, s=0, k=1)(z)

    model = DataFrame(columns = ['z', 'T', 'P', 'nabla_ad',
                     'S', 'rho', 'delta', 'cp', 'cv'], data=array([
        z, T, P, nabla_ad, S, rho, delta, cp, cv
    ]).T
              )
    model.to_csv('data/static/initial_state_T0_{T0:.0f}.dat'.format(T0=T_top), sep='\t', index=False)
    return model
