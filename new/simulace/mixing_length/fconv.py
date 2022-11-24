from numpy import sqrt, log, concatenate, zeros, amax, argmax, nan_to_num
import numpy
from scipy.constants import sigma
from scipy.ndimage.filters import gaussian_filter
from model_s import interp

def nabla_func(T,P):
    logT = log(T)
    logP = log(P)
    return concatenate((
        [(logT[1]-logT[0])/(logP[1]-logP[0])],
        (logT[2:]-logT[:-2])/(logP[2:]-logP[:-2]),
        [(logT[-1]-logT[-2])/(logP[-1]-logP[-2])]
    ))

def U_func(T, cp, rho, opac, H, alpha, g):
    return 12*(sigma*T**3/(cp*rho**2*opac*H**2)*(1/alpha)**2*sqrt(8*H/g))

def frad(mod):
    T = mod['T'].values
    P = mod['P'].values
    rho = mod['rho'].values
    h = mod['z'].iloc[1]-mod['z'].iloc[0]
    dTdz = concatenate((
        [(T[1]-T[0])/h],
        (T[2:]-T[:-2])/(2*h),
        [T[-1]-T[-2]/h]
    ))
    opac = mod['opac'].values
    nabla = nabla_func(T,P)
    g = interp.g(mod['z'].values)
    return -16./3*sigma*g*T**3*dTdz/(opac*rho)

def fconv(alpha,mod):
    rho = mod['rho'].values
    cp = mod['cp'].values
    T = mod['T'].values
    P = mod['P'].values
    delta = mod['delta'].values
    opac = mod['opac'].values
    nabla_ad = mod['nabla_ad'].values
    g = interp.g(mod['z'].values)
    nabla = nabla_func(T,P)
    H = P/(rho*g)
    U = U_func(T,cp,rho,opac,H,alpha,g)
    nabla_e = nabla_ad-2*U**2+2*U*sqrt(abs(nabla-nabla_ad)+U**2)

    negative_index = numpy.where(nabla-nabla_e<0)
    nabla_diff = amax([zeros(len(nabla)), gaussian_filter(nabla-nabla_e,2)], axis=0)
    if len(negative_index[0])>0:
        for ind in negative_index[0]:
            if T[ind]>20000:
                print 'zeroing difference of nablas from index', ind
                nabla_diff[ind:] = 0
                break
    else:
        print 'all values of nabla-nable_e are > 0'

    ret_val = rho*cp*T*sqrt(g*delta)*alpha**2*H**(0.5)*(gaussian_filter(nabla_diff, 0)**1.5)/(4*(sqrt(2)))
    # TODO zminit zhlazovani v praci
    ind_max = argmax(ret_val)
    # frad_vals = frad(mod)
    # ratio = ret_val.max()/frad_vals[ind_max]
    ret_val[:ind_max] = ret_val.max()
    # ret_val[:ind_max] = frad_vals[:ind_max]*ratio
    return -nan_to_num(ret_val)
