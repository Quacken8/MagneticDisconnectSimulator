from .opacity_data import xzcotrin21 as xz
from numpy import zeros_like, atleast_1d

xz.set_opal_dir('OPAL/opacity_data/')
xz.set_mol_dir('OPAL/opacity_data/fergson/')
xz.read_extended_opac(20, 0.02, 'GN93hz', 0., '', 21, 0, 0, '')


def opac(T, rho, X=0.7, ztab=0.02):
    T, rho = atleast_1d(T, rho)
    T6 = T*1e-6
    R = rho*1e-3/T6**3
    opact = zeros_like(T)

    for i in range(len(T6)):
        xz.opac(ztab, X, 0., 0., T6[i], R[i])
        opact[i] = 0.1*10**xz.e_opal_z.opact
    return opact