#!/usr/bin/env python3
import numpy as np
import constants as c
from __init__ import *
assert kap_lib is not None

# things this buddy returns, but becuase it's fortran it wants them passed in as references
kappa_fracs = np.zeros(num_kap_fracs, dtype=float)
kappa = 0.0
dlnkap_dlnRho = 0.0
dlnkap_dlnT = 0.0
dlnkap_dxa = 0.0
ierr = 0

@np.vectorize
def getMESAOpacity(
    temperature: float,
    density: float,
    massFractions=None,
    Zbase: float = 0.0,
    MesaEOSOutput=None,
    fullOutput=False,
):
    """
    returns the opacity in SI or full mesa output in CGS for a given temperature and density
    ------------
    Parameters:
    temperature: gas temperature in kelvin
    density: gas density in kg/m^3
    massFractions: dictionary of mass fractions of all chemical elements; it is expected to have
    the same keys as baseMassFractions, i.e. h1, h2, he3, he4, li7, ..., si30, p31, s32
    Zbase: base metallicity; TODO ask about what this is
    MesaEOSOutput: dictionary of mesa output including lnfree_e, d_lnfree_e_dlnRho, d_lnfree_e_dlnT, eta, d_eta_dlnRho, d_eta_dlnT
    fullOutput: if True, returns a dictionary of all the outputs of the MESA kap module BUT IN CGS
    """
    if massFractions is None:
        massFractions = baseMassFractions

    assert all(key in allKnownChemicalIDs for key in massFractions.keys())

    logRhoCGS = np.log10(density * c.cm * c.cm * c.cm / c.gram)
    logTCGS = np.log10(temperature)

    if MesaEOSOutput is None:
        lnfree_e = 0.0  # free_e := total combined number per nucleon of free electrons and positrons
        d_lnfree_e_dlnRho = 0.0
        d_lnfree_e_dlnT = 0.0
        eta = 0.0  # electron degeneracy parameter outputed by MESA eos
        d_eta_dlnRho = 0.0
        d_eta_dlnT = 0.0
    else:
        lnfree_e = MesaEOSOutput["lnfree_e"]
        d_lnfree_e_dlnRho = MesaEOSOutput["d_lnfree_e_dlnRho"]
        d_lnfree_e_dlnT = MesaEOSOutput["d_lnfree_e_dlnT"]
        eta = MesaEOSOutput["eta"]
        d_eta_dlnRho = MesaEOSOutput["d_eta_dlnRho"]
        d_eta_dlnT = MesaEOSOutput["d_eta_dlnT"]

    ZbaseString = format(Zbase, ".16E")
    kap_lib.kap_set_control_namelist(handle, "Zbase", ZbaseString, ierr)

    Nspec = len(baseMassFractions)  # number of species in the model
    massFractionsInArr = np.array(
        [], dtype=float
    )  # these are the mass fractions we use
    chem_id = np.array([], dtype=int)  # these are their chemical ids
    net_iso = np.zeros(
        num_chem_isos, dtype=int
    )  # maps chem id to species number (index in the array I suppose? idk man, mesa ppl rly dont like clarity)

    for i, (speciesName, massFraction) in enumerate(baseMassFractions.items()):
        massFractionsInArr = np.append(massFractionsInArr, massFraction)
        chem_id = np.append(chem_id, int(allKnownChemicalIDs[speciesName]))
        net_iso[chem_id[-1]] = i + 1  # +1 because fortran arrays start with one

    kap_res = kap_lib.kap_get(
        handle,
        Nspec,
        chem_id,
        net_iso,
        massFractionsInArr,
        logRhoCGS,
        logTCGS,
        lnfree_e,
        d_lnfree_e_dlnRho,
        d_lnfree_e_dlnT,
        eta,
        d_eta_dlnRho,
        d_eta_dlnT,
        kappa_fracs,
        kappa,
        dlnkap_dlnRho,
        dlnkap_dlnT,
        dlnkap_dxa,
        ierr,
    )
    if fullOutput:
        return kap_res
    else:
        kappaCGS = kap_res["kap"]
        return kappaCGS * c.cm * c.cm / c.gram

if __name__ == "__main__":
    temperature = 1e6
    density = 1e4
    getMESAOpacity(temperature, density)