#!/usr/bin/env python3
import numpy as np
import constants as c
try:
    from .eosFromMesa import getEosResult, getEosResultRhoTCGS
except ImportError:
    from eosFromMesa import getEosResult, getEosResultRhoTCGS

try: 
    mesaInit
except NameError:
    try:
        from . import initializer as mesaInit
    except ImportError:
        import initializer as mesaInit




# things this buddy returns, but becuase it's fortran it wants them passed in as references
kappa_fracs = np.zeros(mesaInit.num_kap_fracs, dtype=float)
kappa = 0.0
dlnkap_dlnRho = 0.0
dlnkap_dlnT = 0.0
dlnkap_dxa = 0.0
ierr = 0

@np.vectorize
def getMESAOpacityRhoT(
    density: float,
    temperature: float,
    massFractions: dict | None = None,
    MesaEOSOutput : mesaInit.EOSFullResults | None=None,
):
    """
    returns the opacity or full mesa output in SI for a given temperature and density
    ------------
    Parameters:
    temperature: gas temperature in kelvin
    density: gas density in kg/m^3
    massFractions: dictionary of mass fractions of all chemical elements; it is expected to have
    the same keys as baseMassFractions, i.e. h1, h2, he3, he4, li7, ..., si30, p31, s32
    MesaEOSOutput: dictionary of mesa output including lnfree_e, d_lnfree_e_dlnRho, d_lnfree_e_dlnT, eta, d_eta_dlnRho, d_eta_dlnT in CGS FIXME this is ew, this should be SI 
    fullOutput: if True, returns a dictionary of all the outputs of the MESA kap module BUT IN CGS
    """
    if massFractions is None:
        massFractions = mesaInit.solarAbundancesDict

    assert all(key in mesaInit.allKnownChemicalIDs for key in massFractions.keys())

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
        lnfree_e = MesaEOSOutput.results.lnfree_e
        d_lnfree_e_dlnRho = MesaEOSOutput.d_dPOrRho.lnfree_e
        d_lnfree_e_dlnT = MesaEOSOutput.d_dT.lnfree_e # FIXME maybe create a new class for rho T and P T results?
        eta = MesaEOSOutput.results.eta
        d_eta_dlnRho = MesaEOSOutput.d_dPOrRho.eta
        d_eta_dlnT = MesaEOSOutput.d_dT.eta
     

    Nspec = len(massFractions)  # number of species in the model
    massFractionsInArr = np.array(
        [], dtype=float
    )  # these are the mass fractions we use
    chem_id = np.array([], dtype=int)  # these are their chemical ids
    net_iso = np.zeros(
        mesaInit.num_chem_isos, dtype=int
    )  # maps chem id to species number (index in the array I suppose? idk man, mesa ppl rly dont like clarity)

    Zbase = 0.0
    
    for i, (speciesName, massFraction) in enumerate(mesaInit.solarAbundancesDict.items()):
        if speciesName not in ("h1", "h2", "he3", "he4"):
            Zbase += massFraction
        massFractionsInArr = np.append(massFractionsInArr, massFraction)
        chem_id = np.append(chem_id, int(mesaInit.allKnownChemicalIDs[speciesName]))
        net_iso[chem_id[-1]] = i + 1  # +1 because fortran arrays start with one

    ZbaseString = format(Zbase, ".16E")
    mesaInit.kap_lib.kap_set_control_namelist(mesaInit.kap_handle, "Zbase", ZbaseString, ierr)

    kap_res = mesaInit.kap_lib.kap_get(
        mesaInit.kap_handle,
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

    kappaRes = kap_res["kap"] * c.cm * c.cm / c.gram
    dlnKappadlnRho = kap_res[
        "dlnkap_dlnrho"
    ]  # TODO check if the log rly takes care of the units just to be sure
    dlnKappdlnT = kap_res["dlnkap_dlnt"]

    output = mesaInit.KappaOutput(
        kappa=kappaRes,
        dlnKappadlnRho=dlnKappadlnRho,
        dlnKappdlnT=dlnKappdlnT,
        blendFractions=kap_res["kap_fracs"],
    )

    return output

@np.vectorize
def getMesaOpacity(pressure:float, temperature:float, massFractions:dict|None=None) -> float:
    """Takes in temperature and pressure in SI units and then calls MESA EOS to get all relevant
    input variables for mesa kappa

    Args:
        temperature (np.ndarray): K
        pressure (np.ndarray): Pa
        massFractions (dictionary, optional): dictionary of mass fractions of all chemical elements; it is expected to have
    the same keys as baseMassFractions, i.e. h1, h2, he3, he4, li7, ..., si30, p31, s32. Defaults to solar abundances.

    Returns:
        np.ndarray: MESA kappa full results in SI
    """
    density = getEosResult(temperature, pressure, massFractions, cgsOutput=False).results.rho
    rhoTEosResults = getEosResultRhoTCGS(density, temperature)

    return getMESAOpacityRhoT(density, temperature, massFractions, rhoTEosResults)



if __name__ == "__main__":
    temperature = 1e6
    density = 1e10
    print(getMESAOpacityRhoT(temperature, density))
