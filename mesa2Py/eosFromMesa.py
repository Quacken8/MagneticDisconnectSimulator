# NOTE this file will be deleted after the implementation is complete and the contents move to stateEquations

import numpy as np
import constants as c
from warnings import warn
 
try: 
    mesaInit
except NameError:
    from . import initializer as mesaInit

assert mesaInit.eos_lib is not None

# these are for eos results, but since theyre in fortran they want to be passed as inputs too
res = np.zeros(mesaInit.eosBasicResultsNum)
Rho = 0.0
log10Rho = 0.0
dlnRho_dlnPgas_const_T = 0.0
dlnRho_dlnT_const_Pgas = 0.0
d_dlnRho_const_T = np.zeros(mesaInit.eosBasicResultsNum, dtype=float)
d_dlnT_const_Rho = np.zeros(mesaInit.eosBasicResultsNum, dtype=float)
d_dabar_const_TRho = np.zeros(mesaInit.eosBasicResultsNum, dtype=float)
d_dzbar_const_TRho = np.zeros(mesaInit.eosBasicResultsNum, dtype=float)
ierr = 0

@np.vectorize
def getEosResult(
    temperature: float, pressure: float, massFractions=None, cgs=False
) -> mesaInit.EOSFullResults:
    """
    MESA based eos
    ---
    returns a mesaInit.EOSFullResults object
    which contains basicResults, d_dT, d_dP and blendInfo
    the first three all have variables rho, lnPgas, lnE, lnS, mu, lnfree_e, eta, chiRho, chiT, Cp, Cv, dE_dRho, dS_dT, dS_dRho, gamma1, gamma3, phase, ,latent_ddlnT, latent_ddlnRho, grad_ad
    ---
    temperature: float in Kelvin
    pressure: float in Pa
    """

    if massFractions is None:
        massFractions = mesaInit.solarAbundancesDict

    assert all(key in mesaInit.allKnownChemicalIDs for key in massFractions.keys())

    pressureCGS = pressure / c.barye
    log10Pressure = np.log10(pressureCGS)
    log10T = np.log10(temperature)

    # assign chemical input
    Nspec = len(massFractions)  # number of species in the model
    d_dxa = np.zeros(
        (mesaInit.eosBasicResultsNum, Nspec), dtype=float
    )  # one more output array that fortran needs as input

    massFractionsInArr = np.array(
        [], dtype=float
    )  # these are the mass fractions we use
    chem_id = np.array([], dtype=int)  # these are their chemical ids
    net_iso = np.zeros(
        mesaInit.num_chem_isos, dtype=int
    )  # maps chem id to species number (index in the array I suppose? idk man, mesa ppl rly dont like clarity)

    for i, (speciesName, massFraction) in enumerate(massFractions.items()):
        massFractionsInArr = np.append(massFractionsInArr, massFraction)
        chem_id = np.append(chem_id, int(mesaInit.allKnownChemicalIDs[speciesName]))
        net_iso[chem_id[-1]] = i + 1  # +1 because fortran arrays start with one

    eos_res = mesaInit.eos_lib.eosPT_get(
        mesaInit.eos_handle,
        Nspec,
        chem_id,
        net_iso,
        massFractionsInArr,
        pressureCGS,
        log10Pressure,
        temperature,
        log10T,
        Rho,
        log10Rho,
        dlnRho_dlnPgas_const_T,
        dlnRho_dlnT_const_Pgas,
        res,
        d_dlnRho_const_T,
        d_dlnT_const_Rho,
        d_dxa,
        ierr,
    )

    eosResults = eos_res["res"]
    d_dlnTemp = eos_res["d_dlnt_const_rho"]
    d_dlndens = eos_res["d_dlnrho_const_t"]

    # now for each of eosResults entry we want to map it to a name based on the indexer
    eosResultsDict = {"rho": eos_res['rho']}
    warn(
        "There may be a problem with derivatives of state variables; it wasn't tested that MESA returns the correct values for PT EOS"
    )
    d_dlnTDict = {
        "rho": dlnRho_dlnPgas_const_T
    }  # FIXME I'm not really sure if the PT eos solver really returns P and T derivatives
    blendInfoDict = {}
    d_dlnPDict = {"rho": dlnRho_dlnPgas_const_T}
    for i, _ in enumerate(eosResults):
        entryName = mesaInit.namer[i + 1]
        if entryName in mesaInit.blenInfoNames:
            blendInfoDict[entryName] = eosResults[i]
        else:
            eosResultsDict[entryName] = eosResults[i]
            d_dlnTDict[entryName] = d_dlnTemp[i]
            d_dlnPDict[entryName] = d_dlndens[i]

    # and covert to SI

    if not cgs:
        eosResultsDict["rho"] /= c.gram / c.cm / c.cm / c.cm
        eosResultsDict["lnE"] -= np.log(c.erg / c.gram)
        eosResultsDict["lnS"] -= np.log(c.erg / c.gram)
        eosResultsDict["Cv"] /= c.erg / c.gram
        eosResultsDict["Cp"] /= c.erg / c.gram
        eosResultsDict["dE_dRho"] /= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram
        eosResultsDict["dS_dRho"] /= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram
        eosResultsDict["dS_dT"] /= c.erg / c.gram

        d_dlnTDict["rho"] /= c.gram / c.cm / c.cm / c.cm
        d_dlnTDict["lnE"] -= 0 # becuase the constant gets derivated away since it's in a log
        d_dlnTDict["lnS"] -= 0 # becuase the constant gets derivated away since it's in a log
        d_dlnTDict["Cv"] /= c.erg / c.gram
        d_dlnTDict["Cp"] /= c.erg / c.gram
        d_dlnTDict["dE_dRho"] /= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram
        d_dlnTDict["dS_dRho"] /= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram
        d_dlnTDict["dS_dT"] /= c.erg / c.gram

        gOverCCC = c.gram / c.cm / c.cm / c.cm
        d_dlnPDict["rho"] /= c.gram / c.cm / c.cm / c.cm / gOverCCC
        d_dlnPDict["lnE"] -= 0 # becuase the constant gets derivated away since it's in a log
        d_dlnPDict["lnS"] -= 0 # becuase the constant gets derivated away since it's in a log
        d_dlnPDict["Cv"] /= c.erg / c.gram / gOverCCC
        d_dlnPDict["Cp"] /= c.erg / c.gram / gOverCCC
        d_dlnPDict["dE_dRho"] /= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram / gOverCCC
        d_dlnPDict["dS_dRho"] /= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram / gOverCCC
        d_dlnPDict["dS_dT"] /= c.erg / c.gram / gOverCCC

    basicResults = mesaInit.EOSBasicResults(**eosResultsDict)
    d_dT = mesaInit.EOSd_dTResults(**d_dlnTDict)
    d_dRho = mesaInit.EOSd_dPOrRhoResults(**d_dlnPDict)
    blendInfo = mesaInit.EOSBledningInfo(**blendInfoDict)

    completeResults = mesaInit.EOSFullResults(
        results=basicResults, d_dT=d_dT, d_dPOrRho=d_dRho, blendInfo=blendInfo
    )

    return completeResults


d_dlnd = np.zeros(mesaInit.eosBasicResultsNum, dtype=float)
d_dlnT = np.zeros(mesaInit.eosBasicResultsNum, dtype=float)


def getEosResultRhoTCGS(
    temperature: float, density: float, massFractions=None
) -> mesaInit.EOSFullResults:
    """
    returns results of mesa eos in CGS
    ---
    temperature: float in Kelvin
    density: float in kg/m^3
    """

    if massFractions is None:
        massFractions = mesaInit.solarAbundancesDict

    assert all(key in mesaInit.allKnownChemicalIDs for key in massFractions.keys())

    densityCGS = density * c.cm * c.cm * c.cm / c.gram
    log10Density = np.log10(densityCGS)
    log10T = np.log10(temperature)

    # assign chemical input
    Nspec = len(massFractions)  # number of species in the model
    d_dxa = np.zeros((mesaInit.eosBasicResultsNum, Nspec), dtype=float)

    massFractionsInArr = np.array(
        [], dtype=float
    )  # these are the mass fractions we use
    chem_id = np.array([], dtype=int)  # these are their chemical ids
    net_iso = np.zeros(
        mesaInit.num_chem_isos, dtype=int
    )  # maps chem id to species number (index in the array I suppose? idk man, mesa ppl rly dont like clarity)

    for i, (speciesName, massFraction) in enumerate(massFractions.items()):
        massFractionsInArr = np.append(massFractionsInArr, massFraction)
        chem_id = np.append(chem_id, int(mesaInit.allKnownChemicalIDs[speciesName]))
        net_iso[chem_id[-1]] = i + 1  # +1 because fortran arrays start with one

    eos_res = mesaInit.eos_lib.eosDT_get(
        mesaInit.eos_handle,
        Nspec,
        chem_id,
        net_iso,
        massFractionsInArr,
        densityCGS,
        log10Density,
        temperature,
        log10T,
        res,
        d_dlnd,
        d_dlnT,
        d_dxa,
        ierr,
    )

    eosResults = eos_res["res"]
    d_dlnTemp = eos_res["d_dlnt"]
    d_dlndens = eos_res["d_dlnd"]

    # now for each of eosResults entry we want to map it to a name based on the indexer
    eosResultsDict = {}
    d_dlnTDict = {}
    d_dlndDict = {}
    blendInfoDict = {}
    for i, _ in enumerate(eosResults):
        entryName = mesaInit.namer[i + 1]
        if entryName in mesaInit.blenInfoNames:
            blendInfoDict[entryName] = eosResults[i]
        else:
            eosResultsDict[entryName] = eosResults[i]
            d_dlnTDict[entryName] = d_dlnTemp[i]
            d_dlndDict[entryName] = d_dlndens[i]

    basicResults = mesaInit.EOSBasicResults(**eosResultsDict)
    d_dT = mesaInit.EOSd_dTResults(**d_dlnTDict)
    d_dRho = mesaInit.EOSd_dPOrRhoResults(**d_dlndDict)
    blendInfo = mesaInit.EOSBledningInfo(**blendInfoDict)

    completeResults = mesaInit.EOSFullResults(
        results=basicResults, d_dT=d_dT, d_dPOrRho=d_dRho, blendInfo=blendInfo
    )

    return completeResults


if __name__ == "__main__":
    temperature = 1000000000.0000000
    pressure = 5.115979e20 * c.barye
    massFractions = {"c12": 1.0}
    results = getEosResult(temperature, pressure, massFractions).results

    for name, value in vars(results).items():
        print(f"{name} \t {getattr(results, name):.6e}")

