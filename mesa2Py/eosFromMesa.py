# NOTE this file will be deleted after the implementation is complete and the contents move to stateEquations

import numpy as np
import constants as c
from __init__ import *
import logging

L = logging.getLogger(__name__)

assert eos_lib is not None

# these are for eos results, but since theyre in fortran they want to be passed as inputs too
res = np.zeros(eosBasicResultsNum)
Rho = 0.0
log10Rho = 0.0
dlnRho_dlnPgas_const_T = 0.0
dlnRho_dlnT_const_Pgas = 0.0
d_dlnRho_const_T = np.zeros(eosBasicResultsNum, dtype=float)
d_dlnT_const_Rho = np.zeros(eosBasicResultsNum, dtype=float)
d_dabar_const_TRho = np.zeros(eosBasicResultsNum, dtype=float)
d_dzbar_const_TRho = np.zeros(eosBasicResultsNum, dtype=float)
ierr = 0


@np.vectorize
def getEosResult(temperature: float, pressure: float, massFractions=None) -> EOSFullResults:
    """
    returns results of mesa eos in SI in the form of a dictionary
    ---
    temperature: float in Kelvin
    pressure: float in Pa
    """

    if massFractions is None:
        massFractions = baseMassFractions

    assert all(key in allKnownChemicalIDs for key in massFractions.keys())

    pressureCGS = pressure / c.barye
    log10Pressure = np.log10(pressureCGS)
    log10T = np.log10(temperature)

    # assign chemical input
    Nspec = len(massFractions)  # number of species in the model
    d_dxa = np.zeros(
        (eosBasicResultsNum, Nspec), dtype=float
    )  # one more output array that fortran needs as input

    massFractionsInArr = np.array(
        [], dtype=float
    )  # these are the mass fractions we use
    chem_id = np.array([], dtype=int)  # these are their chemical ids
    net_iso = np.zeros(
        num_chem_isos, dtype=int
    )  # maps chem id to species number (index in the array I suppose? idk man, mesa ppl rly dont like clarity)

    for i, (speciesName, massFraction) in enumerate(massFractions.items()):
        massFractionsInArr = np.append(massFractionsInArr, massFraction)
        chem_id = np.append(chem_id, allKnownChemicalIDs[speciesName])
        net_iso[chem_id[-1]] = i + 1  # +1 because fortran arrays start with one

    eos_res = eos_lib.eosPT_get(
        eos_handle,
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
    d_dlnTemp = eos_res["d_dlnt"]
    d_dlndens = eos_res["d_dlnd"]

    # now for each of eosResults entry we want to map it to a name based on the indexer
    eosResultsDict = {}
    L.warn("There may be a problem with derivatives of state variables; it wasn't tested that MESA returns the correct values for PT EOS")
    d_dlnTDict = {} # FIXME I'm not really sure if the PT eos solver really returns P and T derivatives
    d_dlnPDict = {} 
    blendInfoDict = {}
    for i, _ in enumerate(eosResults):
        entryName = namer[i + 1]
        if entryName in blenInfoNames:
            blendInfoDict[entryName] = eosResults[i]
        else:    
            eosResultsDict[entryName] = eosResults[i]
            d_dlnTDict[entryName] = d_dlnTemp[i]
            d_dlnPDict[entryName] = d_dlndens[i]
    
    # and covert to SI


    eosResultsDict["Rho"] *= c.gram / c.cm / c.cm / c.cm
    eosResultsDict["lnE"] += np.log(c.erg / c.gram)
    eosResultsDict["lnS"] += np.log(c.erg / c.gram)
    eosResultsDict["Cv"] *= c.erg / c.gram
    eosResultsDict["Cp"] *= c.erg / c.gram
    eosResultsDict["dE_dRho"] *= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram
    eosResultsDict["dS_dRho"] *= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram
    eosResultsDict["dS_dT"] *= c.erg / c.gram

    d_dlnTDict["Rho"] *= c.gram / c.cm / c.cm / c.cm
    d_dlnTDict["lnE"] += np.log(c.erg / c.gram)
    d_dlnTDict["lnS"] += np.log(c.erg / c.gram)
    d_dlnTDict["Cv"] *= c.erg / c.gram
    d_dlnTDict["Cp"] *= c.erg / c.gram
    d_dlnTDict["dE_dRho"] *= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram
    d_dlnTDict["dS_dRho"] *= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram
    d_dlnTDict["dS_dT"] *= c.erg / c.gram

    gOverCCC = c.gram / c.cm / c.cm / c.cm
    d_dlnPDict["Rho"] *= c.gram / c.cm / c.cm / c.cm / gOverCCC
    d_dlnPDict["lnE"] += np.log(c.erg / c.gram) / gOverCCC
    d_dlnPDict["lnS"] += np.log(c.erg / c.gram) / gOverCCC
    d_dlnPDict["Cv"] *= c.erg / c.gram / gOverCCC
    d_dlnPDict["Cp"] *= c.erg / c.gram / gOverCCC
    d_dlnPDict["dE_dRho"] *= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram / gOverCCC
    d_dlnPDict["dS_dRho"] *= c.erg * c.cm * c.cm * c.cm / c.gram / c.gram / gOverCCC
    d_dlnPDict["dS_dT"] *= c.erg / c.gram / gOverCCC

    basicResults = EOSBasicResults(**eosResultsDict)
    d_dT = EOSd_dTResults(**d_dlnTDict)
    d_dRho = EOSd_dPOrRhoResults(**d_dlnPDict)
    blendInfo = EOSBledningInfo(**blendInfoDict)

    completeResults = EOSFullResults(results = basicResults, d_dT =d_dT, d_dPOrRho = d_dRho, blendInfo = blendInfo)


    return completeResults


d_dlnd = np.zeros(eosBasicResultsNum, dtype=float)
d_dlnT = np.zeros(eosBasicResultsNum, dtype=float)


def getEosResultRhoTCGS(temperature: float, density: float, massFractions=None) -> EOSFullResults:
    """
    returns results of mesa eos in CGS
    ---
    temperature: float in Kelvin
    density: float in kg/m^3
    """

    if massFractions is None:
        massFractions = baseMassFractions

    assert all(key in allKnownChemicalIDs for key in massFractions.keys())

    densityCGS = density * c.cm * c.cm * c.cm / c.gram
    log10Density = np.log10(densityCGS)
    log10T = np.log10(temperature)

    # assign chemical input
    Nspec = len(massFractions)  # number of species in the model
    d_dxa = np.zeros((eosBasicResultsNum, Nspec), dtype=float)

    massFractionsInArr = np.array(
        [], dtype=float
    )  # these are the mass fractions we use
    chem_id = np.array([], dtype=int)  # these are their chemical ids
    net_iso = np.zeros(
        num_chem_isos, dtype=int
    )  # maps chem id to species number (index in the array I suppose? idk man, mesa ppl rly dont like clarity)

    for i, (speciesName, massFraction) in enumerate(massFractions.items()):
        massFractionsInArr = np.append(massFractionsInArr, massFraction)
        chem_id = np.append(chem_id, int(allKnownChemicalIDs[speciesName]))
        net_iso[chem_id[-1]] = i + 1  # +1 because fortran arrays start with one

    eos_res = eos_lib.eosDT_get(
        eos_handle,
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
        entryName = namer[i + 1]
        if entryName in blenInfoNames:
            blendInfoDict[entryName] = eosResults[i]
        else:    
            eosResultsDict[entryName] = eosResults[i]
            d_dlnTDict[entryName] = d_dlnTemp[i]
            d_dlndDict[entryName] = d_dlndens[i]
    
    basicResults = EOSBasicResults(**eosResultsDict)
    d_dT = EOSd_dTResults(**d_dlnTDict)
    d_dRho = EOSd_dPOrRhoResults(**d_dlndDict)
    blendInfo = EOSBledningInfo(**blendInfoDict)

    completeResults = EOSFullResults(results = basicResults, d_dT =d_dT, d_dPOrRho = d_dRho, blendInfo = blendInfo)

    return completeResults


if __name__ == "__main__":
    temperature = 1000000000.0000000
    pressure = 10.0**2
    densityCGS = 10000.000000000000
    density = densityCGS * c.gram / (c.cm * c.cm * c.cm)
    massFractions = {"c12": 1.0}
    results = getEosResultRhoTCGS(temperature, density, massFractions).results

    lnPgasCGS = results.lnPgas
    PgasCGS = np.exp(lnPgasCGS)
    print(PgasCGS)

    print(getEosResult(temperature, pressure, massFractions))