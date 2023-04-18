# NOTE this file will be deleted after the implementation is complete and the contents move to stateEquations

import numpy as np
import pyMesaUtils as pym
import constants as c
import atexit

# FIXME make an initializer for all of this

eos_lib, eos_def = pym.loadMod("eos")
const_lib, const_def = pym.loadMod("const")
crlibm_lib, _ = pym.loadMod("math")
assert eos_lib is not None
assert eos_def is not None
assert const_lib is not None
assert const_def is not None
assert crlibm_lib is not None
crlibm_lib.math_init()
chem_lib, chem_def = pym.loadMod("chem")
net_lib, net_def = pym.loadMod("net")
rates_lib, rates_def = pym.loadMod("rates")
assert chem_lib is not None
assert chem_def is not None
assert net_lib is not None
assert net_def is not None
assert rates_lib is not None
assert rates_def is not None

atexit.register(print, "Shutting down eos tables...")
atexit.register(eos_lib.eos_shutdown)

ierr = 0

const_lib.const_init(pym.MESA_DIR, ierr)
chem_lib.chem_init("isotopes.data", ierr)
num_chem_isos = chem_def.num_chem_isos

ierr = 0

rates_lib.rates_init(  # TODO What is this?
    "reactions.list",
    "jina_reaclib_results_20130213default2",
    "rate_tables",
    False,
    False,
    "",
    "",
    "",
    ierr,
)

namer = { # maps indices of eos_res to names of things eos returns vased on eos.def
    int(eos_def.i_lnPgas): "lnPgas",
    int(eos_def.i_lnE): "lnE",
    int(eos_def.i_lnS): "lnS",
    int(eos_def.i_mu): "mu",
    int(eos_def.i_lnfree_e): "lnfree_e",
    int(eos_def.i_grad_ad): "grad_ad",
    int(eos_def.i_chiRho): "chiRho",
    int(eos_def.i_chiT): "chiT",
    int(eos_def.i_Cp): "Cp",
    int(eos_def.i_Cv): "Cv",
    int(eos_def.i_dE_dRho): "dE_dRho",
    int(eos_def.i_dS_dT): "dS_dT",
    int(eos_def.i_dS_dRho): "dS_dRho",
    int(eos_def.i_gamma1): "gamma1",
    int(eos_def.i_gamma3): "gamma3",
    int(eos_def.i_phase): "phase",
    int(eos_def.i_latent_ddlnT): "latent_ddlnT",
    int(eos_def.i_latent_ddlnRho): "latent_ddlnRho",
    int(eos_def.i_frac_HELM): "frac_HELM", # from now on down it's information about blending of eos's
    int(eos_def.i_frac_OPAL_SCVH): "frac_OPAL_SCVH",
    int(eos_def.i_frac_OPAL_SCVH): "frac_OPAL_SCVH",
    int(eos_def.i_frac_FreeEOS): "frac_FreeEOS",
    int(eos_def.i_eos_PC): "eos_PC",
    int(eos_def.i_frac_PC): "frac_PC",
    int(eos_def.i_eos_Skye): "eos_Skye",
    int(eos_def.i_frac_Skye): "frac_Skye",
    int(eos_def.i_eos_CMS): "eos_CMS",
    int(eos_def.i_frac_CMS): "frac_CMS",
    int(eos_def.i_eos_ideal): "eos_ideal",
    int(eos_def.i_frac_ideal): "frac_ideal", 
}

net_lib.net_init(ierr)

eosDT_cache_dir = ""  # they say blank means use default heh # FIXME OMG CAN THIS DO PT?
use_cache = True
ierr = 0
eos_lib.eos_init(eosDT_cache_dir, use_cache, ierr)

eos_handle = eos_lib.alloc_eos_handle(ierr)

chem_h1 = chem_def.ih1.get()

net_h1 = net_lib.ih1.get()

# region chemical stuff

baseMassFractions = {  # these are user friendly input, feel free to change them
    "h1": 7.1634195514280885e-01,
    "h2": 1.7352293836755894e-17,
    "he3": 2.9132242698695344e-04,
    "he4": 2.7322406948895939e-01,
    "li7": 2.3928335699763513e-12,
    "be7": 1.1308511228070763e-11,
    "b8": 4.0513713928796461e-21,
    "c12": 1.0460386560101952e-03,
    "c13": 5.0633374170403897e-05,
    "n13": 6.9022734150661748e-99,
    "n14": 1.3425077169185414e-03,
    "n15": 9.7808691108066830e-07,
    "o16": 4.7195548329965354e-03,
    "o17": 2.2368335263402889e-05,
    "o18": 7.7074699765302857e-06,
    "f19": 2.8086891694064844e-07,
    "ne20": 9.0884887562465695e-04,
    "ne21": 2.3054022749731481e-06,
    "ne22": 8.4088700929700188e-04,
    "na22": 4.9718197550528217e-05,
    "na23": 1.3273632976007170e-04,
    "mg24": 2.9311166398416483e-04,
    "mg25": 3.8624611610720092e-05,
    "mg26": 4.4267917796202449e-05,
    "al26": 1.1565109917985960e-08,
    "al27": 3.2404172357425902e-05,
    "si28": 3.7261916365394915e-04,
    "si29": 1.9596271466878853e-05,
    "si30": 1.3363396270305292e-05,
    "p31": 3.5530240519619196e-06,
    "s32": 2.0053598557081472e-04,
}

allKnownChemicalIDs = {  # these are expected by the MESA kap module
    "h1": chem_def.ih1,
    "h2": chem_def.ih2,
    "he3": chem_def.ihe3,
    "he4": chem_def.ihe4,
    "li7": chem_def.ili7,
    "be7": chem_def.ibe7,
    "b8": chem_def.ib8,
    "c12": chem_def.ic12,
    "c13": chem_def.ic13,
    "n13": chem_def.in13,
    "n14": chem_def.in14,
    "n15": chem_def.in15,
    "o16": chem_def.io16,
    "o17": chem_def.io17,
    "o18": chem_def.io18,
    "f19": chem_def.if19,
    "ne20": chem_def.ine20,
    "ne21": chem_def.ine21,
    "ne22": chem_def.ine22,
    "na22": chem_def.ina22,
    "na23": chem_def.ina23,
    "mg24": chem_def.img24,
    "mg25": chem_def.img25,
    "mg26": chem_def.img26,
    "al26": chem_def.ial26,
    "al27": chem_def.ial27,
    "si28": chem_def.isi28,
    "si29": chem_def.isi29,
    "si30": chem_def.isi30,
    "p31": chem_def.ip31,
    "s32": chem_def.is32,
}

# endregion

eosBasicResultsNum = eos_def.num_eos_basic_results.get()

# these are for eos results, but since theyre in fortran they want to be passed as inputs too
res = np.zeros(eosBasicResultsNum)
Rho = 0.0
log10Rho = 0.0
dlnRho_dlnPgas_const_T =0.0
dlnRho_dlnT_const_Pgas =0.0
d_dlnRho_const_T = np.zeros(eosBasicResultsNum, dtype=float)
d_dlnT_const_Rho = np.zeros(eosBasicResultsNum, dtype=float)
d_dabar_const_TRho = np.zeros(eosBasicResultsNum, dtype=float)
d_dzbar_const_TRho = np.zeros(eosBasicResultsNum, dtype=float)
ierr = 0

@np.vectorize
def getEosResultCGS(temperature: float, pressure: float, massFractions=None):
    """
    returns results of mesa eos in CGS
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
    d_dxa = np.zeros((eosBasicResultsNum, Nspec), dtype=float)  # one more output array that fortran needs as input

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
    
    # now for each of eosResults entry we want to map it to a name based on the indexer
    eosResultsDict = {}
    for i, entry in enumerate(eosResults):
        entryName = namer[i+1]
        eosResultsDict[entryName] = entry
    
    return eosResultsDict

def getEosResultSI(temperature: float, pressure: float, masFractions = None):
    CGSResuts = getEosResultCGS(temperature, pressure, masFractions)

d_dlnd = np.zeros(eosBasicResultsNum, dtype=float)
d_dlnT = np.zeros(eosBasicResultsNum, dtype=float)


def getEosResultRhoTCGS(temperature: float, density: float, massFractions=None):
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
    for i, _ in enumerate(eosResults):
        entryName = namer[i+1]
        eosResultsDict[entryName] = eosResults[i]
        d_dlnTDict[entryName] = d_dlnTemp[i]
        d_dlndDict[entryName] = d_dlndens[i]
    
    completeResults = {
        "eosResults": eosResultsDict,
        "d_dlnT": d_dlnTDict,
        "d_dlnRho": d_dlndDict,
    }

    return completeResults


if __name__ == "__main__":
    temperature = 1000000000.0000000
    pressure = 10.0**2
    densityCGS = 10000.000000000000     
    density = densityCGS * c.gram / (c.cm * c.cm * c.cm)
    massFractions = {"c12": 1.0}
    results = getEosResultRhoTCGS(temperature, density, massFractions)['eosResults']

    lnPgasCGS = results["lnPgas"]
    PgasCGS = np.exp(lnPgasCGS)
    print(PgasCGS)

    for key, entry in results.items():
        print(key, entry)
    
    print(getEosResultCGS(temperature, pressure, massFractions))
