#!/usr/bin/env python3

# initializes everything important for mesa interface and registers cleanup functions

# NOTE this file will be deleted after the implementation is complete and the contents move to stateEquations

import pyMesaUtils as pym
import atexit
import logging
from dataclasses import dataclass

L = logging.getLogger(__name__)

L.info("Initializing mesa interface")
# universal suff

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

# region eos part

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

atexit.register(L.info, "Shutting down eos tables...")
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

namer = {  # maps indices of eos_res to names of things eos returns vased on eos.def
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
    # from now on down it's information about blending of eos's
    int(eos_def.i_frac_HELM): "frac_HELM",
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
blenInfoNames = ["frac_HELM",
"frac_OPAL_SCVH",
"frac_OPAL_SCVH",
"frac_FreeEOS",
"eos_PC",
"frac_PC",
"eos_Skye",
"frac_Skye",
"eos_CMS",
"frac_CMS",
"eos_ideal",
"frac_ideal"]

net_lib.net_init(ierr)

use_cache = True
ierr = 0
eos_lib.eos_init(pym.EOSPT_CACHE, use_cache, ierr)

eos_handle = eos_lib.alloc_eos_handle(ierr)

chem_h1 = chem_def.ih1.get()

net_h1 = net_lib.ih1.get()

allKnownChemicalIDs = {  # these are expected by the MESA kap module
    # FIXME koukni na enumy
    "h1":   int(chem_def.ih1),
    "h2":   int(chem_def.ih2),
    "he3":  int(chem_def.ihe3),
    "he4":  int(chem_def.ihe4),
    "li7":  int(chem_def.ili7),
    "be7":  int(chem_def.ibe7),
    "b8":   int(chem_def.ib8),
    "c12":  int(chem_def.ic12),
    "c13":  int(chem_def.ic13),
    "n13":  int(chem_def.in13),
    "n14":  int(chem_def.in14),
    "n15":  int(chem_def.in15),
    "o16":  int(chem_def.io16),
    "o17":  int(chem_def.io17),
    "o18":  int(chem_def.io18),
    "f19":  int(chem_def.if19),
    "ne20": int(chem_def.ine20),
    "ne21": int(chem_def.ine21),
    "ne22": int(chem_def.ine22),
    "na22": int(chem_def.ina22),
    "na23": int(chem_def.ina23),
    "mg24": int(chem_def.img24),
    "mg25": int(chem_def.img25),
    "mg26": int(chem_def.img26),
    "al26": int(chem_def.ial26),
    "al27": int(chem_def.ial27),
    "si28": int(chem_def.isi28),
    "si29": int(chem_def.isi29),
    "si30": int(chem_def.isi30),
    "p31":  int(chem_def.ip31),
    "s32":  int(chem_def.is32),
}
eosBasicResultsNum = eos_def.num_eos_basic_results.get()

eosResults = {

'lnPgas':
47.6840706307857,
'lnE':
41.29759091643693,
'lnS':
21.113056762605826,
'eos_PC':
1.7142857142857142,
'eos_Skye':
-0.6931471805599453,
'eos_CMS':
-4.182055570682342,
'eos_ideal':
0.24409135915991434,
'chiRho':
0.15209607875679848,
'chiT':
3.6227296799497877,
'Cp':
29601300438.290466,
'Cv':
3425550951.747412,
'dE_dRho':
-79560740344475.05,
'dS_dT':
3.4255509517474123,
'dS_dRho':
-109895.8301376398,
'gamma1':
1.3143117081558038,
'gamma3':
1.3208121312035388,
'phase':
0.0,
'latent_ddlnT':
-0.0,
'latent_ddlnRho':
-0.0,
'frac_HELM':
0.0,
'frac_OPAL_SCVH':
0.0,
'frac_FreeEOS':
0.0,
'frac_PC':
0.0,
'frac_Skye':
1.0,
'frac_CMS':
0.0,
'frac_ideal':
0.0,

}

@dataclass
class EOSBledningInfo():
    frac_HELM : float
    frac_OPAL_SCVH : float
    frac_FreeEOS : float
    frac_PC : float
    frac_Skye : float
    frac_CMS : float
    frac_ideal : float
    eos_PC : float
    eos_Skye : float
    eos_CMS : float
    eos_ideal : float

@dataclass
class EOSBasicResults():
    lnPgas:float
    lnE:float
    lnS:float
    chiRho:float
    chiT:float
    Cp:float
    Cv:float
    dE_dRho:float
    dS_dT:float
    dS_dRho:float
    gamma1:float
    gamma3:float
    phase:float
    latent_ddlnT:float
    latent_ddlnRho:float

@dataclass
class EOSd_dTResults(EOSBasicResults):
    pass

@dataclass
class EOSd_dPOrRhoResults(EOSBasicResults):
    pass

@dataclass
class EOSFullResults():
    results: EOSBasicResults
    d_dT: EOSd_dTResults
    d_dPOrRho: EOSd_dPOrRhoResults
    blendInfo: EOSBledningInfo

        


# endregion



# region opacity stuff

kap_lib, kap_def = pym.loadMod("kap")
assert kap_def is not None
assert kap_lib is not None

atexit.register(L.info, "Shutting down kappa tables...")
atexit.register(kap_lib.kap_shutdown)


ierr = 0

num_kap_fracs = kap_def.num_kap_fracs
num_chem_isos = chem_def.num_chem_isos

kap_lib.kap_init(False, pym.KAP_CACHE, ierr) # TODO find out how this init works
kap_handle = kap_lib.alloc_kap_handle(ierr)
kap_lib.kap_setup_tables(kap_handle, ierr)
kap_lib.kap_setup_hooks(kap_handle, ierr)


handle = kap_handle
atexit.register(L.info, "Deallocating kap_handle...")
atexit.register(kap_lib.free_kap_handle, kap_handle)


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

@dataclass
class KappaOutput():
    kappa: float
    dlnKappadlnRho: float
    dlnKappdlnT: float
    blendFractions: list
# endregion

if ierr != 0:
    L.critical(f"Mesa initialization failed with ierr {ierr}")
else: 
    L.info("Mesa initialized succesfully")