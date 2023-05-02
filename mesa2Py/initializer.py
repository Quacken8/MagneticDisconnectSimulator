#!/usr/bin/env python3

# initializes everything important for mesa interface and registers cleanup functions
import numpy as np
import pyMesaUtils as pym
import atexit
import logging
from dataclasses import dataclass


L = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

if not hasattr(pym, 'mesa2Py'):
    L.info("Initializing mesa interface")
else:
    L.info("mesa interface already initialized, skipping initialization")
# universal suff

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
rates_lib, rates_def = pym.loadMod("rates") # reaction rates
assert chem_lib is not None
assert chem_def is not None
assert net_lib is not None
assert net_def is not None
assert rates_lib is not None
assert rates_def is not None

ierr = 0

# this trick makes sure these inits only happen once
if not hasattr(pym, 'mesa2Py'):
    # EOS initialization
    atexit.register(L.info, "Shutting down eos tables...")
    atexit.register(eos_lib.eos_shutdown)
    const_lib.const_init(pym.MESA_DIR, ierr)
    chem_lib.chem_init("isotopes.data", ierr)

    reactionlistFilename = "reactions.list"
    jinaReaclibFilename = "jina_reaclib_results_20130213default2"
    rateTablesDir = "rate_tables"
    useSuzukiWeakRates = False
    useSpecialWeakRates = False
    specialWeakStatesFile = ""
    speacialWeakTransitionsFile = ""
    cacheDir = "" # "" means use default

    rates_lib.rates_init(
        reactionlistFilename,
        jinaReaclibFilename,
        rateTablesDir,
        useSuzukiWeakRates,
        useSpecialWeakRates,
        specialWeakStatesFile,
        speacialWeakTransitionsFile,
        cacheDir,
        ierr,
    )
    net_lib.net_init(ierr)

    use_cache = True

    eos_lib.eos_init(pym.EOSPT_CACHE, use_cache, ierr)

    eos_handle = eos_lib.alloc_eos_handle(ierr)

num_chem_isos = chem_def.num_chem_isos
namer = {  # maps indices of eos_res to names of things eos returns vased on eos.def
    int(eos_def.i_lnPgas): "lnPgas",
    int(eos_def.i_lnE): "lnE",
    int(eos_def.i_lnS): "lnS",
    int(eos_def.i_mu): "mu",
    int(eos_def.i_lnfree_e): "lnfree_e",
    int(eos_def.i_eta): "eta",
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
    int(eos_def.i_frac_PC): "frac_PC",
    int(eos_def.i_frac_Skye): "frac_Skye",
    int(eos_def.i_frac_CMS): "frac_CMS",
    int(eos_def.i_frac_ideal): "frac_ideal",
}
blenInfoNames = [
    "frac_HELM",
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
    "frac_ideal",
]

chem_h1 = chem_def.ih1.get()

net_h1 = net_lib.ih1.get()

isotopeNames = [ # must be harcoded becuase chem_def.namsol for some reason only returns first element :c
       'h1   ','h2   ','he3  ','he4  ','li6  ','li7  ','be9  ','b10  ',
       'b11  ','c12  ','c13  ','n14  ','n15  ','o16  ','o17  ','o18  ',
       'f19  ','ne20 ','ne21 ','ne22 ','na23 ','mg24 ','mg25 ','mg26 ',
       'al27 ','si28 ','si29 ','si30 ','p31  ','s32  ','s33  ','s34  ',
       's36  ','cl35 ','cl37 ','ar36 ','ar38 ','ar40 ','k39  ','k40  ',
       'k41  ','ca40 ','ca42 ','ca43 ','ca44 ','ca46 ','ca48 ','sc45 ',
       'ti46 ','ti47 ','ti48 ','ti49 ','ti50 ','v50  ','v51  ','cr50 ',
       'cr52 ','cr53 ','cr54 ','mn55 ','fe54 ','fe56 ','fe57 ','fe58 ',
       'co59 ','ni58 ','ni60 ','ni61 ','ni62 ','ni64 ','cu63 ','cu65 ',
       'zn64 ','zn66 ','zn67 ','zn68 ','zn70 ','ga69 ','ga71 ','ge70 ',
       'ge72 ','ge73 ','ge74 ','ge76 ','as75 ','se74 ','se76 ','se77 ',
       'se78 ','se80 ','se82 ','br79 ','br81 ','kr78 ','kr80 ','kr82 ',
       'kr83 ','kr84 ','kr86 ','rb85 ','rb87 ','sr84 ','sr86 ','sr87 ',
       'sr88 ','y89  ','zr90 ','zr91 ','zr92 ','zr94 ','zr96 ','nb93 ',
       'mo92 ','mo94 ','mo95 ','mo96 ','mo97 ','mo98 ','mo100','ru96 ',
       'ru98 ','ru99 ','ru100','ru101','ru102','ru104','rh103','pd102',
       'pd104','pd105','pd106','pd108','pd110','ag107','ag109','cd106',
       'cd108','cd110','cd111','cd112','cd113','cd114','cd116','in113',
       'in115','sn112','sn114','sn115','sn116','sn117','sn118','sn119',
       'sn120','sn122','sn124','sb121','sb123','te120','te122','te123',
       'te124','te125','te126','te128','te130','i127 ','xe124','xe126',
       'xe128','xe129','xe130','xe131','xe132','xe134','xe136','cs133',
       'ba130','ba132','ba134','ba135','ba136','ba137','ba138','la138',
       'la139','ce136','ce138','ce140','ce142','pr141','nd142','nd143',
       'nd144','nd145','nd146','nd148','nd150','sm144','sm147','sm148',
       'sm149','sm150','sm152','sm154','eu151','eu153','gd152','gd154',
       'gd155','gd156','gd157','gd158','gd160','tb159','dy156','dy158',
       'dy160','dy161','dy162','dy163','dy164','ho165','er162','er164',
       'er166','er167','er168','er170','tm169','yb168','yb170','yb171',
       'yb172','yb173','yb174','yb176','lu175','lu176','hf174','hf176',
       'hf177','hf178','hf179','hf180','ta180','ta181','w180 ','w182 ',
       'w183 ','w184 ','w186 ','re185','re187','os184','os186','os187',
       'os188','os189','os190','os192','ir191','ir193','pt190','pt192',
       'pt194','pt195','pt196','pt198','au197','hg196','hg198','hg199',
       'hg200','hg201','hg202','hg204','tl203','tl205','pb204','pb206',
       'pb207','pb208','bi209','th232','u235 ','u238 ']

solarAbundances = [ # also ardcoded cuz pyMesaUtils for some reason screws up when trying to import arrays???
                    # abundances from Anders, Grevesse (1989)
             7.0573e-01, 4.8010e-05, 2.9291e-05, 2.7521e01, 6.4957e-10, 
             9.3490e-09, 1.6619e-10, 1.0674e-09, 4.7301e-09, 3.0324e-03, 
             3.6501e-05, 1.1049e-03, 4.3634e-06, 9.5918e-03, 3.8873e-06, 
             2.1673e-05, 4.0515e-07, 1.6189e-03, 4.1274e-06, 1.3022e-04, 
             3.3394e-05, 5.1480e-04, 6.7664e-05, 7.7605e-05, 5.8052e-05, 
             6.5301e-04, 3.4257e-05, 2.3524e-05, 8.1551e-06, 3.9581e-04, 
             3.2221e-06, 1.8663e-05, 9.3793e-08, 2.5320e-06, 8.5449e-07, 
             7.7402e-05, 1.5379e-05, 2.6307e-08, 3.4725e-06, 4.4519e-10, 
             2.6342e-07, 5.9898e-05, 4.1964e-07, 8.9734e-07, 1.4135e-06, 
             2.7926e-09, 1.3841e-07, 3.8929e-08, 2.2340e-07, 2.0805e-07, 
             2.1491e-06, 1.6361e-07, 1.6442e-07, 9.2579e-10, 3.7669e-07, 
             7.4240e-07, 1.4863e-05, 1.7160e-06, 4.3573e-07, 1.3286e-05, 
             7.1301e-05, 1.1686e-03, 2.8548e-05, 3.6971e-06, 3.3579e-06, 
             4.9441e-05, 1.9578e-05, 8.5944e-07, 2.7759e-06, 7.2687e-07, 
             5.7528e-07, 2.6471e-07, 9.9237e-07, 5.8765e-07, 8.7619e-08, 
             4.0593e-07, 1.3811e-08, 3.9619e-08, 2.7119e-08, 4.3204e-08, 
             5.9372e-08, 1.7136e-08, 8.1237e-08, 1.7840e-08, 1.2445e-08, 
             1.0295e-09, 1.0766e-08, 9.1542e-09, 2.9003e-08, 6.2529e-08, 
             1.1823e-08, 1.1950e-08, 1.2006e-08, 3.0187e-10, 2.0216e-09, 
             1.0682e-08, 1.0833e-08, 5.4607e-08, 1.7055e-08, 1.1008e-08, 
             4.3353e-09, 2.8047e-10, 5.0468e-09, 3.6091e-09, 4.3183e-08, 
             1.0446e-08, 1.3363e-08, 2.9463e-09, 4.5612e-09, 4.7079e-09, 
             7.7706e-10, 1.6420e-09, 8.7966e-10, 5.6114e-10, 9.7562e-10, 
             1.0320e-09, 5.9868e-10, 1.5245e-09, 6.2225e-10, 2.5012e-10, 
             8.6761e-11, 5.9099e-10, 5.9190e-10, 8.0731e-10, 1.5171e-09, 
             9.1547e-10, 8.9625e-10, 3.6637e-11, 4.0775e-10, 8.2335e-10, 
             1.0189e-09, 1.0053e-09, 4.5354e-10, 6.8205e-10, 6.4517e-10,
             5.3893e-11, 3.9065e-11, 5.5927e-10, 5.7839e-10, 1.0992e-09, 
             5.6309e-10, 1.3351e-09, 3.5504e-10, 2.2581e-11, 5.1197e-10, 
             1.0539e-10, 7.1802e-11, 3.9852e-11, 1.6285e-09, 8.6713e-10, 
             2.7609e-09, 9.8731e-10, 3.7639e-09, 5.4622e-10, 6.9318e-10, 
             5.4174e-10, 4.1069e-10, 1.3052e-11, 3.8266e-10, 1.3316e-10, 
             7.1827e-10, 1.0814e-09, 3.1553e-09, 4.9538e-09, 5.3600e-09, 
             2.8912e-09, 1.7910e-11, 1.6223e-11, 3.3349e-10, 4.1767e-09, 
             6.7411e-10, 3.3799e-09, 4.1403e-09, 1.5558e-09, 1.2832e-09, 
             1.2515e-09, 1.5652e-11, 1.5125e-11, 3.6946e-10, 1.0108e-09, 
             1.2144e-09, 1.7466e-09, 1.1240e-08, 1.3858e-12, 1.5681e-09, 
             7.4306e-12, 9.9136e-12, 3.5767e-09, 4.5258e-10, 5.9562e-10, 
             8.0817e-10, 3.6533e-10, 7.1757e-10, 2.5198e-10, 5.2441e-10, 
             1.7857e-10, 1.7719e-10, 2.9140e-11, 1.4390e-10, 1.0931e-10, 
             1.3417e-10, 7.2470e-11, 2.6491e-10, 2.2827e-10, 1.7761e-10, 
             1.9660e-10, 2.5376e-12, 2.8008e-11, 1.9133e-10, 2.6675e-10, 
             2.0492e-10, 3.2772e-10, 2.9180e-10, 2.8274e-10, 8.6812e-13, 
             1.4787e-12, 3.7315e-11, 3.0340e-10, 4.1387e-10, 4.0489e-10, 
             4.6047e-10, 3.7104e-10, 1.4342e-12, 1.6759e-11, 3.5397e-10, 
             2.4332e-10, 2.8557e-10, 1.6082e-10, 1.6159e-10, 1.3599e-12, 
             3.2509e-11, 1.5312e-10, 2.3624e-10, 1.7504e-10, 3.4682e-10, 
             1.4023e-10, 1.5803e-10, 4.2293e-12, 1.0783e-12, 3.4992e-11, 
             1.2581e-10, 1.8550e-10, 9.3272e-11, 2.4131e-10, 1.1292e-14, 
             9.4772e-11, 7.8768e-13, 1.6113e-10, 8.7950e-11, 1.8989e-10, 
             1.7878e-10, 9.0315e-11, 1.5326e-10, 5.6782e-13, 5.0342e-11, 
             5.1086e-11, 4.2704e-10, 5.2110e-10, 8.5547e-10, 1.3453e-09, 
             1.1933e-09, 2.0211e-09, 8.1702e-13, 5.0994e-11, 2.1641e-09, 
             2.2344e-09, 1.6757e-09, 4.8231e-10, 9.3184e-10, 2.3797e-12, 
             1.7079e-10, 2.8843e-10, 3.9764e-10, 2.2828e-10, 5.1607e-10, 
             1.2023e-10, 2.7882e-10, 6.7411e-10, 3.1529e-10, 3.1369e-09, 
             3.4034e-09, 9.6809e-09, 7.6127e-10, 1.9659e-10, 3.8519e-13, 
             5.3760e-11 ]
                                  
solarAbundancesDict = {name.rstrip() : abundance for name, abundance in zip(isotopeNames, solarAbundances)} # takes care of trailing spaces in names

allKnownChemicalIDs = {name.rstrip() : int(chem_def.get_nuclide_index(name)) for name in isotopeNames} # ditto

eosBasicResultsNum = eos_def.num_eos_basic_results.get() # how many things EOS returns

@dataclass
class EOSBledningInfo:
    frac_HELM: float
    frac_OPAL_SCVH: float
    frac_FreeEOS: float
    frac_PC: float
    frac_Skye: float
    frac_CMS: float
    frac_ideal: float


@dataclass
class EOSBasicResults:
    rho : float
    lnPgas : float
    lnE : float
    lnS : float
    mu : float
    lnfree_e : float
    eta : float
    chiRho : float
    chiT : float
    Cp : float
    Cv : float
    dE_dRho : float
    dS_dT : float
    dS_dRho : float
    gamma1 : float
    gamma3 : float
    phase : float
    latent_ddlnT : float
    latent_ddlnRho : float
    grad_ad : float


@dataclass
class EOSd_dTResults(EOSBasicResults):
    pass


@dataclass
class EOSd_dPOrRhoResults(EOSBasicResults):
    pass


@dataclass
class EOSFullResults:
    results: EOSBasicResults
    d_dT: EOSd_dTResults
    d_dPOrRho: EOSd_dPOrRhoResults
    blendInfo: EOSBledningInfo


# endregion


# region opacity stuff

kap_lib, kap_def = pym.loadMod("kap")
assert kap_def is not None
assert kap_lib is not None


num_kap_fracs = kap_def.num_kap_fracs
num_chem_isos = chem_def.num_chem_isos

if not hasattr(pym, 'mesa2Py'):
    atexit.register(L.info, "Shutting down kappa tables...")
    atexit.register(kap_lib.kap_shutdown)
    useCache = True
    kap_lib.kap_init(useCache, pym.KAP_CACHE, ierr)
    kap_handle = kap_lib.alloc_kap_handle(ierr)
    kap_lib.kap_setup_tables(kap_handle, ierr)
    kap_lib.kap_setup_hooks(kap_handle, ierr)
    atexit.register(L.info, "Deallocating kap_handle...")
    atexit.register(kap_lib.free_kap_handle, kap_handle)

@dataclass
class KappaBleningInfo:
    frac_lowT: float
    frac_highT: float
    frac_Type2: float
    frac_Compton: float
@dataclass
class KappaOutput:
    kappa: float
    dlnKappadlnRho: float
    dlnKappdlnT: float
    blendFractions: KappaBleningInfo


# endregion

if not hasattr(pym, 'mesa2Py'):
    if ierr != 0:
        L.critical(f"Mesa initialization failed with ierr {ierr}")
    else:
        L.info("Mesa initialized succesfully")

pym.mesa2Py = True #type: ignore FIXME very hacky way to make sure the fortran modules are only initialized once. It works tho

pass
