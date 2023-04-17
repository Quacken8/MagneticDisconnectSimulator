import pyMesaUtils as pym
import constants as c
import numpy as np
import atexit


# region initialize opacity tables
eos_lib, eos_def = pym.loadMod("eos")
const_lib, const_def = pym.loadMod("const")
crlibm_lib, _ = pym.loadMod("math")
assert (
    eos_def is not None
)  # having these asserts is kinda not elegant, but for i havent found a prettier way
assert eos_lib is not None
assert const_def is not None
assert const_lib is not None
assert crlibm_lib is not None
crlibm_lib.math_init()
chem_lib, chem_def = pym.loadMod("chem")
kap_lib, kap_def = pym.loadMod("kap")
assert kap_def is not None
assert chem_def is not None
assert kap_lib is not None
assert chem_lib is not None

atexit.register(print, "Shutting down kappa tables...")
atexit.register(kap_lib.kap_shutdown)


ierr = 0

num_kap_fracs = kap_def.num_kap_fracs
num_chem_isos = chem_def.num_chem_isos

const_lib.const_init("", ierr)
chem_lib.chem_init("isotopes.data", ierr)

kap_lib.kap_init(False, pym.KAP_CACHE, ierr)

kap_handle = kap_lib.alloc_kap_handle(ierr)

kap_lib.kap_setup_tables(kap_handle, ierr)
kap_lib.kap_setup_hooks(kap_handle, ierr)


handle = kap_handle
atexit.register(print, "Deallocating kap_handle...")
atexit.register(kap_lib.free_kap_handle, kap_handle)

# endregion

# region base settings and mesa expected values

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
    """debugging"""
    from initialConditionsSetterUpper import loadModelS

    modelS = loadModelS()
    massFractions = {
        "h1": c.massFractionOfHydrogen,
        "he4" : c.massFractionOfHelium
    }

    modelSRhos = modelS.rhos
    modelSTs = modelS.temperatures
    modelSKappas = modelS.derivedQuantities["kappas"]

    mesaKappas = getMESAOpacity(modelSTs, modelSRhos, massFractions=massFractions)

    zs = modelS.zs
    
    import matplotlib.pyplot as plt
    plt.loglog(zs, modelSKappas, label="modelS")
    plt.loglog(zs, mesaKappas, label="mesa")
    plt.legend()
    plt.show()
