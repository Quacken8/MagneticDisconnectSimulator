#!/usr/bin/env python3
"""
fundamental constants and some used units for the code with their units and source in the comments
"""
hour = 60 * 60  # seconds in an hour
Mm = 1e6  # meters in megameter Mm
cm = 1e-2  # meters in a centimeter
gram = 1e-3  # kilgrams in a gram
Gauss = 1e-4  # Teslas in one Gauss
barye = 1e-1  # Pascals in one barye
erg = 1e1  # ergs in Joule
eV = 1.602176634e-19  # Joules in one eV [CODATA 2018]
BoltzmannConstant = 1.380649e-23  # J/K [CODATA 2018]
gasConstant = 8.314462618  # J/K/mol [CODATA 2018]
AH = 1.0078250322  # atomic weight of hydrogen atom [CIAAW 2000]
AHe = 4.002_603_2545  # atomic weight of helium atom [CIAAW 2000]
massFractionOfHydrogen = 0.6  # of solar matter
massFractionOfHelium = 0.4  # of solar matter
meanMolecularWeight = gram / (
    massFractionOfHydrogen / AH + massFractionOfHelium / AHe
)  # kg/mol of solar matter
mu0 = 1.25663706212e-6  # N/A^2 permeability of vacuum [CODATA 2018]
G = 6.67430e-11  # m^3/(kg s^2) gravitational constant [CODATA 2018]
solarLuminosity = 3.828e26  # W [IAU 2015]
speedOfLight = 299792458  # m/s [CODATA 2018]
SteffanBoltzmann = 5.670374419e-8  # W/(m^2 K^4) [CODATA 2018]
ionizationEnergyOfHydrogen = (
    13.59844 * eV
)  # TODO get better source [Lide, 1992, Ionization potentials of atoms and atomic ions]
L_sun = 3.828e26  # W [IAU 2015]
M_sun = 1.3271244e20 / G  # kg [IAU 2015]


solarAbundances = { # according to Anders, Grevesse (1989)
    "h1": 0.70573,
    "h2": 4.801e-05,
    "he3": 2.9291e-05,
    "he4": 27.521,
    "li6": 6.4957e-10,
    "li7": 9.349e-09,
    "be9": 1.6619e-10,
    "b10": 1.0674e-09,
    "b11": 4.7301e-09,
    "c12": 0.0030324,
    "c13": 3.6501e-05,
    "n14": 0.0011049,
    "n15": 4.3634e-06,
    "o16": 0.0095918,
    "o17": 3.8873e-06,
    "o18": 2.1673e-05,
    "f19": 4.0515e-07,
    "ne20": 0.0016189,
    "ne21": 4.1274e-06,
    "ne22": 0.00013022,
    "na23": 3.3394e-05,
    "mg24": 0.0005148,
    "mg25": 6.7664e-05,
    "mg26": 7.7605e-05,
    "al27": 5.8052e-05,
    "si28": 0.00065301,
    "si29": 3.4257e-05,
    "si30": 2.3524e-05,
    "p31": 8.1551e-06,
    "s32": 0.00039581,
    "s33": 3.2221e-06,
    "s34": 1.8663e-05,
    "s36": 9.3793e-08,
    "cl35": 2.532e-06,
    "cl37": 8.5449e-07,
    "ar36": 7.7402e-05,
    "ar38": 1.5379e-05,
    "ar40": 2.6307e-08,
    "k39": 3.4725e-06,
    "k40": 4.4519e-10,
    "k41": 2.6342e-07,
    "ca40": 5.9898e-05,
    "ca42": 4.1964e-07,
    "ca43": 8.9734e-07,
    "ca44": 1.4135e-06,
    "ca46": 2.7926e-09,
    "ca48": 1.3841e-07,
    "sc45": 3.8929e-08,
    "ti46": 2.234e-07,
    "ti47": 2.0805e-07,
    "ti48": 2.1491e-06,
    "ti49": 1.6361e-07,
    "ti50": 1.6442e-07,
    "v50": 9.2579e-10,
    "v51": 3.7669e-07,
    "cr50": 7.424e-07,
    "cr52": 1.4863e-05,
    "cr53": 1.716e-06,
    "cr54": 4.3573e-07,
    "mn55": 1.3286e-05,
    "fe54": 7.1301e-05,
    "fe56": 0.0011686,
    "fe57": 2.8548e-05,
    "fe58": 3.6971e-06,
    "co59": 3.3579e-06,
    "ni58": 4.9441e-05,
    "ni60": 1.9578e-05,
    "ni61": 8.5944e-07,
    "ni62": 2.7759e-06,
    "ni64": 7.2687e-07,
    "cu63": 5.7528e-07,
    "cu65": 2.6471e-07,
    "zn64": 9.9237e-07,
    "zn66": 5.8765e-07,
    "zn67": 8.7619e-08,
    "zn68": 4.0593e-07,
    "zn70": 1.3811e-08,
    "ga69": 3.9619e-08,
    "ga71": 2.7119e-08,
    "ge70": 4.3204e-08,
    "ge72": 5.9372e-08,
    "ge73": 1.7136e-08,
    "ge74": 8.1237e-08,
    "ge76": 1.784e-08,
    "as75": 1.2445e-08,
    "se74": 1.0295e-09,
    "se76": 1.0766e-08,
    "se77": 9.1542e-09,
    "se78": 2.9003e-08,
    "se80": 6.2529e-08,
    "se82": 1.1823e-08,
    "br79": 1.195e-08,
    "br81": 1.2006e-08,
    "kr78": 3.0187e-10,
    "kr80": 2.0216e-09,
    "kr82": 1.0682e-08,
    "kr83": 1.0833e-08,
    "kr84": 5.4607e-08,
    "kr86": 1.7055e-08,
    "rb85": 1.1008e-08,
    "rb87": 4.3353e-09,
    "sr84": 2.8047e-10,
    "sr86": 5.0468e-09,
    "sr87": 3.6091e-09,
    "sr88": 4.3183e-08,
    "y89": 1.0446e-08,
    "zr90": 1.3363e-08,
    "zr91": 2.9463e-09,
    "zr92": 4.5612e-09,
    "zr94": 4.7079e-09,
    "zr96": 7.7706e-10,
    "nb93": 1.642e-09,
    "mo92": 8.7966e-10,
    "mo94": 5.6114e-10,
    "mo95": 9.7562e-10,
    "mo96": 1.032e-09,
    "mo97": 5.9868e-10,
    "mo98": 1.5245e-09,
    "mo100": 6.2225e-10,
    "ru96": 2.5012e-10,
    "ru98": 8.6761e-11,
    "ru99": 5.9099e-10,
    "ru100": 5.919e-10,
    "ru101": 8.0731e-10,
    "ru102": 1.5171e-09,
    "ru104": 9.1547e-10,
    "rh103": 8.9625e-10,
    "pd102": 3.6637e-11,
    "pd104": 4.0775e-10,
    "pd105": 8.2335e-10,
    "pd106": 1.0189e-09,
    "pd108": 1.0053e-09,
    "pd110": 4.5354e-10,
    "ag107": 6.8205e-10,
    "ag109": 6.4517e-10,
    "cd106": 5.3893e-11,
    "cd108": 3.9065e-11,
    "cd110": 5.5927e-10,
    "cd111": 5.7839e-10,
    "cd112": 1.0992e-09,
    "cd113": 5.6309e-10,
    "cd114": 1.3351e-09,
    "cd116": 3.5504e-10,
    "in113": 2.2581e-11,
    "in115": 5.1197e-10,
    "sn112": 1.0539e-10,
    "sn114": 7.1802e-11,
    "sn115": 3.9852e-11,
    "sn116": 1.6285e-09,
    "sn117": 8.6713e-10,
    "sn118": 2.7609e-09,
    "sn119": 9.8731e-10,
    "sn120": 3.7639e-09,
    "sn122": 5.4622e-10,
    "sn124": 6.9318e-10,
    "sb121": 5.4174e-10,
    "sb123": 4.1069e-10,
    "te120": 1.3052e-11,
    "te122": 3.8266e-10,
    "te123": 1.3316e-10,
    "te124": 7.1827e-10,
    "te125": 1.0814e-09,
    "te126": 3.1553e-09,
    "te128": 4.9538e-09,
    "te130": 5.36e-09,
    "i127": 2.8912e-09,
    "xe124": 1.791e-11,
    "xe126": 1.6223e-11,
    "xe128": 3.3349e-10,
    "xe129": 4.1767e-09,
    "xe130": 6.7411e-10,
    "xe131": 3.3799e-09,
    "xe132": 4.1403e-09,
    "xe134": 1.5558e-09,
    "xe136": 1.2832e-09,
    "cs133": 1.2515e-09,
    "ba130": 1.5652e-11,
    "ba132": 1.5125e-11,
    "ba134": 3.6946e-10,
    "ba135": 1.0108e-09,
    "ba136": 1.2144e-09,
    "ba137": 1.7466e-09,
    "ba138": 1.124e-08,
    "la138": 1.3858e-12,
    "la139": 1.5681e-09,
    "ce136": 7.4306e-12,
    "ce138": 9.9136e-12,
    "ce140": 3.5767e-09,
    "ce142": 4.5258e-10,
    "pr141": 5.9562e-10,
    "nd142": 8.0817e-10,
    "nd143": 3.6533e-10,
    "nd144": 7.1757e-10,
    "nd145": 2.5198e-10,
    "nd146": 5.2441e-10,
    "nd148": 1.7857e-10,
    "nd150": 1.7719e-10,
    "sm144": 2.914e-11,
    "sm147": 1.439e-10,
    "sm148": 1.0931e-10,
    "sm149": 1.3417e-10,
    "sm150": 7.247e-11,
    "sm152": 2.6491e-10,
    "sm154": 2.2827e-10,
    "eu151": 1.7761e-10,
    "eu153": 1.966e-10,
    "gd152": 2.5376e-12,
    "gd154": 2.8008e-11,
    "gd155": 1.9133e-10,
    "gd156": 2.6675e-10,
    "gd157": 2.0492e-10,
    "gd158": 3.2772e-10,
    "gd160": 2.918e-10,
    "tb159": 2.8274e-10,
    "dy156": 8.6812e-13,
    "dy158": 1.4787e-12,
    "dy160": 3.7315e-11,
    "dy161": 3.034e-10,
    "dy162": 4.1387e-10,
    "dy163": 4.0489e-10,
    "dy164": 4.6047e-10,
    "ho165": 3.7104e-10,
    "er162": 1.4342e-12,
    "er164": 1.6759e-11,
    "er166": 3.5397e-10,
    "er167": 2.4332e-10,
    "er168": 2.8557e-10,
    "er170": 1.6082e-10,
    "tm169": 1.6159e-10,
    "yb168": 1.3599e-12,
    "yb170": 3.2509e-11,
    "yb171": 1.5312e-10,
    "yb172": 2.3624e-10,
    "yb173": 1.7504e-10,
    "yb174": 3.4682e-10,
    "yb176": 1.4023e-10,
    "lu175": 1.5803e-10,
    "lu176": 4.2293e-12,
    "hf174": 1.0783e-12,
    "hf176": 3.4992e-11,
    "hf177": 1.2581e-10,
    "hf178": 1.855e-10,
    "hf179": 9.3272e-11,
    "hf180": 2.4131e-10,
    "ta180": 1.1292e-14,
    "ta181": 9.4772e-11,
    "w180": 7.8768e-13,
    "w182": 1.6113e-10,
    "w183": 8.795e-11,
    "w184": 1.8989e-10,
    "w186": 1.7878e-10,
    "re185": 9.0315e-11,
    "re187": 1.5326e-10,
    "os184": 5.6782e-13,
    "os186": 5.0342e-11,
    "os187": 5.1086e-11,
    "os188": 4.2704e-10,
    "os189": 5.211e-10,
    "os190": 8.5547e-10,
    "os192": 1.3453e-09,
    "ir191": 1.1933e-09,
    "ir193": 2.0211e-09,
    "pt190": 8.1702e-13,
    "pt192": 5.0994e-11,
    "pt194": 2.1641e-09,
    "pt195": 2.2344e-09,
    "pt196": 1.6757e-09,
    "pt198": 4.8231e-10,
    "au197": 9.3184e-10,
    "hg196": 2.3797e-12,
    "hg198": 1.7079e-10,
    "hg199": 2.8843e-10,
    "hg200": 3.9764e-10,
    "hg201": 2.2828e-10,
    "hg202": 5.1607e-10,
    "hg204": 1.2023e-10,
    "tl203": 2.7882e-10,
    "tl205": 6.7411e-10,
    "pb204": 3.1529e-10,
    "pb206": 3.1369e-09,
    "pb207": 3.4034e-09,
    "pb208": 9.6809e-09,
    "bi209": 7.6127e-10,
    "th232": 1.9659e-10,
    "u235": 3.8519e-13,
    "u238": 5.376e-11,
}
