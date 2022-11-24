from numpy import genfromtxt, zeros, pi
from scipy.interpolate import UnivariateSpline
from scipy.constants import G


z = genfromtxt("model_s/model_S_raw.dat", usecols = (0))
rho = genfromtxt("model_s/model_S_raw.dat", usecols = (3))

arr_size = z.size
mass = zeros(arr_size)
mass[-1] = 0
r = z.max()-z

for i in range (0,arr_size-1)[::-1]:
    # hmotnost az do polomeru r+dr je hmotnost do r + hmotnost slupky (r,r+dr)
    mass[i-1] = mass[i]+4*pi*rho[i]*(r[i]*r[i])*(r[i-1]-r[i])

gravity = zeros(arr_size)

for i in range (0,arr_size-1):
    gravity[i] = G*mass[i]/(r[i]*r[i])

g = UnivariateSpline(z,gravity,s=0,k=1) # linearni interpolace
