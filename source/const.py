import numpy as np
import scipy
from scipy import interpolate

# from source import *


#Avogadro's number
NAvo = 6.022*1e23
# from GeV^-2 to cm^2
GeV2_to_cm2 = 3.9204e-28
cmINV_to_GeV = 1.973e-14
GeVINV_to_cm = 1.0/1.973e-14
m_to_eVINV = 1.0/1.973e-7

# speed of light (PDG) m/s
c_LIGHT = 299792458

## FLAGS
electron = 1
muon = 2
tau = 3

nue_to_nue = 1
numu_to_nue = 2
nutau_to_nue = 3

def is_antineutrino(channel):
	return channel < 0

VECTOR = 1
SCALAR = 2

# Experiments 
KAMLAND21     = "kamland21"
KAMLAND     = "kamland"
BOREXINO     = "borexino"
SUPERK_IV     = "SUPERK_IV"
SUPERK_IV_DEPRECATED     = "SUPERK_IV_DEPRECATED"


###########
# Solar properties

# https://arxiv.org/pdf/1611.09867.pdf
B8FLUX_B16 = 5.46*1e6 # /cm^2/s

# https://wwwmpa.mpa-garching.mpg.de/~aldos/SSM/AGSS09/nu_fluxes.dat
B8FLUX_AGSS09 = 5.88*1e6 # /cm^2/s

# C. Pena-Garay and A. Serenelli,  (2008), arXiv:0811.2424 [astro-ph].
B8FLUX_GS98 = 5.94*1e6 # /cm^2/s

solar_core_Ne = 102*NAvo*cmINV_to_GeV**3 * 1e27# ev^3
solarR = 6.955e8*m_to_eVINV # eVinv


###########
# IBD cross section
IBD_THRESHOLD=1.8 # MeV
Enu_BEG_OF_SPECTRUM = IBD_THRESHOLD
Enu_END_OF_SPECTRUM = 17.0


###########
# 3 neutrino properties 
# Normal Ordering
theta12 = 0.583996
theta13 = 0.148190
theta23 = 0.737324
dmSQR21 = 7.5e-5
dmSQR31 = 2.57e-3

c13 = np.cos(theta13)
s13 = np.sin(theta13)
c12 = np.cos(theta12)
s12 = np.sin(theta12)
c23 = np.cos(theta23)
s23 = np.sin(theta23)

## MASSES in GeV
Me  = 511e-6 
Mmu = 0.105
Mtau = 1.777 
mproton = 0.938
mneutron = 0.939
MAVG = (mproton + mneutron)/2.0
Mw = 80.35 
Mz = 91
higgsvev = 246.22 

## SM COUPLINGS 
s2w = 0.231
sw = np.sqrt(0.231)
cw = np.sqrt(1.0 - s2w)
gl_lepton = -0.5 + s2w
gr_lepton = s2w

eQED = np.sqrt(4.0*np.pi/137.0)
alphaQED = 1./137.0359991

gvP = 1.0
Gf = 1.16e-5 # GeV^-2
g = np.sqrt(Gf*8/np.sqrt(2)*Mw*Mw)

#########
# CKM elements
Vud = 0.97420
Vus = 0.2243
Vcd = 0.218
Vcs = 0.997
Vcb = 42.2e-3
Vub = 3.94e-3
Vtd = 8.1e-3 
Vts = 39.4e-3
Vtb = 1

#########
# safe_momentum
def momentum(E,m):
	p = E*np.sqrt(1.0 - (m/E)**2)
	return np.where(p==p,p,E)

#########
# auxiliary
def Heaviside(x):
    return 1 * (x > 0)

def get_centers(bins):
	return (bins[:-1]+bins[1:])/2

def get_avg_in_bins(bin_edges, func):
    # integrate within bin for every bin
    avg_f = []
    for bl, br in zip(bin_edges[:-1], bin_edges[1:]):
        x=np.linspace(bl, br, 100)
        dx = x[1]-x[0]
        integral = np.sum(func(x)*dx)/(br-bl)
        avg_f.append(integral)
    return np.array(avg_f)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_arg(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx