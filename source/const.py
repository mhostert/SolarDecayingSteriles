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

VECTOR = 1
SCALAR = 2

# Experiments 
KAMLAND     = "kamland"
BOREXINO     = "borexino"
SUPERK_IV     = "SUPERK_IV"


###########
# Solar properties
## from C. Pena-Garay and A. Serenelli,  (2008), arXiv:0811.2424 [astro-ph].
B8FLUX = 5.94*1e6 # /cm^2/s
B8FLUX_HM = 5.94*1e6 # /cm^2/s
B8FLUX_LM = 5.94*1e6 # /cm^2/s

IBD_THRESHOLD=1.8 # MeV
Enu_BEG_OF_SPECTRUM = IBD_THRESHOLD
Enu_END_OF_SPECTRUM = 17.0

solar_core_Ne = 102*NAvo*cmINV_to_GeV**3 * 1e27# ev^3
solarR = 6.955e8*m_to_eVINV # eVinv
parkeSolarR = solarR/10.54 # eVinv

# Normal Ordering
theta12 = 0.583996
theta13 = 0.148190
theta23 = 0.737324
dmSQR21 = 7.5e-5
dmSQR31 = 2.57e-3


## MASSES in GeV
higgsvev = 246 # GeV
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

################

def momentum(E,m):
	################
	## FIX ME -- hack to overcome small masses in this application
	p = E*np.sqrt(1.0 - (m/E)**2)
	return np.where(p==p,p,E)

def Heaviside(x):
    return 1 * (x > 0)
