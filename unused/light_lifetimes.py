import numpy as np
import scipy

from source import *


def tauLAB(gphi, mi, mj, Enu, UsiUsj):
	gamma = gphi**2/16.0/np.pi*mi**2/Enu*UsiUsj**2*(1-(mj/mi)**2)**2
	return dr.tau_GeV_to_s(gamma)
def tau_over_m(gphi, mi, mj, UsiUsj):
	gamma = gphi**2/16.0/np.pi*mi*UsiUsj**2*(1-(mj/mi)**2)**2
	return dr.tau_GeV_to_s(gamma)/(mi*1e9)

m2=np.sqrt(const.dmSQR21)*1e-9 # GeV
m3=np.sqrt(const.dmSQR31)*1e-9 # GeV

U2 = np.sqrt(1e-5)

print('nu2 -> nu1 phi')
print(tauLAB(1,m2,0,10e-3,U2))
print(const.c_LIGHT*tauLAB(1,m2,0,10e-3,U2)/1.496e+11)
print(tau_over_m(1,m2,0,U2))

print('\n\nnu3 -> nu1 phi')
print(tauLAB(1,m3,0,10e-3,U2))
print(const.c_LIGHT*tauLAB(1,m3,0,10e-3,U2)/1.496e+11)
print(tau_over_m(1,m3,0,U2))
