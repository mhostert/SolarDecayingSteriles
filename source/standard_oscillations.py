import numpy as np

import pdg
import const
import model

# see https://arxiv.org/pdf/hep-ph/0310238.pdf
def Padiabatic(Enu):
	ACC = 2*np.sqrt(2)*Enu*const.Gf*const.solar_core_Ne*const.cmINV_to_GeV**3 * 1e+18 # eV^2
	theta12MATTER = 0.5 * np.arctan( np.tan(2*const.theta12)/(1.0 - ACC/const.dmSQR21/np.cos(2*const.theta12) )  )
	Pee = 0.5 + 0.5*np.cos(2*const.theta12) * np.cos(2*theta12MATTER)
	return Pee

def solarNe(x): # x in meters 
	return const.solar_core_Ne*np.exp(-x/const.parkeSolarR)
def DsolarNeDX(x): # x in meters 
	return const.solar_core_Ne*np.exp(-x/const.parkeSolarR)/const.parkeSolarR

def P_Parke(Enu):
	ACC = 2*np.sqrt(2)*Enu*const.Gf*const.solar_core_Ne*const.cmINV_to_GeV**3 * 1e+18 # eV^2
	theta12MATTER = 0.5 * np.arctan( np.tan(2*const.theta12)/(1.0 - ACC/const.dmSQR21/np.cos(2*const.theta12) )  )

	F = 1 - np.tan(const.theta12)**2
	Res = np.log(ACC/const.dmSQR21/np.cos(2*const.theta12))*const.parkeSolarR # in meters

	Pee = 0.5 + 0.5*np.cos(2*const.theta12) * np.cos(2*theta12MATTER)
	return Pee

E = np.logspace(-4,-2)
print Padiabatic(E)