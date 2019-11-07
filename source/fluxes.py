import numpy as np
from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

def get_exp_flux(fluxfile):

	if (fluxfile == "fluxes/MiniBooNE_nu_mode_flux.dat"):
		Elo, Ehi, numu, numub, nue, nueb = np.loadtxt(fluxfile, unpack=True)
	 	Eminiboone = (Ehi+Elo)/2.0
		numu *= 1.0/0.05
		flux = scipy.interpolate.interp1d(Eminiboone, numu, fill_value=0.0, bounds_error=False)
		# print "Running MiniBooNE fluxes"
	return flux


	# if (fluxfile == "fluxes/MINERvANUbeamNumu.txt"):
	# 	E, numu = np.loadtxt(fluxfile, unpack=True)
	#  	numu *= (1e-4)*(1e-7)
	# 	mask = (E > Mn**2/2.0/MA + Mn)
	# 	flux = scipy.interpolate.interp1d(E, numu*mask, fill_value=0.0, bounds_error=False)
	# 	EMIN = 0.1
	# 	EMAX = 19
	# 	DET_SIZE = 2.5/2.0 # meters

	# 	# print "Running Minerva fluxes"
	# if (fluxfile == "fluxes/MINERVA_LE_numu_flux.dat"):
	# 	E, numu = np.loadtxt(fluxfile, unpack=True)
	#  	numu *=  (1e-4)*(1e-6)*2
	# 	mask = (E > Mn**2/2.0/MA + Mn)
	# 	flux = scipy.interpolate.interp1d(E, numu*mask, fill_value=0.0, bounds_error=False)
	# 	EMIN = 0.1
	# 	EMAX = 19
	# 	DET_SIZE = 2.5/2.0 # meters

	# 	# print "Running Minerva fluxes"
	# if (fluxfile == "fluxes/CHARMII.dat"):
	# 	E, numu = np.loadtxt(fluxfile, unpack=True)
	#  	numu *= 1.0/(370)**2 *1e-13
	# 	mask = (E > Mn**2/2.0/MA + Mn)
	# 	flux = scipy.interpolate.interp1d(E, numu*mask, fill_value=0.0, bounds_error=False)
	# 	EMIN = 1.5
	# 	EMAX = 198
	# 	DET_SIZE = 35.67/2.0 # meters

	# if (fluxfile == "MINIBOONE_CARLOS"):
	# 	Elo, Ehi, numu, numub, nue, nueb = np.loadtxt("fluxes/MiniBooNE_nu_mode_flux.dat", unpack=True)
	#  	Eminiboone = (Ehi+Elo)/2.0
	# 	numu *= 1.0/0.05
	# 	mask = (Eminiboone > Mn**2/2.0/MA + Mn)
	# 	flux = scipy.interpolate.interp1d(Eminiboone, numu*mask, fill_value=0.0, bounds_error=False)
	# 	EMIN = 0.05
	# 	EMAX = 9
	# 	DET_SIZE = 6.1 # meters

	# if (fluxfile == "MINIBOONE_ANTINEUTRINO_CARLOS"):
	# 	Elo, Ehi, numu, numub, nue, nueb = np.loadtxt("fluxes/MiniBooNE_nubar_mode_flux.dat", unpack=True)
	#  	Eminiboone = (Ehi+Elo)/2.0
	# 	numub *= 1.0/0.05
	# 	mask = (Eminiboone > Mn**2/2.0/MA + Mn)
	# 	flux = scipy.interpolate.interp1d(Eminiboone, numub*mask, fill_value=0.0, bounds_error=False)
	# 	EMIN = 0.05
	# 	EMAX = 9
	# 	DET_SIZE = 6.1 # meters

	# 	# print "Running MiniBooNE fluxes"
	# if (fluxfile == "UNIFORM"):
	# 	def flux(E):
	# 		return 1.0
	# 	EMIN = 0.05
	# 	EMAX = 9.0
	# 	DET_SIZE = 1e10 # meters
