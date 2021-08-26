import numpy as np
import scipy
import numpy.random

from source import *

class superk_data():
	def __init__(self):

		N_PROTONS = 1.5e33 ## free protons in 22.5 kt of water
		TIME = 2970.1*60*60*24# seconds
		tot_efficiency = 0.043 # lowest value of efficiency
		self.norm = N_PROTONS*tot_efficiency*TIME

		self.exp_name=const.SUPERK_IV
		self.err_back=0.20
		self.err_flux=0.20

		self.smearing_function=superk_Esmear

		#######################
		# data
		_, self.data = np.loadtxt("digitized/superk-IV-article/data.dat", unpack=True)
		
		_, self.MCall = np.loadtxt("digitized/superk-IV-article/MCall.dat", unpack=True)
		_, self.MCallbkg = np.loadtxt("digitized/superk-IV-article/MC_except_signal.dat", unpack=True)
		_, self.MC_minus_acc = np.loadtxt("digitized/superk-IV-article/MC_except_signal_acc.dat", unpack=True)
		_, self.MC_minus_acc_reactors = np.loadtxt("digitized/superk-IV-article/MC_except_signal_acc_reactors.dat", unpack=True)

		self.MCatm = self.MC_minus_acc_reactors
		self.MCreactor = self.MC_minus_acc - self.MC_minus_acc_reactors
		self.MCacc = self.MCall - self.MC_minus_acc

		self.bin_e = np.linspace(7.5+0.8,29.5+0.8,12,endpoint=True)
		self.bin_w = (self.bin_e[1:] - self.bin_e[:-1])
		self.bin_c = self.bin_e[:-1] + self.bin_w/2.0

		self.fit_endpoint = 15.5+0.8
		self.fit_startpoint = 7.5+0.8

class superk_outdated_data():
	def __init__(self):

		N_PROTONS = 1.5e33 ## free protons in 22.5 kt of water
		TIME = 2778*60*60*24# seconds
		tot_efficiency = 0.193*0.354*0.50 # conservative estimate of efficiency -- L. Wan
		self.norm = N_PROTONS*tot_efficiency*TIME

		self.exp_name=const.SUPERK_IV_DEPRECATED
		self.err_back=0.10
		self.err_flux=0.10

		self.smearing_function=superk_Esmear

		#######################
		# data
		_, self.data = np.loadtxt("digitized/superk/data.dat", unpack=True)
		
		_, self.MCall = np.loadtxt("digitized/superk/MCall.dat", unpack=True)
		_, self.MCaccidental = np.loadtxt("digitized/superk/MC_only_accidental.dat", unpack=True)
		_, self.MCreactor = np.loadtxt("digitized/superk/MC_except_reactor.dat", unpack=True)
		_, self.MCreactorLi = np.loadtxt("digitized/superk/MC_except_Li_reactor.dat", unpack=True)

		self.bin_e = np.linspace(9.3,31.3,23,endpoint=True)
		self.bin_w = (self.bin_e[1:] - self.bin_e[:-1])
		self.bin_c = self.bin_e[:-1] + self.bin_w/2.0
		
		self.fit_endpoint = 16.5+0.8
		self.fit_startpoint = 7.5+0.8

class borexino_data():
	def __init__(self):

		N_PROTONS = 1.32e31
		avg_efficiency = 0.850
		exposure = 2485*60*60*24 # seconds
		self.norm = N_PROTONS*avg_efficiency*exposure

		self.exp_name=const.BOREXINO
		self.err_back=0.10
		self.err_flux=0.10

		self.smearing_function=borexino_Esmear

		#######################
		# neutrino energy from digitization of the data
		self.Enu_binc, self.data = np.loadtxt("digitized/borexino/data.dat", unpack=True)
		
		_, self.MCatm = np.loadtxt("digitized/borexino/atmospheric.dat", unpack=True)
		_, self.MCgeo = np.loadtxt("digitized/borexino/geoneutrinos.dat", unpack=True)
		_, self.MCreactor = np.loadtxt("digitized/borexino/reactors.dat", unpack=True)
		self.MCall = self.MCatm+self.MCreactor+self.MCgeo

		self.bin_e = np.array([1.8,2.8,3.8,4.8,5.8,6.8,7.8,8.8,9.8,10.8,11.8,12.8,13.8,14.8,15.8,16.8])
		self.bin_w = (self.bin_e[1:] - self.bin_e[:-1])
		self.bin_c = self.bin_e[:-1] + self.bin_w/2.0

		
		self.fit_endpoint = 15+0.8
		self.fit_startpoint = 1+0.8

class kamland_data():
	def __init__(self):

		self.exp_name=const.KAMLAND
		self.err_back=0.10
		self.err_flux=0.10

		EXPOSURE=2343.0*24*60*60
		fid_cut=(6.0/6.50)**3
		efficiency=0.92
		mass=1e9 # grams
		NA=6.022e23
		fraction_free = 0.145
		fudge = 0.88
		self.norm = EXPOSURE*fid_cut*efficiency*mass*NA*fraction_free*fudge

		self.smearing_function = kamland_Esmear

		#######################
		# Ep energy data
		self.Enu_binc, self.data = np.loadtxt("digitized/Kamland/data.dat", unpack=True)
		
		self.bin_e = np.array([7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5])
		#####################
		# neutrino energy bins
		self.bin_e = self.bin_e + 0.8

		self.bin_w = (self.bin_e[1:] - self.bin_e[:-1])
		self.bin_c = self.bin_e[:-1] + self.bin_w/2.0

		e, self.MCall = np.loadtxt("digitized/Kamland/MCall.dat", unpack=True)
		f = scipy.interpolate.interp1d(e+0.8, self.MCall, fill_value=0.0, bounds_error=False)
		self.MCall = f
		
		e, self.MCreactor = np.loadtxt("digitized/Kamland/MCall_exceptReactors.dat", unpack=True)
		f = scipy.interpolate.interp1d(e+0.8, self.MCreactor, fill_value=0.0, bounds_error=False)
		self.MCreactor = f
		
		e, self.MCreactor_spall = np.loadtxt("digitized/Kamland/MC_all_exceptReactorsANDspallation.dat", unpack=True)
		f = scipy.interpolate.interp1d(e+0.8, self.MCreactor_spall, fill_value=0.0, bounds_error=False)
		self.MCreactor_spall = f	
		
		e, self.MClimit = np.loadtxt("digitized/Kamland/solar_BG_limit.dat", unpack=True)
		f = scipy.interpolate.interp1d(e+0.8, self.MClimit, fill_value=0.0, bounds_error=False)
		self.MClimit = f


		######################
		# Binning Montecarlo 
		Elin = np.linspace(np.min(self.bin_e),17.31,1000)
		dxlin = Elin[1]-Elin[0]

		MCall = self.MCall(Elin)
		MCreactor = self.MCreactor(Elin)
		MCreactor_spall = self.MCreactor_spall(Elin)
		MClimit = self.MClimit(Elin)

		self.MCall_binned = np.zeros(np.size(self.bin_e)-1)
		self.MCreactor_binned = np.zeros(np.size(self.bin_e)-1)
		self.MCreactor_spall_binned = np.zeros(np.size(self.bin_e)-1)
		self.MClimit_binned = np.zeros(np.size(self.bin_e)-1)
		for i in range(0,np.size(self.bin_e)-1):
			self.MCall_binned[i] = np.sum( MCall[ (Elin<self.bin_e[i+1]) & (Elin>self.bin_e[i]) ]*dxlin ) 
			self.MCreactor_binned[i] = np.sum( MCreactor[ (Elin<self.bin_e[i+1]) & (Elin>self.bin_e[i]) ]*dxlin ) 
			self.MCreactor_spall_binned[i] = np.sum( MCreactor_spall[ (Elin<self.bin_e[i+1]) & (Elin>self.bin_e[i]) ]*dxlin ) 
			self.MClimit_binned[i] = np.sum( MClimit[ (Elin<self.bin_e[i+1]) & (Elin>self.bin_e[i]) ]*dxlin ) 

		self.fit_startpoint = 7.5+0.8
		self.fit_endpoint   = 17.31

class kamland21_data():
	def __init__(self):

		self.exp_name=const.KAMLAND21
		self.err_back=0.10
		self.err_flux=0.10

		EXPOSURE=4528.5*24*60*60 # s
		efficiency=0.73 # with fid volume cut
		nprotons=4.6e31
		self.norm = EXPOSURE*nprotons*efficiency

		self.smearing_function = kamland_Esmear

		#######################
		# Ep energy data
		Enu_binc, data = np.loadtxt("digitized/kamland_21/data.dat", unpack=True)
		# combine data
		self.comb_data = data[::2][:-1]+data[1::2]
		self.data = self.comb_data[:-1]

		self.bin_e = np.array([7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5])
		#####################
		# neutrino energy bins
		self.bin_e = self.bin_e[::2][:-1] + 0.8

		self.bin_w = (self.bin_e[1:] - self.bin_e[:-1])
		self.bin_c = self.bin_e[:-1] + self.bin_w/2.0

		e, self.MCall = np.loadtxt("digitized/kamland_21/MCall.dat", unpack=True)
		self.MCall = scipy.interpolate.interp1d(e+0.8, self.MCall, fill_value=0.0, bounds_error=False)
		
		self.fit_startpoint = 7.5+0.8
		self.fit_endpoint   = 17.31+1

		######################
		# Binning Montecarlo 
		Elin = np.linspace(self.fit_startpoint,self.fit_endpoint,1000)
		dxlin = Elin[1]-Elin[0]

		MCall = self.MCall(Elin)

		self.MCall_binned = np.zeros(np.size(self.bin_e)-1)
		for i in range(0,np.size(self.bin_e)-1):
			self.MCall_binned[i] = np.sum( MCall[ (Elin<self.bin_e[i+1]) & (Elin>self.bin_e[i]) ]*dxlin ) 

################
# information on the published limits on nubar flux as a function of Enu 
class superk_limit():
	def __init__(self):
		_, self.fluxlimit = np.loadtxt("digitized/superk/fluxlimits_superK_IV.dat", unpack=True)
		self.Enu_bin_e = np.linspace(9.3,31.3,23,endpoint=True)
		self.Enu_bin_w = (self.Enu_bin_e[1:] - self.Enu_bin_e[:-1])
		self.Enu_bin_c = self.Enu_bin_e[:-1] + self.Enu_bin_w/2.0

class borexino_limit():
	def __init__(self):
		self.Enu_bin_l, self.Nevents, self.Nbkg, self.NbkgATM, self.fluxlimit, self.fluxlimitATM = np.loadtxt("digitized/borexino/Table2.dat", unpack=True)
		self.Enu_bin_e = np.append(self.Enu_bin_l,self.Enu_bin_l[-1]+1.0)
		self.Enu_bin_c = self.Enu_bin_l+0.5
		self.Enu_bin_w = np.ones(np.size(self.Enu_bin_c))

class kamland_limit():
	def __init__(self):
		self.Enu_bin_l, self.fluxlimit = np.loadtxt("digitized/Kamland/Table4.dat", unpack=True)
		self.Enu_bin_c = self.Enu_bin_l+0.5
		self.Enu_bin_w = np.ones(np.size(self.Enu_bin_c))
		self.Enu_bin_e = np.append(self.Enu_bin_l,self.Enu_bin_l[-1]+1.0)

class kamland21_limit():
	def __init__(self):
		self.Enu_bin_c, self.fluxlimit = np.loadtxt("digitized/kamland_21/fluxlimit.dat", unpack=True)
		self.Enu_bin_w = np.ones(np.size(self.Enu_bin_c))
		self.Enu_bin_e = np.append(self.Enu_bin_c-0.5,self.Enu_bin_c[-1]+0.5)

class superk21_limit():
	def __init__(self):
		self.Enu_bin_c, self.fluxlimit = np.loadtxt("digitized/kamland_21/fluxlimit_SK21.dat", unpack=True)
		self.Enu_bin_w = np.ones(np.size(self.Enu_bin_c))
		self.Enu_bin_e = np.append(self.Enu_bin_c-1,self.Enu_bin_c[-1]+1)

class superk15_limit():
	def __init__(self):
		self.Enu_bin_c, self.fluxlimit = np.loadtxt("digitized/kamland_21/fluxlimit_SK15.dat", unpack=True)
		self.Enu_bin_w = np.ones(np.size(self.Enu_bin_c))
		self.Enu_bin_e = np.append(self.Enu_bin_c-0.5,self.Enu_bin_c[-1]+0.5)


# E in MeV
def borexino_Esmear(E):
	Ep = E - 0.8
	return np.random.normal(Ep, 0.0556/np.sqrt(Ep))+0.8
def kamland_Esmear(E):
	Ep = E - 0.8
	return np.random.normal(Ep, 0.064/np.sqrt(Ep))+0.8
def superk_Esmear(E):
	Ep = E - 0.8
	return np.random.normal(Ep, 0.20/np.sqrt(Ep))+0.8
