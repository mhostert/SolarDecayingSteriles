import numpy as np
import scipy
import numpy.random

import const 

class miniboone_data():
	def __init__(self):

		#######################
		# neutrino energy data
		self.Enu_binc, self.data_MB_enu_nue = np.loadtxt("digitized/miniboone/Enu_excess_nue.dat", unpack=True)
		self.Enu_binc, self.data_MB_enu_nue_errorlow = np.loadtxt("digitized/miniboone/Enu_excess_nue_lowererror.dat", unpack=True)
		self.Enu_binc, self.data_MB_enu_nue_errorup = np.loadtxt("digitized/miniboone/Enu_excess_nue_uppererror.dat", unpack=True)
		
		self.binw_enu = np.array([0.1,0.075,0.1,0.075,0.125,0.125,0.15,0.15,0.2,0.2])
		self.bin_e = np.array([0.2,0.3,0.375,0.475,0.550,0.675,0.8,0.95,1.1,1.3,1.5])
		
		self.data_MB_enu_nue *= self.binw_enu*1e3
		self.data_MB_enu_nue_errorlow *= self.binw_enu*1e3
		self.data_MB_enu_nue_errorup *= self.binw_enu*1e3

		#######################
		# Angular data
		self.cost_binc, self.data_MB_cost_nue = np.loadtxt("digitized/miniboone/costheta_nu_data.dat", unpack=True)
		self.cost_binc, self.data_MB_cost_nue_errorlow = np.loadtxt("digitized/miniboone/costheta_nu_data_lowererror.dat", unpack=True)
		self.cost_binc, self.data_MB_cost_nue_errorup = np.loadtxt("digitized/miniboone/costheta_nu_data_uppererror.dat", unpack=True)
		self.cost_binc, self.data_MB_cost_nue_bkg = np.loadtxt("digitized/miniboone/costheta_nu_data_bkg.dat", unpack=True)
		
		self.binw_cost = np.ones(np.size(self.cost_binc))*0.2
		self.bincost_e = np.array([-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0])

		#########################
		# Efficiencies
		#########################
		# WATCH OUT I ADDED A POINT HERE
		########################
		self.eff = np.array([0.0,0.089,0.135,0.139,0.131,0.123,0.116,0.106,0.102,0.095,0.089,0.082,0.073,0.067,0.052,0.026,0.026])
		self.enu_eff = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.3,1.5,1.7,1.9,2.1,3.0])


class superk_data():
	def __init__(self):

		N_PROTONS = 1.5e33
		TIME = 2778**60*60*24# seconds
		avg_efficiency = 0.193
		self.norm = N_PROTONS*avg_efficiency*TIME

		self.exp_name=const.SUPERK_IV
		self.err_back=0.10

		self.smearing_function=superK_Esmear

		#######################
		# neutrino energy data
		self.Enu_binc, self.data = np.loadtxt("digitized/borexino/data.dat", unpack=True)
		
		_, self.MCatm = np.loadtxt("digitized/borexino/atmospheric.dat", unpack=True)
		_, self.MCgeo = np.loadtxt("digitized/borexino/geoneutrinos.dat", unpack=True)
		_, self.MCreactor = np.loadtxt("digitized/borexino/reactors.dat", unpack=True)

		self.bin_e = np.linspace(9.3,)
		self.bin_w = (self.bin_e[1:] - self.bin_e[:-1])
		self.bin_c = self.bin_e[:-1] + self.bin_w/2.0

class borexino_data():
	def __init__(self):

		N_PROTONS = 1.32e31
		avg_efficiency = 0.850
		exposure = 2485*60*60*24 # seconds
		self.norm = N_PROTONS*avg_efficiency*exposure

		self.exp_name=const.BOREXINO
		self.err_back=0.10

		self.smearing_function=borexino_Esmear

		#######################
		# neutrino energy data
		self.Enu_binc, self.data = np.loadtxt("digitized/borexino/data.dat", unpack=True)
		
		_, self.MCatm = np.loadtxt("digitized/borexino/atmospheric.dat", unpack=True)
		_, self.MCgeo = np.loadtxt("digitized/borexino/geoneutrinos.dat", unpack=True)
		_, self.MCreactor = np.loadtxt("digitized/borexino/reactors.dat", unpack=True)

		self.bin_e = np.array([1.8,2.8,3.8,4.8,5.8,6.8,7.8,8.8,9.8,10.8,11.8,12.8,13.8,14.8,15.8,16.8])
		self.bin_w = (self.bin_e[1:] - self.bin_e[:-1])
		self.bin_c = self.bin_e[:-1] + self.bin_w/2.0

class superK_IV_limit():
	def __init__(self):
		_, self.fluxlimit = np.loadtxt("digitized/superK/fluxlimits_superK_IV.dat", unpack=True)
		self.Enu_bin_e = np.linspace(9.3,31.3,23,endpoint=True)
		self.Enu_bin_w = (self.Enu_bin_e[1:] - self.Enu_bin_e[:-1])
		self.Enu_bin_c = self.Enu_bin_e[:-1] + self.Enu_bin_c/2.0

class borexino_limit():
	def __init__(self):
		self.Enu_bin_l, self.Nevents, self.Nbkg, self.NbkgATM, self.fluxlimit, self.fluxlimitATM = np.loadtxt("digitized/borexino/Table2.dat", unpack=True)
		self.Enu_bin_c = self.Enu_bin_l+0.5
		self.Enu_bin_w = np.ones(np.size(self.Enu_bin_c))
		self.Enu_bin_e = np.append(self.Enu_bin_l,self.Enu_bin_l[-1]+1.0)

class kamland_limit():
	def __init__(self):
		self.Enu_bin_l, self.fluxlimit = np.loadtxt("digitized/Kamland/Table4.dat", unpack=True)
		self.Enu_bin_c = self.Enu_bin_l+0.5
		self.Enu_bin_w = np.ones(np.size(self.Enu_bin_c))
		self.Enu_bin_e = np.append(self.Enu_bin_l,self.Enu_bin_l[-1]+1.0)


class kamland_data():
	def __init__(self):

		self.exp_name=const.KAMLAND
		self.err_back=0.20

		EXPOSURE=2343.0*24*60*60
		year=365*24*60*60
		fid_cut=(6.0/6.50)**3
		efficiency=0.92
		mass=1e9 # grams
		NA=6.022e23
		fraction_free = 0.145
		fudge = 0.88
		self.norm=EXPOSURE*fid_cut*efficiency*mass*NA*fraction_free*fudge
		# print EXPOSURE*fid_cut*efficiency*mass/year/1e9

		self.smearing_function=kamland_Esmear

		#######################
		# Ep energy data
		self.Enu_binc, self.data = np.loadtxt("digitized/Kamland/data.dat", unpack=True)
		
		self.bin_e = np.array([7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5])
		#####################
		# Neutrino energy
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
		Elin = np.linspace(np.min(self.bin_e),17.3,1000)
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




# E in MeV
def borexino_Esmear(E):
	Ep = E - 0.8
	return np.random.normal(Ep, 0.0556/np.sqrt(Ep))+0.8
def kamland_Esmear(E):
	Ep = E - 0.8
	return np.random.normal(Ep, 0.064/np.sqrt(Ep))+0.8