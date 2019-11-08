import numpy as np
import scipy
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


class borexino_data():
	def __init__(self):

		#######################
		# neutrino energy data
		self.Enu_binc, self.data = np.loadtxt("digitized/borexino/data.dat", unpack=True)
		
		_, self.MCatm = np.loadtxt("digitized/borexino/atmospheric.dat", unpack=True)
		_, self.MCgeo = np.loadtxt("digitized/borexino/geoneutrinos.dat", unpack=True)
		_, self.MCreactor = np.loadtxt("digitized/borexino/reactors.dat", unpack=True)

		self.bin_e = np.array([1.8,2.8,3.8,4.8,5.8,6.8,7.8,8.8,9.8,10.8,11.8,12.8,13.8,14.8,15.8,16.8])
		self.bin_w = (self.bin_e[1:] - self.bin_e[:-1])
		self.bin_c = self.bin_e[:-1] + self.bin_w/2.0

class kamland_data():
	def __init__(self):

		#######################
		# neutrino energy data
		self.Enu_binc, self.data = np.loadtxt("digitized/Kamland/data.dat", unpack=True)
		


		self.bin_e = np.array([7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5])
		#####################
		# Neutrino energy
		self.bin_e = self.bin_e + 0.8

		self.bin_w = (self.bin_e[1:] - self.bin_e[:-1])
		self.bin_c = self.bin_e[:-1] + self.bin_w/2.0

		e, self.MCatm = np.loadtxt("digitized/Kamland/atmospheric.dat", unpack=True)
		f = scipy.interpolate.interp1d(e, self.MCatm, fill_value=0.0, bounds_error=False)
		self.MCatm = f(self.bin_c)
		
		e, self.MCreactor = np.loadtxt("digitized/Kamland/MCall_exceptReactors.dat", unpack=True)
		f = scipy.interpolate.interp1d(e, self.MCreactor, fill_value=0.0, bounds_error=False)
		self.MCreactor = f(self.bin_c)		
		
		e, self.MCreactor_spall = np.loadtxt("digitized/Kamland/MC_all_exceptReactorsANDspallation.dat", unpack=True)
		f = scipy.interpolate.interp1d(e, self.MCreactor_spall, fill_value=0.0, bounds_error=False)
		self.MCreactor_spall = f(self.bin_c)		
		
		e, self.MClimit = np.loadtxt("digitized/Kamland/solar_BG_limit.dat", unpack=True)
		f = scipy.interpolate.interp1d(e, self.MClimit, fill_value=0.0, bounds_error=False)
		self.MClimit = f(self.bin_c)