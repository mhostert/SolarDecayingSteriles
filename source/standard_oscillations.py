import numpy as np


import pdg
import const
import model

# see https://arxiv.org/pdf/hep-ph/0310238.pdf
def Padiabatic(Enu, channel):
	Enu = Enu*1e9
	ACC = 2*np.sqrt(2)*Enu*const.Gf*const.solar_core_Ne * 1e-18 # eV^2

	# ANTINEUTRINO
	if channel<0:
		ACC *= -1 
	theta12MATTER = 0.5 * np.arctan2( np.tan(2*const.theta12), (1.0 - ACC/const.dmSQR21/np.cos(2*const.theta12) )  )
	Pee = 0.5 + 0.5*np.cos(2*const.theta12) * np.cos(2*theta12MATTER)
	if np.abs(channel) == const.nue_to_nue:
		return np.cos(const.theta13)**4*Pee + np.sin(const.theta13)**4
	if np.abs(channel) == const.nue_to_numu:
		return 1 - np.cos(const.theta13)**4*Pee + np.sin(const.theta13)**4



def solarNe(x): # x in eV^-1 
	return const.solar_core_Ne*np.exp(-x/const.parkeSolarR)

def DsolarNeDX(x): # x in eV^-1 
	return 1.0/const.parkeSolarR

def P_Parke(Enu, channel):
	Enu = Enu*1e9
	ACC = 2*np.sqrt(2)*Enu*const.Gf*const.solar_core_Ne * 1e-18 # eV^2

	# ANTINEUTRINO
	if channel<0:
		ACC *= -1 
	theta12MATTER = 0.5 * np.arctan2( np.tan(2*const.theta12), 1.0 - ACC/const.dmSQR21/np.cos(2*const.theta12)  )

	F = 1 - np.tan(const.theta12)**2
	Res = np.log(ACC/const.dmSQR21/np.cos(2*const.theta12))*const.parkeSolarR  # in eV

	dNdx = np.abs(DsolarNeDX(Res))

	gamma = const.dmSQR21*np.sin(2*const.theta12)*np.sin(2*const.theta12)/2.0/Enu/np.cos(2*const.theta12)/dNdx
	
	Pc = (np.exp(-np.pi/2.0*gamma*F) - np.exp(-np.pi/2.0 * gamma * F / np.sin(const.theta12)/ np.sin(const.theta12))) 
	Pc /= (1 - np.exp(-np.pi/2.0* gamma * F / np.sin(const.theta12)/ np.sin(const.theta12)))

	Pee = 0.5 + (0.5 - Pc)*np.cos(2*const.theta12) * np.cos(2*theta12MATTER)
	if np.abs(channel) == const.nue_to_nue:
		return np.cos(const.theta13)**4*Pee + np.sin(const.theta13)**4
	if np.abs(channel) == const.nue_to_numu:
		return 1 - np.cos(const.theta13)**4*Pee + np.sin(const.theta13)**4




if __name__ == "__main__":
	import matplotlib 
	matplotlib.use('agg') 
	import matplotlib.pyplot as plt
	from matplotlib import rc, rcParams
	from matplotlib.pyplot import *
	from matplotlib.legend_handler import HandlerLine2D

	fsize=11
	rc('text', usetex=True)
	rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	rcParams.update(rcparams)
	axes_form  = [0.15,0.15,0.82,0.76]
	fig = plt.figure()
	ax = fig.add_axes(axes_form)

	E = np.logspace(-4,-2,100)

	ax.plot(E*1e3, Padiabatic(E,const.nue_to_nue), lw=1.0, label=r'$\nu_e \to \nu_e$')
	ax.plot(E*1e3, Padiabatic(E,-const.nue_to_nue), lw=1.0, label=r'$\overline{\nu_e} \to \overline{\nu_e}$')

	# ax.plot(E, P_Parke(E)/Padiabatic(E), lw=1.5, ls='--', label=r'Parke')


	##############
	# STYLE
	ax.set_xscale('log')
	ax.legend(loc='upper left',frameon=False,ncol=1)
	ax.set_xlabel(r'$E_\nu/$MeV')
	ax.set_ylabel(r'P')
	fig.savefig('../plots/Psolar.pdf')

