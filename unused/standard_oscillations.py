import numpy as np
import scipy


from source import *
from source.const import *

# see https://arxiv.org/pdf/hep-ph/0310238.pdf
def Padiabatic(Enu, channel):
	Enu = Enu*1e6
	ACC = 2*np.sqrt(2)*Enu*Gf*solar_core_Ne * 1e-18 # eV^2

	# ANTINEUTRINO
	if channel<0:
		ACC *= -1 
	theta12MATTER = 0.5 * np.arctan2( np.tan(2*theta12), (1.0 - ACC/dmSQR21/np.cos(2*theta12) )  )
	Pee = 0.5 + 0.5*np.cos(2*theta12) * np.cos(2*theta12MATTER)
	if np.abs(channel) == nue_to_nue:
		return np.cos(theta13)**4*Pee + np.sin(theta13)**4
	if np.abs(channel) == numu_to_nue:
		return 1 - np.cos(theta13)**4*Pee - np.sin(theta13)**4


a = np.genfromtxt('AGS09.dat')
r = a[:,0]
d = a[:,2]
f = a[:,8]
fraction = scipy.interpolate.interp1d(r,f,bounds_error=False,fill_value=0)
density = scipy.interpolate.interp1d(r,10**(d)*NAvo*cmINV_to_GeV**3*1e27,bounds_error=False,fill_value=0)
rnew = np.linspace(0,0.5,1000)
I = np.sum(fraction(rnew)*(rnew[1]-rnew[0])*density(rnew))


def Pmine_r(Enu, channel,Ue4SQR,Umu4SQR,r,i1=1,i2=1,i3=1):
	Enu = Enu*1e6

	ACC = 2*np.sqrt(2)*Enu*Gf*1e-18*density(r)#solar_core_Ne  # eV^2

	# ANTINEUTRINO
	if channel<0:
		ACC *= -1 
	theta12MATTER = 0.5 * np.arctan2( np.tan(2*theta12), (1.0 - ACC/dmSQR21/np.cos(2*theta12) )  )
	theta23MATTER = theta23# 0.5 * np.arctan2( np.tan(2*theta23), (1.0 - ACC/dmSQR21/np.cos(2*theta12) )  )

	Ue4  = np.sqrt(Ue4SQR)
	Umu4 = np.sqrt(Umu4SQR)
	
	s12MATTER = np.sin(theta12MATTER)
	c12MATTER = np.cos(theta12MATTER)
	
	Pall = i1*c13**2*c12**2*(Umu4*s12MATTER*c23+c12MATTER*(-Ue4*c13+Umu4*s13*s23) )**2 \
		\
		 + i2*c13**2*s12**2*(Umu4*c12MATTER*c23+s12MATTER*( Ue4*c13-Umu4*s13*s23) )**2 \
		\
		 + i3*s13**2*(Ue4*s13+Umu4*c13*s23)**2
	
	return (Pall)/(Umu4SQR+Ue4SQR)

def Pmine_avg(Enu, channel,Ue4SQR,Umu4SQR,i1=1,i2=1,i3=1):

	r=np.linspace(0,0.5,1000)
	dr = r[1]-r[0]

	P = np.array([ np.sum( fraction(r) * density(r) * Pmine_r(E, channel,Ue4SQR,Umu4SQR,r,i1=i1,i2=i2,i3=i3) * dr)/I for E in Enu])
	return P



def solarNe(x): # x in eV^-1 
	return solar_core_Ne*np.exp(-x/parkeSolarR)

def DsolarNeDX(x): # x in eV^-1 
	return 1.0/parkeSolarR

def P_Parke(Enu, channel):
	Enu = Enu*1e6
	ACC = 2*np.sqrt(2)*Enu*Gf*solar_core_Ne * 1e-18 # eV^2

	# ANTINEUTRINO
	if channel<0:
		ACC *= -1 
	theta12MATTER = 0.5 * np.arctan2( np.tan(2*theta12), 1.0 - ACC/dmSQR21/np.cos(2*theta12)  )

	F = 1 - np.tan(theta12)**2
	Res = np.log(ACC/dmSQR21/np.cos(2*theta12))*parkeSolarR  # in eV

	dNdx = np.abs(DsolarNeDX(Res))

	gamma = dmSQR21*np.sin(2*theta12)*np.sin(2*theta12)/2.0/Enu/np.cos(2*theta12)/dNdx
	
	Pc = (np.exp(-np.pi/2.0*gamma*F) - np.exp(-np.pi/2.0 * gamma * F / np.sin(theta12)/ np.sin(theta12))) 
	Pc /= (1 - np.exp(-np.pi/2.0* gamma * F / np.sin(theta12)/ np.sin(theta12)))

	Pee = 0.5 + (0.5 - Pc)*np.cos(2*theta12) * np.cos(2*theta12MATTER)
	if np.abs(channel) == nue_to_nue:
		return np.cos(theta13)**4*Pee + np.sin(theta13)**4
	if np.abs(channel) == numu_to_nue:
		return 1 - np.cos(theta13)**4*Pee - np.sin(theta13)**4




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

	E = np.logspace(-1,1,100)

	ax.plot(E, Padiabatic(E,nue_to_nue), lw=1.0, label=r'$\nu_e \to \nu_e$')
	ax.plot(E, Padiabatic(E,-nue_to_nue), lw=1.0, label=r'$\overline{\nu_e} \to \overline{\nu_e}$')

	# ax.plot(E, P_Parke(E)/Padiabatic(E), lw=1.5, ls='--', label=r'Parke')


	##############
	# STYLE
	ax.set_xscale('log')
	ax.legend(loc='upper left',frameon=False,ncol=1)
	ax.set_xlabel(r'$E_\nu/$MeV')
	ax.set_ylabel(r'P')
	fig.savefig('../plots/Psolar.pdf')

