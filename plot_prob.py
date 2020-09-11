import numpy as np
import scipy

from source import *



# see https://arxiv.org/pdf/hep-ph/0310238.pdf
def Padiabatic(Enu, channel):
	Enu = Enu*1e6
	ACC = 2*np.sqrt(2)*Enu*const.Gf*const.solar_core_Ne * 1e-18 # eV^2

	# ANTINEUTRINO
	if channel<0:
		ACC *= -1 
	theta12MATTER = 0.5 * np.arctan2( np.tan(2*const.theta12), (1.0 - ACC/const.dmSQR21/np.cos(2*const.theta12) )  )
	Pee = 0.5 + 0.5*np.cos(2*const.theta12) * np.cos(2*theta12MATTER)
	if np.abs(channel) == const.nue_to_nue:
		return np.cos(const.theta13)**4*Pee + np.sin(const.theta13)**4
	if np.abs(channel) == const.numu_to_nue:
		return 1 - np.cos(const.theta13)**4*Pee - np.sin(const.theta13)**4


a = np.genfromtxt('AGS09.dat')
r = a[:,0]
d = a[:,2]
f = a[:,8]
# print(10**(d))
fraction = scipy.interpolate.interp1d(r,f,bounds_error=False,fill_value=0)
density = scipy.interpolate.interp1d(r,10**(d)*const.NAvo*const.cmINV_to_GeV**3*1e27,bounds_error=False,fill_value=0)
rnew = np.linspace(0,0.5,1000)
I = np.sum(fraction(rnew)*(rnew[1]-rnew[0])*density(rnew))


def Pmine_r(Enu, channel,Ue4SQR,Umu4SQR,r,i1=1,i2=1,i3=1):
	Enu = Enu*1e6

	ACC = 2*np.sqrt(2)*Enu*const.Gf*1e-18*density(r)#const.solar_core_Ne  # eV^2
	# print((ACC/2/Enu).mean())
	# ANTINEUTRINO
	if channel<0:
		ACC *= -1 
	theta12MATTER = 0.5 * np.arctan2( np.tan(2*const.theta12), (1.0 - ACC/const.dmSQR21/np.cos(2*const.theta12) )  )
	theta23MATTER = const.theta23# 0.5 * np.arctan2( np.tan(2*const.theta23), (1.0 - ACC/const.dmSQR21/np.cos(2*const.theta12) )  )


	Pee = (0.5 + 0.5*np.cos(2*const.theta12) * np.cos(2*theta12MATTER))*Ue4SQR
	Pmue = np.cos(2*const.theta12) * np.sin(2*theta12MATTER) *  np.cos(theta23MATTER)*np.sqrt(Umu4SQR*Ue4SQR)
	Pmumu = (0.5 - 0.5*np.cos(2*const.theta12) * np.cos(2*theta12MATTER) )* np.cos(theta23MATTER)**2 *Umu4SQR

	Ue4=np.sqrt(Ue4SQR)
	Umu4=np.sqrt(Umu4SQR)
	Pall =  i2*np.cos(const.theta13)**2*np.sin(const.theta12)**2*(Umu4*np.cos(theta12MATTER)*np.cos(const.theta23)
				+np.sin(theta12MATTER)*(Ue4*np.cos(const.theta13)-Umu4*np.sin(const.theta13)*np.sin(const.theta23) ) )**2 \
		+ i1*np.cos(const.theta13)**2*np.cos(const.theta12)**2*(Umu4*np.sin(theta12MATTER)*np.cos(const.theta23)
				+np.cos(theta12MATTER)*(-Ue4*np.cos(const.theta13)+Umu4*np.sin(const.theta13)*np.sin(const.theta23) ) )**2 \
		+ i3*np.sin(const.theta13)**2*(Ue4*np.sin(const.theta13)+Umu4*np.cos(const.theta13)*np.sin(const.theta23))**2
	return (Pall)/(Umu4SQR+Ue4SQR)

def Pmine(Enu, channel,Ue4SQR,Umu4SQR,i1=1,i2=1,i3=1):

	r=np.linspace(0,0.5,1000)
	dr = r[1]-r[0]

	P = np.array([ np.sum(fraction(r)*density(r)*Pmine_r(E, channel,Ue4SQR,Umu4SQR,r,i1=i1,i2=i2,i3=i3) * dr)/I for E in Enu])
	return P



def solarNe(x): # x in eV^-1 
	return const.solar_core_Ne*np.exp(-x/const.parkeSolarR)

def DsolarNeDX(x): # x in eV^-1 
	return 1.0/const.parkeSolarR

def P_Parke(Enu, channel):
	Enu = Enu*1e6
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
	if np.abs(channel) == const.numu_to_nue:
		return 1 - np.cos(const.theta13)**4*Pee - np.sin(const.theta13)**4




if __name__ == "__main__":
	import matplotlib 
	matplotlib.use('agg') 
	import matplotlib.pyplot as plt
	from matplotlib import rc, rcParams
	from matplotlib.pyplot import *
	from matplotlib.legend_handler import HandlerLine2D

	fsize=14
	rc('text', usetex=True)
	rcparams={'axes.labelsize':9,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,6)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	matplotlib.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth

	rcParams.update(rcparams)
	axes_form1  = [0.16,0.13+0.46,0.82,0.35]
	axes_form2  = [0.16,0.11,0.82,0.35]
	fig = plt.figure()
	ax = fig.add_axes(axes_form1)
	ax2 = fig.add_axes(axes_form2)

	E = np.logspace(np.log10(0.1),np.log10(50),100)

	uesqr=0.0
	umusqr=0.01

	# ax.plot(E, uesqr*Padiabatic(E,const.nue_to_nue)+umusqr*Padiabatic(E,const.numu_to_nue), lw=1.0, color='red', label=r'$\nu_e \to \nu_e$')
	# ax.plot(E, Pmine(E,const.nue_to_nue,uesqr,umusqr), lw=1.0, color='red', dashes=(6,0), label=r'$\hat{\nu}_s \to \nu_e$')
	# ax.plot(E, (uesqr*flavour_transitions.Psolar(E,const.nue_to_nue)+umusqr*flavour_transitions.Psolar(E,const.numu_to_nue))/(uesqr+umusqr), lw=1.0, color='red', dashes=(1,1),label=r'$\nu_e \to \nu_e$')
	
	# ax.plot(E, uesqr*Padiabatic(E,-const.nue_to_nue)+umusqr*Padiabatic(E,-const.numu_to_nue), lw=1.0, color='blue', label=r'$\overline{\nu_e} \to \overline{\nu_e}$')
	# for umusqr in np.linspace(0,0.8,20):
	ax.plot(E, Pmine(E,const.nue_to_nue,uesqr,umusqr), lw=1.0, color='black', dashes=(6,0), label=r'$\left\langle P_{\hat{\nu}_{s}\to \nu_e}\right\rangle_N$')
	ax.plot(E, Pmine(E,const.nue_to_nue,uesqr,umusqr, i1=1.0/(np.cos(const.theta13)**2*np.cos(const.theta12)**2),i2=0,i3=0), lw=1.5, color='violet', dashes=(2,1), label=r'$\left\langle |U_{s1}^m|^2\right\rangle_N$')
	ax.plot(E, Pmine(E,const.nue_to_nue,uesqr,umusqr, i1=0,i2=1.0/(np.cos(const.theta13)**2*np.sin(const.theta12)**2),i3=0), lw=1.5, color='dodgerblue', dashes=(4,1), label=r'$\left\langle |U_{s2}^m|^2\right\rangle_N$')
	ax.plot(E, Pmine(E,const.nue_to_nue,uesqr,umusqr, i1=0,i2=0,i3=1.0/(np.sin(const.theta13)**2)), lw=1.5, color='darkorange', dashes=(6,3), label=r'$\left\langle|U_{s3}^m|^2\right\rangle_N$')
	
	ax2.plot(E, Pmine(E,-const.nue_to_nue,uesqr,umusqr), lw=1.0, color='black', dashes=(6,0), label=r'$\left\langle P_{\overline{\hat{\nu}_{s}}\to \overline{\nu_e}}\right\rangle_N$')
	ax2.plot(E, Pmine(E,-const.nue_to_nue,uesqr,umusqr,i1=1.0/(np.cos(const.theta13)**2*np.cos(const.theta12)**2),i2=0,i3=0), lw=1.5, color='violet', dashes=(2,1), label=r'$\left\langle |U_{s1}^m|^2\right\rangle_N$')
	ax2.plot(E, Pmine(E,-const.nue_to_nue,uesqr,umusqr,i1=0,i2=1.0/(np.cos(const.theta13)**2*np.sin(const.theta12)**2),i3=0), lw=1.5, color='dodgerblue', dashes=(4,1), label=r'$\left\langle |U_{s2}^m|^2\right\rangle_N$')
	ax2.plot(E, Pmine(E,-const.nue_to_nue,uesqr,umusqr,i1=0,i2=0,i3=1.0/(np.sin(const.theta13)**2)), lw=1.5, color='darkorange', dashes=(6,3), label=r'$\left\langle |U_{s3}^m|^2\right\rangle_N$')
	

	# ax.plot(E, (uesqr*flavour_transitions.Psolar(E,-const.nue_to_nue)+umusqr*flavour_transitions.Psolar(E,-const.numu_to_nue))/(uesqr+umusqr), lw=1.0, color='blue', dashes=(1,1),label=r'$\overline{\nu_e} \to \overline{\nu_e}$')

	ax.text(30,1.05,r'$\nu$',fontsize=15)
	ax2.text(30,1.05,r'$\overline{\nu}$',fontsize=15)

	# ax.set_title(r'$|U_{e4}|^2=10^{-2}, |U_{\mu4}|=10^{-2}$',fontsize=8)
	ax.set_title(r'$|U_{e4}|^2=0.01, |U_{\mu4}|^2=%.g$'%umusqr,fontsize=14)

	# ax.plot(E, P_Parke(E)/Padiabatic(E), lw=1.5, ls='--', label=r'Parke')



	##############
	# STYLE
	# ax.set_yscale('log')
	ax.legend(loc='upper left',frameon=False,ncol=2,fontsize=12,handletextpad=0.3)
	ax2.legend(loc='upper left',frameon=False,ncol=2,fontsize=12,handletextpad=0.3)
	
	ax.set_ylim(0,1.2)
	ax2.set_ylim(0,1.2)

	ax.set_xlim(0.1,50)
	ax2.set_xlim(0.1,50)

	ax.set_xscale('log')
	ax2.set_xscale('log')

	ax.set_ylabel(r'Probability',fontsize=15)
	ax2.set_ylabel(r'Probability',fontsize=15)

	ax.set_xlabel(r'$E_\nu/$MeV',fontsize=15)
	ax2.set_xlabel(r'$E_\nu/$MeV',fontsize=15)



	fig.savefig('plots/Psolar_%.i.pdf'%(umusqr*1000))

