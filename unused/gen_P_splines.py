import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

import numpy as np
from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

from source import *

data = np.genfromtxt('AGS09.dat')
r = data[:,0]
d = data[:,2]
f = data[:,8]
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

	P = np.sum(fraction(r)*density(r)*Pmine_r(Enu, channel,Ue4SQR,Umu4SQR,r,i1=i1,i2=i2,i3=i3) * dr)/I
	return P

NPOINTS=40
P=np.zeros((NPOINTS,NPOINTS,NPOINTS))
xue = np.logspace(-3.1,-0.9,NPOINTS)
xumu = np.linspace(0,0.1,NPOINTS)
en = np.logspace(-2,np.log10(20),NPOINTS)
for i in range(NPOINTS):
	for j in range(NPOINTS):
		for k in range(NPOINTS):
			P[i,j,k]=Pmine(en[k], -const.nue_to_nue, xue[i], xumu[j])

from scipy.interpolate import RegularGridInterpolator

func = RegularGridInterpolator((xue, xumu, en), P, method='linear',bounds_error=False,fill_value=0.0)

np.save('probs/interp.npy',{"interp": func})
load= np.load('probs/interp.npy',allow_pickle=True)
print(load)
func = load.item()['interp']
Enu =  np.logspace(-2,np.log10(20),100)
plt.plot(Enu, func(np.array([0.01*Enu/Enu,0.001*Enu/Enu,Enu]).T))
plt.ylim(0,1.2)
plt.xscale('log')
plt.savefig('plots/test_prob.pdf')
