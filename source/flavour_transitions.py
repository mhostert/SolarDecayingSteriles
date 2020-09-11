import numpy as np
import scipy

# from numba import jit

from source import *

a = np.genfromtxt('source/Pab_noCP.dat', unpack=True)
aCP = np.genfromtxt('source/Pab_CP.dat', unpack=True)
b = np.genfromtxt('source/Pab_bar_noCP.dat', unpack=True)
bCP = np.genfromtxt('source/Pab_bar_CP.dat', unpack=True)

Eint = b[0,:]

# @jit
def Pab(i,CP):
	if i>0 and not CP:
		return a[i,:]
	if i>0 and CP:
		return aCP[i,:]
	if i<0 and not CP:
		return b[-i,:]
	if i<0 and CP:
		return bCP[-i,:]

def Psolar(Enu,channel):
	if np.abs(channel) == const.nue_to_nue:
		y=Pab(np.sign(channel)*1,False)
	if np.abs(channel) == const.numu_to_nue:
		y=Pab(np.sign(channel)*7,False)
	if np.abs(channel) == const.nutau_to_nue:
		y=Pab(np.sign(channel)*13,False)
	P = scipy.interpolate.interp1d(Eint,y, fill_value=(y[0],y[-1]), bounds_error=False)
	return P(Enu)




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

	r=np.linspace(0,0.5,100)
	dr = r[1]-r[0]

	P = np.array([ np.sum(fraction(r)*density(r)*Pmine_r(E, channel,Ue4SQR,Umu4SQR,r,i1=i1,i2=i2,i3=i3) * dr)/I for E in Enu])
	return P

load= np.load('probs/interp.npy',allow_pickle=True)
print(load)
func = load.item()['interp']
def Pse(Enu, Ue4SQR,Umu4SQR,i1=1,i2=1,i3=1):
	one = np.ones(np.size(Enu))
	return func(np.array([Ue4SQR*one,Umu4SQR*one,Enu]).T)
