import numpy as np
import scipy

from numba import jit

from source import *

a = np.genfromtxt('source/Pab_noCP.dat', unpack=True)
aCP = np.genfromtxt('source/Pab_CP.dat', unpack=True)
b = np.genfromtxt('source/Pab_bar_noCP.dat', unpack=True)
bCP = np.genfromtxt('source/Pab_bar_CP.dat', unpack=True)

Eint = b[0,:]

@jit
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
