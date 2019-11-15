import numpy as np
from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

import vegas
import gvar as gv

import const
import pdg
import model
import decay_rates as dr
import prob
import fluxes
import xsecs
import exps
import standard_oscillations as std_osc
import stats
################################################################
# Compute the chi2 using the model independent limits provided by collaborations
def fill_bins(exp,params,fluxfile,endpoint=1e100):
	###########
	# SET BINS TO BE THE EXPERIMENTAL BINS
	bins = exp.bin_e # bin edges
	bin_c = exp.bin_c # bin edges

	###########
	# NUMU FLUX
	flux = fluxes.get_exp_flux(fluxfile)

	############
	# NUE/BAR XS
	xsec = lambda x : np.zeros(np.size(x))
	xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
	xsecbar = xsecs.get_IBD(xsfile)

	############
	# efficiencies
	enu_eff= bins
	eff= np.ones((np.size(bins)-1))

	############
	# number of events
	NCASCADE, dNCASCADE = RATES_dN_HNL_CASCADE_NU_NUBAR(\
												flux=flux,\
												xsec=xsec,\
												xsecbar=xsecbar,\
												dim=3,\
												enumin=0,\
												enumax=16.8,\
												params=params,\
												bins=bins,\
												PRINT=False,\
												enu_eff=enu_eff,\
												eff=eff,
												smearing_function=exp.smearing_function)

	NP_MC = dNCASCADE[bin_c<endpoint]*exp.norm
	bin_c_end = bin_c[bin_c<endpoint]

	if exp.exp_name==const.KAMLAND:
		back_MC = exp.MCall_binned[bin_c<endpoint]
		D = exp.data[bin_c<endpoint]
	elif exp.exp_name==const.BOREXINO:
		back_MC = exp.MCreactor[bin_c<endpoint]
		D = exp.data[bin_c<endpoint]
	
	return bin_c_end, NP_MC, back_MC, D

def RATES_dN_HNL_CASCADE_NU_NUBAR(flux,xsec,xsecbar,dim=3,enumin=0,enumax=2.0,params=None,bins=None,PRINT=False,eff=None,enu_eff=None,smearing_function=None):
	f = HNL_CASCADE_NU_NUBAR(flux,xsec,xsecbar,dim=dim,enumin=enumin,enumax=enumax,params=params,bins=bins,eff=eff,enu_eff=enu_eff,smearing_function=smearing_function)
	integ = vegas.Integrator(f.dim * [[0, 1]])
	# adapt grid
	training = integ(f, nitn=20, neval=1e4)
	# compute integral
	result = integ(f, nitn=20, neval=1e4)
	if PRINT:	
		print result.summary()
		print '   I =', result['I']
		print 'dI/I =', result['dI'] / result['I']
		print 'check:', sum(result['dI'])

	dNdEf = gv.mean(result['dI'])
	N = gv.mean(result['I'])

	return N, dNdEf

def RATES_dN_HNL_TO_ZPRIME(flux,xsec,dim=2,enumin=0,enumax=2.0,params=None,bins=None,PRINT=False,eff=None,enu_eff=None,smearing_function=None):
	f = HNL_TO_NU_ZPRIME(flux,xsec,dim=dim,enumin=enumin,enumax=enumax,params=params,bins=bins,eff=eff,enu_eff=enu_eff,smearing_function=smearing_function)
	integ = vegas.Integrator(f.dim * [[0, 1]])
	# adapt grid
	training = integ(f, nitn=20, neval=1000)
	# compute integral
	result = integ(f, nitn=20, neval=1e5)
	if PRINT:	
		print result.summary()
		print '   I =', result['I']
		print 'dI/I =', result['dI'] / result['I']
		print 'check:', sum(result['dI'])

	dNdEf = gv.mean(result['dI'])
	N = gv.mean(result['I'])

	return N, dNdEf


def RATES_SBL_OSCILLATION(flux,xsec,dim=1,enumin=0,enumax=2.0,params=None,bins=None,PRINT=False,L=0.541,eff=None,enu_eff=None,smearing_function=None):
	f = SBL_OSCILLATION(flux,xsec,dim=dim,enumin=enumin,enumax=enumax,params=params,bins=bins,L=L,eff=eff,enu_eff=enu_eff,smearing_function=smearing_function)
	integ = vegas.Integrator(f.dim * [[0, 1]])
	# adapt grid
	training = integ(f, nitn=20, neval=1000)
	# compute integral
	result = integ(f, nitn=20, neval=1e4)
	if PRINT:	
		print result.summary()
		print '   I =', result['I']
		print 'dI/I =', result['dI'] / result['I']
		print 'check:', sum(result['dI'])

	dNdEf = gv.mean(result['dI'])
	N = gv.mean(result['I'])

	return N, dNdEf

def dN(kin,flux,xsec,params,Enu,E1):
	# SPECIAL CASE 
	h=1
	N = flux(Enu)*xsec(E1)*prob.dPdEnu1(params,kin,Enu,E1,h)
	h=-1
	N += flux(Enu)*xsec(E1)*prob.dPdEnu1(params,kin,Enu,E1,h)
	return N

############
# Full Cascade -- only take nuebar
def dN2(kin,flux,xsec,xsecbar,params,Enu,E1,E2):
	# fraction of final states with nuebar
	fe = params.Ue4**2/(params.Ue4**2 + params.Umu4**2 +params.Utau4**2)
	# fraction of final states with numubar and nutaubar
	fmutau = (params.Umu4**2+params.Utau4**2)/(params.Ue4**2 + params.Umu4**2 +params.Utau4**2)

	h=-1
	# neutrinos from Boson decay
	# N = flux(Enu)*(xsec(E1)*prob.dPdEnu2dEnu1(params,kin,Enu,E1,E2,h)*std_osc.P_Parke(E2, const.nue_to_nue))
	
	# antineutrinos from Boson decay
	N= flux(Enu)*(fe*std_osc.Padiabatic(E2, -const.nue_to_nue) + fmutau*std_osc.Padiabatic(E2, -const.numu_to_nue))*xsecbar(E2)*prob.dPdEnu2dEnu1(params,kin,Enu,E1,E2,h)
	
	h=1
	# neutrinos from Boson decay
	# N = flux(Enu)*(xsec(E1)*prob.dPdEnu2dEnu1(params,kin,Enu,E1,E2,h)*std_osc.P_Parke(E2, const.nue_to_nue))
	
	# antineutrinos from Boson decay
	N+= flux(Enu)*(fe*std_osc.Padiabatic(E2, -const.nue_to_nue) + fmutau*std_osc.Padiabatic(E2, -const.numu_to_nue))*xsecbar(E2)*prob.dPdEnu2dEnu1(params,kin,Enu,E1,E2,h)
	
	return N


def dN_OSCILLATION(flux,xsec,params,Enu,L):
	N = flux(Enu)*xsec(Enu)*prob.dPdE1_OSCILLATION(params,Enu,L)
	return N


class SBL_OSCILLATION(vegas.BatchIntegrand):
    def __init__(self, flux,xsec, dim, enumin,enumax,params,bins,enu_eff,eff,L,exp):
		self.dim = dim
		self.enumin = enumin
		self.enumax = enumax
		self.params = params
		self.bins = bins
		self.L = L
		self.flux=flux
		self.xsec=xsec
		self.enu_eff=enu_eff
		self.eff=eff
    def __call__(self, x):	
		
		# Return final answer as a dict with multiple quantities
		ans = {}

		# Physical limits of integration
		enu = (self.enumax-self.enumin)*x[:,0] + x[:,0]*self.enumin

		# integral
		I = dN_OSCILLATION(self.flux,self.xsec,self.params,enu,self.L)*(self.enumax-self.enumin)

		# distribution
		dI = np.zeros((np.size(x[:,0]),np.size(self.bins[:-1])), dtype=float)

		# fill distribution
		for i in range(np.size(x[:,0])):
			j = np.where( (self.bins[:-1] < enu[i]) & (self.bins[1:] > enu[i] ))[0]
			dI[i,j] = I[i]

		################################
		## EFFICIENCIES -- IMPROVE ME
		for i in range(np.size(x[:,0])):
			j = np.where( (self.enu_eff[:-1] < enu[i]) & (self.enu_eff[1:] > enu[i] ))[0]
			dI[i,:] *= self.eff[j]
			I[i] *= self.eff[j]
		################################


		ans['I'] = I
		ans['dI'] = dI
		return ans


class HNL_TO_NU_ZPRIME(vegas.BatchIntegrand):
    def __init__(self,flux, xsec, dim, enumin,enumax, params,bins,enu_eff,eff,smearing_function):
		self.dim = dim
		self.enumin = enumin
		self.enumax = enumax
		self.params = params
		self.bins = bins
		self.flux=flux
		self.xsec=xsec
		self.enu_eff=enu_eff
		self.eff=eff
		self.smearing_function=smearing_function
    def __call__(self, x):	
		
		# Return final answer as a dict with multiple quantities
		ans = {}

		# Physical limits of integration
		enu = (self.enumax-self.enumin)*x[:,0] + x[:,0]*self.enumin
		kin = model.kinematics(self.params,enu)
		e1min = kin.E1L_MIN()
		e1max = kin.E1L_MAX()
		e1 = (e1max-e1min)*x[:,1] + x[:,1]*e1min
		
		# integral
		I = dN(kin,self.flux,self.xsec,self.params,enu,e1)*(self.enumax-self.enumin)*(e1max-e1min)

		# distribution
		dI = np.zeros((np.size(x[:,0]),np.size(self.bins[:-1])), dtype=float)

		# Smearing on the detected neutrino energy
		e2 = self.smearing_function(e2)

		# fill distribution
		for i in range(np.size(x[:,0])):
			j = np.where( (self.bins[:-1] < e1[i]) & (self.bins[1:] > e1[i] ))[0]
			dI[i,j] = I[i]
		# print np.shape(dI)
		# print np.shape(self.eff)
		################################
		## EFFICIENCIES -- IMPROVE ME
		# for i in range(np.size(x[:,0])):
		# 	j = np.where( (self.enu_eff[:-1] < e1[i]) & (self.enu_eff[1:] > e1[i] ))[0]
		# 	dI[i,:] *= self.eff[j]
		# 	I[i] *= self.eff[j]
		################################

		ans['I'] = I
		ans['dI'] = dI
		return ans


class HNL_CASCADE_NU_NUBAR(vegas.BatchIntegrand):
    def __init__(self,flux,xsec,xsecbar,dim,enumin,enumax,params,bins,enu_eff,eff,smearing_function):
		self.dim = dim
		self.enumin = enumin
		self.enumax = enumax
		self.params = params
		self.bins = bins
		self.flux=flux
		self.xsec=xsec
		self.xsecbar=xsecbar
		self.enu_eff=enu_eff
		self.eff=eff
		self.smearing_function=smearing_function

    def __call__(self, x):	
		
		# Return final answer as a dict with multiple quantities
		ans = {}

		# Physical limits of integration
		enu = (self.enumax-self.enumin)*x[:,0] + x[:,0]*self.enumin
		
		kin = model.kinematics(self.params,enu)
		e1min = kin.E1L_MIN()
		e1max = kin.E1L_MAX()
		e1 = (e1max-e1min)*x[:,1] + x[:,1]*e1min
		
		# Zprime decay
		ez = enu - e1
		kin.set_BOSON_decay_variables(ez)
		e2min = kin.E2L_MIN()
		e2max = kin.E2L_MAX()
		e2 = (e2max-e2min)*x[:,2] + x[:,2]*e2min

		# integral
		I = dN2(kin,self.flux,self.xsec,self.xsecbar,self.params,enu,e1,e2)*(self.enumax-self.enumin)*(e1max-e1min)*(e2max-e2min)

		# distribution
		dI = np.zeros((np.size(x[:,0]),np.size(self.bins[:-1])), dtype=float)

		##########
		# Smearing on the detected neutrino energy
		e2 = np.array([self.smearing_function(EE)  if EE>const.IBD_THRESHOLD else EE for EE in e2])

		# fill distribution
		for i in range(np.size(x[:,0])):
			j = np.where( (self.bins[:-1] < e2[i]) & (self.bins[1:] > e2[i] ))[0]
			dI[i,j] = I[i]
		
		################################
		## EFFICIENCIES -- FIX ME
		# for i in range(np.size(x[:,0])):
		# 	j = np.where( (self.enu_eff[:-1] < e2[i]) & (self.enu_eff[1:] > e2[i] ))[0]
		# 	dI[i,:] *= self.eff[j]
		# 	I[i] *= self.eff[j]
		################################

		ans['I'] = I
		ans['dI'] = dI
		return ans