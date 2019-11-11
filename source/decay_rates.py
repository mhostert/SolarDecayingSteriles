import numpy as np
from scipy import interpolate
import scipy.stats
import matplotlib.pyplot as plt
from scipy.integrate import quad

import const
import pdg
	
def tau_GeV_to_s(decay_rate):
	return 1./decay_rate/1.52/1e24

def L_GeV_to_cm(decay_rate):
	return 1./decay_rate/1.52/1e24*2.998e10

def lam(a,b,c):
	return a**2 + b**2 + c**2 -2*a*b - 2*b*c - 2*a*c

def I1_2body(x,y):
	return ((1+x-y)*(1+x) - 4*x)*np.sqrt(lam(1.0,x,y))

####
def dCostheta_dE1(kin):
	return (kin.mh/kin.PnuH/kin.P1CM)

####
def dCostheta_dEZprime(kin):
	return (kin.mh/kin.PnuH/kin.PzprimeCM)

####
def dCosthetaZ_dE2(kin):
	return (kin.mzprime/kin.Pzprime/kin.P2CM)


##########################################
# nu_h DECAYS 
def dGamma_nuh_nualpha_Zprime_dCostheta(params,CosTheta):
	mh = params.m4
	mzprime = params.mzprime
	gx = params.gx
	Ue4 = params.Ue4
	
	couplings = gx*gx*Ue4*Ue4

	amp2 = (mh*mh*mh*mh + mh*mh*(mzprime*mzprime) - 2*(mzprime*mzprime*mzprime*mzprime) + \
		     CosTheta*(mh*mh*mh)*np.sqrt(((-(mh*mh) + mzprime*mzprime)*(-(mh*mh) + mzprime*mzprime))/(mh*mh)) - \
		     2*CosTheta*mh*(mzprime*mzprime)*np.sqrt(((-(mh*mh) + mzprime*mzprime)*(-(mh*mh) + mzprime*mzprime))/(mh*mh)))/(2.*(mzprime*mzprime))

	# amp2 = (mh*mh + h*h*(mh*mh) - mzprime*mzprime - h*h*(mzprime*mzprime) + 
	# 	      2*CosTheta*h*mh*Sqrt(((mh*mh - mzprime*mzprime)*(mh*mh - mzprime*mzprime))/(mh*mh)))/2. - 
	# 	   (mh*mh*(-(mh*mh) - h*h*(mh*mh) + mzprime*mzprime + h*h*(mzprime*mzprime) + 
	# 	        2*CosTheta*h*mh*Sqrt(((mh*mh - mzprime*mzprime)*(mh*mh - mzprime*mzprime))/(mh*mh))))/
	# 	    (4.*(mzprime*mzprime))

	dPS2 = (1.0-mzprime*mzprime/mh/mh)/32.0/np.pi/np.pi
	flux_factor = 1.0/2.0/mh
	integral_dPhi = 2*np.pi

	return amp2*dPS2*flux_factor*couplings*integral_dPhi

def GammaTOT_nuh_nualpha_Zprime(params):
	# mh = params.m4
	# mzprime = params.mzprime
	# gx = params.gx
	# Ue4 = params.Ue4
	# Pi = np.pi
	# couplings = gx*gx*Ue4*Ue4
	# return couplings*((mh - mzprime)*(mh - mzprime)*((mh + mzprime)*(mh + mzprime))*(mh*mh + 2*(mzprime*mzprime)))/(32.*(mh*mh*mh)*(mzprime*mzprime)*Pi)
	return dGamma_nuh_nualpha_Zprime_dCostheta(params,0)*2

##########################################
# Z PRIME DECAYS 
def dGamma_Zprime_nu_nu_dCostheta(params,CosThetaZ):
	mzprime = params.mzprime
	gx = params.gx
	Ue4 = params.Ue4

	couplings = gx*gx*Ue4*Ue4*Ue4*Ue4
	amp2 = 4.0*mzprime/3.0
	dPS2 = 1.0/32.0/np.pi/np.pi
	flux_factor = 1.0/2.0/mzprime
	integral_dphi = 2.0*np.pi
	return amp2*dPS2*flux_factor*couplings*integral_dphi

def GammaTOT_Zprime_nu_nu(params):
	integral_dcos = 2
	return dGamma_Zprime_nu_nu_dCostheta(params,None)*integral_dcos

















################# OLD CRAP



def nu4_to_nualpha_l_l(params, final_lepton):
	if (final_lepton==pdg.PDG_tau):
		m_ell = const.Mtau
	elif(final_lepton==pdg.PDG_muon):
		m_ell = const.Mmu
	elif(final_lepton==pdg.PDG_electron):
		m_ell = const.Me
	else:
		print "WARNING! Unable to set charged lepton mass. Assuming massless."
		m_ell = 0

	if (final_lepton==pdg.PDG_tau):
		CC_mixing = params.Utau4
	elif(final_lepton==pdg.PDG_muon):
		CC_mixing = params.Umu4
	elif(final_lepton==pdg.PDG_electron):
		CC_mixing = params.Ue4
	else:
		print "WARNING! Unable to set CC mixing parameter for decay. Assuming 0."
		CC_mixing = 0

	mi = params.m4
	m0 = 0.0
	def func(u,t):
		gv = (const.g/const.cw)**2/2.0 *( params.cmu4*params.ceV/const.Mz**2 - params.dmu4*params.deV/(t-params.Mzprime**2) ) \
						- const.g**2/4.0*CC_mixing/const.Mw**2
		ga = (const.g/const.cw)**2/2.0 *(-params.cmu4*params.ceA/const.Mz**2 + params.dmu4*params.deA/(t-params.Mzprime**2) ) \
						+ const.g**2/4.0*CC_mixing/const.Mw**2
		# print "gv, ga: ", gv, ga
		return 4.0*((gv + ga)**2 *(mi**2 + m_ell**2 - u)*(u - m0**2 -m_ell**2)
						+ (gv - ga)**2*(mi**2 - m0**2 - m_ell**2)*(mi**2 + m_ell**2 - mi**2)
							+ (gv**2 - ga**2)*m_ell**2/2.0*(mi**2 + m0**2 - t))
	
	uminus = lambda t: (mi**2 - m0**2)**2/4.0/t - t/4.0*(np.sqrt(lam(1, mi**2/t, m0**2/t)) + np.sqrt(lam(1,m_ell**2/t, m_ell**2/t)))**2
	uplus = lambda t: (mi**2 - m0**2)**2/4.0/t - t/4.0*(np.sqrt(lam(1, mi**2/t, m0**2/t)) - np.sqrt(lam(1,m_ell**2/t, m_ell**2/t)))**2

	integral, error = scipy.integrate.dblquad(	func,
												(mi-m0)**2, 
												4*m_ell**2,
												uplus,
												uminus,
												args=(), epsabs=1.49e-08, epsrel=1.49e-08)

	return integral*1.0/(2.0*np.pi)**3 / 32.0 / mi**3


############### HEAVY NEUTRINO ############################
def N_to_Z_nu(params):
	Mn = params.m4
	return params.alphaD/2.0 * params.UD4**2 * (params.Ue4**2 + params.Umu4**2 + params.Utau4**2) * Mn**3/params.Mzprime**2 *(1.0 - params.Mzprime**2/Mn**2)*(1 + params.Mzprime**2/Mn**2 - 2.0 * params.Mzprime**4/Mn**4) * ( 1 + params.Dirac*(-1/2.0) )

def N_total(params):
	return N_to_Z_nu(params)


############### Z PRIME ############################
def Z_to_ll(params, ml):
	if 2*ml < params.Mzprime:
		### THIS EQUATION NEEDS TO BE FIXED TO INCLUDE lepton mass effects
		# return const.alphaQED*epsilon**2/3.0 * (Mzprime) 
		
		### includes lepton mass effects
		return (const.alphaQED*(params.epsilon*params.epsilon)*np.sqrt(-4.0*(ml*ml) + params.Mzprime*params.Mzprime)*(5*(ml*ml) + 2*(params.Mzprime*params.Mzprime)))/(6.*(params.Mzprime*params.Mzprime))
	elif 2*ml >= params.Mzprime:
		return 0.0

def Z_to_nunu(params):
	return params.alphaD/3.0 * (params.Ue4**2 + params.Umu4**2 + params.Utau4**2)**2 * params.Mzprime

def Z_total(params):
	return Z_to_nunu(params) + Z_to_ll(params, const.Me) + Z_to_ll(params, const.Mmu)