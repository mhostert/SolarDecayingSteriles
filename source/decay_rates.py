import numpy as np

from numba import jit

from source import *

def tau_GeV_to_s(decay_rate):
	return 1./decay_rate/1.52/1e24

def L_GeV_to_cm(decay_rate):
	return 1./decay_rate/1.52/1e24*2.998e10

def lam(a,b,c):
	return a**2 + b**2 + c**2 -2*a*b - 2*b*c - 2*a*c

def I1_2body(x,y):
	return ((1+x-y)*(1+x) - 4*x)*np.sqrt(lam(1.0,x,y))




######################################################################
# GENERAL KINEMAICS FUNCTIONS
######################################################################

####
@jit
def dCostheta_dE1(kin):
	return (kin.mh/kin.PnuH/kin.P1CM)

####
@jit
def dCostheta_dEZprime(kin):
	return (kin.mh/kin.PnuH/kin.PBOSONCM)

####
@jit
def dCosthetaZ_dE2(kin):
	return (kin.mBOSON/kin.PBOSON/kin.P2CM)



######################################################################
# VECTOR CASE
######################################################################

##########################################
# nu_h DECAYS to nu_alpha Zprime
@jit
def dGamma_nuh_nualpha_Zprime_dCostheta(params,CosTheta, h):
	mh = params.m4
	mzprime = params.mBOSON
	gx = params.gx
	Ue4 = params.Ue4
	couplings = gx*gx*Ue4*Ue4

	amp2 = -((-1 + 2*CosTheta*h - h*h)*(mh - mzprime)*(mh + mzprime))/2. - ((1 + 2*CosTheta*h + h*h)*(mh*mh)*(-mh + mzprime)*(mh + mzprime))/(4.*(mzprime*mzprime))

	dPS2 = (1.0-mzprime*mzprime/mh/mh)/32.0/np.pi/np.pi
	flux_factor = 1.0/2.0/mh
	integral_dPhi = 2*np.pi

	return amp2*dPS2*flux_factor*couplings*integral_dPhi

def GammaTOT_nuh_nualpha_Zprime(params):
	return dGamma_nuh_nualpha_Zprime_dCostheta(params,0,0)*2

##########################################
# Z PRIME DECAYS 
@jit
def dGamma_Zprime_nu_nu_dCostheta(params,CosThetaZ):
	mzprime = params.mBOSON
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



######################################################################
# SCALAR CASE
######################################################################

##########################################
# nu_h DECAYS to nu_alpha PHI
def dGamma_nuh_nualpha_Phi_dCostheta(params,CosTheta,h):
	mh = params.m4
	mphi = params.mBOSON
	gx = params.gx
	Ue4 = params.Ue4
	couplings = gx*gx*Ue4*Ue4

	amp2 = -((-1 + 2*CosTheta*h - h*h)*(mh - mphi)*(mh + mphi))/4.

	dPS2 = (1.0-mphi*mphi/mh/mh)/32.0/np.pi/np.pi
	flux_factor = 1.0/2.0/mh
	integral_dPhi = 2*np.pi

	return amp2*dPS2*flux_factor*couplings*integral_dPhi

def GammaTOT_nuh_nualpha_Phi(params):
	integral_dcos = 2.0
	return dGamma_nuh_nualpha_Phi_dCostheta(params,0,0)*integral_dcos

##########################################
# PHI DECAYS 
def dGamma_Phi_nu_nu_dCostheta(params,CosThetaZ):
	mphi = params.mBOSON
	gx = params.gx
	Ue4 = params.Ue4

	couplings = gx*gx*Ue4*Ue4*Ue4*Ue4
	amp2 = 4.0*mphi/3.0
	dPS2 = 1.0/32.0/np.pi/np.pi
	flux_factor = 1.0/2.0/mphi
	integral_dphi = 2.0*np.pi
	return amp2*dPS2*flux_factor*couplings*integral_dphi

def GammaTOT_Phi_nu_nu(params):
	integral_dcos = 2.0
	return dGamma_Zprime_nu_nu_dCostheta(params,None)*integral_dcos
