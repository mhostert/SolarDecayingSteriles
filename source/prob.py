import numpy as np

import pdg
import const
import decay_rates
import model


def Heaviside(x):
    return 1 * (x > 0)


#####################################3
# Probabilities of flavour transition 
def dPdEnu1(params,kin,Enu,E1,h):
	
	Umu4 = params.Umu4
	mh = params.m4
		
	lproper_decay_N = decay_rates.L_GeV_to_cm(decay_rates.GammaTOT_nuh_nualpha_Zprime(params))/100.0 # meters 
	
	ans = Umu4*Umu4
	ans*= R1(params,kin,E1,h)
	# ans*= (1.0 - np.exp(-const.MB_baseline/lproper_decay_N/kin.gamma/(-kin.beta) ) )
	ans*= Heaviside(-E1 + kin.E1L_MAX())*Heaviside(E1 - kin.E1L_MIN())#
	return ans


#####################################3
# Probabilities of flavour transition
def dPdEnu2dEnu1(params,kin,Enu,E1,E2,h):

	Umu4 = params.Umu4
	mh = params.m4

	Ez = Enu-E1

	lproper_decay_N = decay_rates.L_GeV_to_cm(decay_rates.GammaTOT_nuh_nualpha_Zprime(params))/100.0 # meters 
	lproper_decay_Zprime = decay_rates.L_GeV_to_cm(decay_rates.GammaTOT_Zprime_nu_nu(params))/100.0 # meters 

	ans = Umu4*Umu4
	ans*= R1(params,kin,E1,h)*R2(params,kin,E1,E2,h)
	
	# ans*= (1.0 - np.exp(-const.MB_baseline/lproper_decay_N/kin.gamma/(-kin.beta) ) )
	ans*= Heaviside(-E1 + kin.E1L_MAX())*Heaviside(E1 - kin.E1L_MIN())#
	
	# ans*= (1.0 - np.exp(-const.MB_baseline/lproper_decay_Zprime/kin.gammaz/(-kin.betaz) ) )
	ans*= Heaviside(-E2 + kin.E2L_MAX())*Heaviside(E2 - kin.E2L_MIN())
	
	return ans


def R1(params,kin,E1,h):
	if h==1:
		tot=decay_rates.GammaTOT_nuh_nualpha_Zprime(params)
		dif=decay_rates.dGamma_nuh_nualpha_Zprime_dCostheta(params,kin.CosTheta(E1))*decay_rates.dCostheta_dE1(kin)
		# return dif/tot
		return dif/dif
	# FIX ME -- needs the helicity flipping channel -- not relevant for Z' so far
	if h==-1:
		return 0.0

def R2(params,kin,E1,E2,h):
	if h==1:
		tot=decay_rates.GammaTOT_Zprime_nu_nu(params)
		dif=decay_rates.dGamma_Zprime_nu_nu_dCostheta(params,kin.CosThetaZ(E2))*decay_rates.dCosthetaZ_dE2(kin)
		# return dif/tot
		return dif/dif
	# FIX ME -- needs the helicity flipping channel -- not relevant for Z' so far
	if h==-1:
		return 0.0




##############################
# SBL OSCILLATION
def dPdE1_OSCILLATION(params,Enu,L):
	Umu4 = params.Umu4
	Ue4 = params.Ue4
	dm4SQR = params.dm4SQR
	return 4.0*Umu4*Umu4*Ue4*Ue4*(np.sin(1.27*(dm4SQR)*L/Enu))**2
