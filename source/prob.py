import numpy as np

import pdg
import const
import decay_rates
import model


#####################################3
# Probabilities of flavour transition for nu_H -> nu Z'
def dPdEnu1(params,kin,Enu,E1,h):
	
	Umu4 = params.Umu4
	mh = params.m4
		# FIX MEEEE
	lproper_decay_N = decay_rates.L_GeV_to_cm(decay_rates.GammaTOT_nuh_nualpha_Zprime(params))/100.0 # meters 
	
	ans = Umu4*Umu4
	ans*= R1(params,kin,E1,h)
	# ans*= (1.0 - np.exp(-const.MB_baseline/lproper_decay_N/kin.gamma/(-kin.beta) ) )
	ans*= const.Heaviside(-E1 + kin.E1L_MAX())*const.Heaviside(E1 - kin.E1L_MIN())#
	return ans


#####################################3
# Probabilities of flavour transition nu_H -> nu Z' -> nu nu nubar
def dPdEnu2dEnu1(params,kin,Enu,E1,E2,h):

	Umu4 = params.Umu4
	mh = params.m4

	Ez = Enu-E1

	if params.model == const.VECTOR:
		lproper_decay_N = decay_rates.L_GeV_to_cm(decay_rates.GammaTOT_nuh_nualpha_Zprime(params))/100.0 # meters 
		lproper_decay_Zprime = decay_rates.L_GeV_to_cm(decay_rates.GammaTOT_Zprime_nu_nu(params))/100.0 # meters 
	elif params.model == const.SCALAR:
		lproper_decay_N = decay_rates.L_GeV_to_cm(decay_rates.GammaTOT_nuh_nualpha_Phi(params))/100.0 # meters 
		lproper_decay_Zprime = decay_rates.L_GeV_to_cm(decay_rates.GammaTOT_Phi_nu_nu(params))/100.0 # meters 
	else:
		print 'ERROR! Could not specify what model we have.'
		return None

	ans = Umu4*Umu4
	ans *= R1(params,kin,E1,h)*R2(params,kin,E1,E2,h)
	
	# ans*= (1.0 - np.exp(-params.BASELINE/lproper_decay_N/kin.gamma/(-kin.beta) ) )
	ans*= const.Heaviside(-E1 + kin.E1L_MAX())*const.Heaviside(E1 - kin.E1L_MIN())#
	
	# ans*= (1.0 - np.exp(-params.BASELINE/lproper_decay_Zprime/kin.gammaz/(-kin.betaz) ) )
	ans*= const.Heaviside(-E2 + kin.E2L_MAX())*const.Heaviside(E2 - kin.E2L_MIN())
	
	return ans

######
def R1(params,kin,E1,h):
	if params.model == const.VECTOR:
		tot=decay_rates.GammaTOT_nuh_nualpha_Zprime(params)
		dif=decay_rates.dGamma_nuh_nualpha_Zprime_dCostheta(params,kin.CosTheta(E1), h)*decay_rates.dCostheta_dE1(kin)
	elif params.model == const.SCALAR:
		tot=decay_rates.GammaTOT_nuh_nualpha_Phi(params)
		dif=decay_rates.dGamma_nuh_nualpha_Phi_dCostheta(params,kin.CosTheta(E1), h)*decay_rates.dCostheta_dE1(kin)
	else:
		print 'ERROR! Could not specify what model we have.'
		return None
	return dif/tot

def R2(params,kin,E1,E2,h):
	if params.model == const.VECTOR:
		tot=decay_rates.GammaTOT_Zprime_nu_nu(params)
		dif=decay_rates.dGamma_Zprime_nu_nu_dCostheta(params,kin.CosThetaZ(E2))*decay_rates.dCosthetaZ_dE2(kin)
	elif params.model == const.SCALAR:
		tot=decay_rates.GammaTOT_Phi_nu_nu(params)
		dif=decay_rates.dGamma_Phi_nu_nu_dCostheta(params,kin.CosThetaZ(E2))*decay_rates.dCosthetaZ_dE2(kin)
	else:
		print 'ERROR! Could not specify what model we have.'
		return None
	
	return dif/tot

##############################
# Deprecated -- test for oscillations at SBL
# SBL OSCILLATION
def dPdE1_OSCILLATION(params,Enu,L):
	Umu4 = params.Umu4
	Ue4 = params.Ue4
	dm4SQR = params.dm4SQR
	return 4.0*Umu4*Umu4*Ue4*Ue4*(np.sin(1.27*(dm4SQR)*L/Enu))**2
