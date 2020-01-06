import numpy as np

from source import *

class decay_model_params():

	def __init__(self, MODEL):

		self.model = MODEL
		self.gx		= 0.0
		self.Ue4		= 0.0
		self.Umu4		= 0.0
		self.Utau4		= 0.0
		self.Ue5		= 0.0
		self.Umu5		= 0.0
		self.Utau5		= 0.0
		self.UD4		= 1.0
		self.UD5		= 0.0
		self.m4		= 1.0
		self.m5		= 1e10

		self.mBOSON= 1.0

	# def set_high_level_variables(self):


class kinematics:

	def __init__(self,params,Enu):

		# INPUT Parameters
		self.Enu = Enu
		self.mh = params.m4
		self.mBOSON=params.mBOSON
		
	##########################################
	# HNL DECAYS
	
		# HNL CM frame kinematics
		self.gamma=self.Enu/self.mh
		self.PnuH=const.momentum(self.Enu,self.mh)
		self.beta=-self.PnuH/self.Enu

		self.E1CM=(self.mh*self.mh-self.mBOSON*self.mBOSON)/2.0/self.mh
		self.P1CM=self.E1CM
		
		self.EBOSONCM=(self.mh*self.mh+self.mBOSON*self.mBOSON)/2.0/self.mh
		self.PBOSONCM=const.momentum(self.EBOSONCM,self.mBOSON)

	# def EzprimeL(self,CosTheta):
	# 	return self.gamma*(self.EzprimeCM - self.beta*self.PzprimeCM*CosTheta) 

	# def EzprimeL_MAX(self):
	# 	return self.EzprimeL(1.0)
	
	# def EzprimeL_MIN(self):
	# 	return self.EzprimeL(-1.0)

	######
	# WATCH IT! BETA IS NEGATIVE!
	def E1L(self,CosTheta):
		return self.gamma*(self.E1CM - self.beta*self.P1CM*CosTheta) 

	def E1L_MAX(self):
		return self.E1L(1.0)
	
	def E1L_MIN(self):
		return self.E1L(-1.0)

	def CosTheta(self,E1):
		return (self.gamma*self.E1CM - E1)/self.gamma/self.beta/self.P1CM


	##########################################
	# ZPRIME DECAYS 
	def set_BOSON_decay_variables(self,Ez):
		self.Ez=Ez
		# BOSON CM frame kinematics
		self.gammaz=self.Ez/self.mBOSON
		self.PBOSON=const.momentum(self.Ez,self.mBOSON)
		self.betaz=-self.PBOSON/self.Ez
		
		self.E2CM=self.mBOSON/2.0
		self.P2CM=self.E2CM

	def E2L(self,CosThetaZ):
		return self.gammaz*self.E2CM*(1.0 - self.betaz*CosThetaZ) 
	
	def E2L_MAX(self):
		return self.E2L(1.0)
	
	def E2L_MIN(self):
		return self.E2L(-1.0)

	def CosThetaZ(self,E2):
		return (self.gammaz*self.E2CM - E2)/self.gammaz/self.betaz/self.E2CM


#######################################
# OSCILLATION MODEL FOR TEST
class osc_model_params():
	def __init__(self):
		self.Ue4		= 0.0
		self.Umu4		= 0.0
		self.Utau4		= 0.0
		self.Ue5		= 0.0
		self.Umu5		= 0.0
		self.Utau5		= 0.0
		self.UD4		= 1.0
		self.UD5		= 0.0
		self.dm4SQR		= 1.0
		self.dm5SQR		= 1e10