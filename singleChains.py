import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as op
import scipy.io as io
import emcee

class fitPixel(object):
	"""
	Class to fit the Stokes Q, U anv V of a single pixel using emcee and the weak-field limit approximation
	"""
	
	def __init__(self, CQ, CU, CV, sigmaNoise):
		self.CQ = CQ
		self.CU = CU
		self.CV = CV
		self.sigmaNoise = sigmaNoise
		self.limits = np.asarray([[0.01,2500.0],[-1.0,1.0],[0.0,1.0],[-np.pi/2.0,np.pi/2.0]])
		
	def sigmoid(self, x, lower, upper):
		return lower + (upper-lower) / (1.0 + np.exp(-x))
	
	def diffSigmoid(self, x, lower, upper):
		return (upper-lower) * np.exp(-x) / (1.0 + np.exp(-x))**2
	
	def invSigmoid(self, x, lower, upper):
		return np.log( (lower-x) / (x-upper) )

	def lnlike(self, theta, includeQU):
		
		B, mu, f, phi = theta
				
		cos2Phi = np.cos(2.0*phi)
		sin2Phi = np.sin(2.0*phi)
		
		logp = 0.0
		
		p = B*mu*f		
		logp = p**2 * self.CV[1] - p * self.CV[2]
		
		if (includeQU):
			p = B**2*(1.0-mu**2)*f*cos2Phi
			logp += self.CQ[0] + p**2 * self.CQ[1] - p * self.CQ[2]
			
			p = B**2*(1.0-mu**2)*f*sin2Phi
			logp += self.CU[0] + p**2 * self.CU[1] - p * self.CU[2]
				
		return -0.5*logp/self.sigmaNoise**2
	
	def lnprior(self, theta):
		B, mu, f, phi = theta
		if 0.0 < B < 2500.0 and -1.0 < mu < 1.0 and 0.0 < f < 1.0 and -np.pi/2 < phi < np.pi/2:			
			return 0.0
		return -np.inf

	def lnprob(self, thetaIn, includeQU):
		theta = np.copy(thetaIn)
		
		lnJac = 0.0
		for i in range(4):
			theta[i] = self.sigmoid(thetaIn[i],self.limits[i,0],self.limits[i,1])
			lnJac += np.log(self.limits[i,1]-self.limits[i,0]) - thetaIn[i] - 2.0 * np.log(1.0 + np.exp(-thetaIn[i]))
		
		lp = self.lnprior(theta) + lnJac
		if not np.isfinite(lp):
			return -np.inf
		return lp + self.lnlike(theta, includeQU)
		
	def doInference(self, *args):
		
		BParML = 0.5 * self.CV[2] / self.CV[1]
		BPerpML = np.sqrt(0.5*np.sqrt(self.CQ[2]**2 + self.CU[2]**2) / self.CQ[1])
		phiML = 0.5 * np.arctan2(self.CU[2], self.CQ[2])
		ffML = 0.5
		BML = np.sqrt(BParML**2 + BPerpML**2) / ffML
		if (BML > 2500.0):
			BML = 2400.0
		muML = np.cos(np.arctan2(BPerpML,BParML))
				
		self.theta0 = np.asarray([BML, muML, ffML, phiML])*0.8		
				
		for i in range(4):
			self.theta0[i] = self.invSigmoid(self.theta0[i],self.limits[i,0],self.limits[i,1])
													
		ndim, nwalkers = 4, 50
		self.pos = [self.theta0 + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]		
		self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=args)
		self.sampler.run_mcmc(self.pos, 500)
							
		for i in range(4):
			self.sampler.flatchain[:,i] = self.sigmoid(self.sampler.flatchain[:,i],self.limits[i,0],self.limits[i,1])				

# CQ, CU and CV coefficients for each pixel, as shown in the Appendix A of the paper
dat = io.readsav('weakField.idl')

indMask = np.where(dat['mask'] == 1)
CV1Map, CV2Map, CV3Map = dat['cv1'][indMask[0],indMask[1]], dat['cv2'][indMask[0],indMask[1]], dat['cv3'][indMask[0],indMask[1]]
CQ1Map, CQ2Map, CQ3Map = dat['cq1'][indMask[0],indMask[1]], dat['cq2'][indMask[0],indMask[1]], dat['cq3'][indMask[0],indMask[1]]
CU1Map, CU2Map, CU3Map = dat['cu1'][indMask[0],indMask[1]], dat['cu2'][indMask[0],indMask[1]], dat['cu3'][indMask[0],indMask[1]]

# Random permutation
ind = np.random.permutation(len(CV1Map))

CV1Map = CV1Map[ind]
CV2Map = CV2Map[ind]
CV3Map = CV3Map[ind]

CQ1Map = CQ1Map[ind]
CQ2Map = CQ2Map[ind]
CQ3Map = CQ3Map[ind]

CU1Map = CU1Map[ind]
CU2Map = CU2Map[ind]
CU3Map = CU3Map[ind]

np.save('/scratch1/aasensio/HINODE/MAP_LITES/chains/randomPermutation.npy',ind)

posX = 0
for indX in range(500):
	dataV = []
	dataQUV = []
	for indY in range(1000):
		CV = [CV1Map[posX],CV2Map[posX],CV3Map[posX]]
		CQ = [CQ1Map[posX],CQ2Map[posX],CQ3Map[posX]]
		CU = [CU1Map[posX],CU2Map[posX],CU3Map[posX]]
		noise = 1.1e-3

# Estimate the max-likelihood value of the parameters using
# the weak-field approximation and a random value for the ff
			
		pixel = fitPixel(CQ, CU, CV, noise)
		print 'QUV  ', indX, indY, posX
		pixel.doInference(True)
		ind = np.random.permutation(25000)[0:1000]
		res = pixel.sampler.flatchain[ind, :]
		res = res[np.newaxis,:,:]
		dataQUV.append(res)
		
		print 'V    ', indX, indY, posX
		pixel.doInference(False)
		ind = np.random.permutation(25000)[0:1000]
		res = pixel.sampler.flatchain[ind, :]
		res = res[np.newaxis,:,:]
		dataV.append(res)
		posX += 1
				
	dataV = np.vstack(dataV)
	dataQUV = np.vstack(dataQUV)
	np.savez('/scratch1/aasensio/HINODE/MAP_LITES/chains/scan{0}'.format(indX),dataV,dataQUV)
