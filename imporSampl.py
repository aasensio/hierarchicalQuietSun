import numpy as np
import matplotlib.pyplot as pl
import scipy.special as sp
import scipy.misc as mi
import glob
from DIRECT import solve
import emcee
import pdb

class impSampling(object):
	def __init__(self, lower, upper):
		self.files = glob.glob('chains/scan*npz')
		self.files.sort()
		
		self.nFiles = len(self.files)
				
		nPoints = 100
		self.B = np.zeros((self.nFiles,nPoints,1000))
		self.mu = np.zeros((self.nFiles,nPoints,1000))		
		for i, f in enumerate(self.files):
			print i, f
			dat = np.load(f)
			self.B[i,:,:] = dat['arr_1'][0:nPoints,:,0]
			self.mu[i,:,:] = dat['arr_1'][0:nPoints,:,1]
			
		self.B = self.B.reshape((self.nFiles*nPoints,1000))
		self.mu = self.mu.reshape((self.nFiles*nPoints,1000))
		
		self.lower = np.asarray(lower)
		self.upper = np.asarray(upper)						
		
	def logLike(self, x):
		aB, bB, amu, bmu = x
		nu = 0.001
		
		cteB = aB * np.log(bB) - sp.gammaln(aB)
		ctemu = (1.0-amu-bmu) * np.log(2.0) - sp.betaln(amu,bmu)
		
		vecB = cteB - (aB + 1.0) * np.log(self.B) - bB / self.B
		vecmu = ctemu + (amu-1.0) * np.log(self.mu+1.0) + (bmu-1.0) * np.log(1.0-self.mu)
				
		lnP = - 2.5 * np.log(amu + bmu)
		lnP += -(nu-1.0)*np.log(bmu) - nu * bmu
		lnP += np.sum(mi.logsumexp(vecB + vecmu - np.log(1000.0), axis=1))
		print x, lnP
		
		return lnP
	
	def logLikeLN(self, x):
		aB, bB, amu, bmu = x
		nu = 0.001
		
		cteB = -np.log(bB) - 0.5 * np.log(2.0*np.pi)
		ctemu = (1.0-amu-bmu) * np.log(2.0) - sp.betaln(amu,bmu)
		
		vecB = cteB - np.log(self.B) - 0.5*(np.log(self.B)-aB)**2 / bB**2
		vecmu = ctemu + (amu-1.0) * np.log(self.mu+1.0) + (bmu-1.0) * np.log(1.0-self.mu)
				
		lnP = - 2.5 * np.log(amu + bmu)
		lnP += -(nu-1.0)*np.log(bmu) - nu * bmu
		lnP += np.sum(mi.logsumexp(vecB + vecmu - np.log(1000.0), axis=1))
		print x, lnP
		
		return lnP
	
	def evalDIRECT(self, x, user_data):
		return -self.logLike(x), 0
	
	def optimizeDIRECT(self):		
		x, fmin, ierror = solve(self.logLike, self.lower, self.upper, volper=1e-10, algmethod=1)
		
	def logPrior(self, x):		
		if (np.all(np.logical_and(self.lower < x, x < self.upper))):
			return 0.0
		return -np.inf
	
	def logPosterior(self, x):
		logP = self.logPrior(x)
		if (not np.isfinite(logP)):
			return -np.inf
		return logP + self.logLike(x)
	
	def logPosteriorLN(self, x):
		logP = self.logPrior(x)
		if (not np.isfinite(logP)):
			return -np.inf
		return logP + self.logLikeLN(x)
			
	def sample(self):
		ndim, nwalkers = 4, 10		
		p0 = np.zeros((nwalkers,ndim))		
		p0 = emcee.utils.sample_ball(np.asarray([4.5,0.5,2.5,2.3]), 0.1*np.ones(ndim), size=nwalkers)
		#for i in range(ndim):
			#p0[:,i] = np.random.uniform(low=self.lower[i],high=self.upper[i],size=nwalkers)			
			
		self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logPosteriorLN)
		self.sampler.run_mcmc(p0, 1000)
		
lower = [0.1, 0.1, 0.01, 0.01]
upper = [8.0, 4.0, 3.0, 3.0]
#x, fmin, ierror = solve(logLike, lower, upper, volper=1e-10, algmethod=1)
#print x, fmin

out = impSampling(lower, upper)
out.sample()

samples = out.sampler.flatchain[-3000:,:]

np.save('samplesHyperPar.npy',samples)

pl.close('all')
f = pl.figure(num=1)

for i in range(4):
	ax = f.add_subplot(2,2,i+1)
	ax.plot(samples[:,i])

#out.optimizeDIRECT()
