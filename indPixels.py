import emcee
import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as op
import scipy.io as io

def lnlike(theta, CQ, CU, CV, sigmaNoise):
	B, mu, f, phi = theta
	
	phi = np.deg2rad(phi)
	cos2Phi = np.cos(2.0*phi)
	sin2Phi = np.sin(2.0*phi)
    
	res = 0.0
    
	p = B*mu*f
	res -= CV[0] + p**2 * CV[1] - p * CV[2]
    
	#p = B**2*(1.0-mu**2)*f*cos2Phi
	#res -= CQ[0] + p**2 * CQ[1] - p * CQ[2]
    
	#p = B**2*(1.0-mu**2)*f*sin2Phi
	#res -= CU[0] + p**2 * CU[1] - p * CU[2]
    
	return 0.5*res/sigmaNoise**2

def lnprior(theta):
	B, mu, f, phi = theta
	if 0.0 < B < 2500.0 and -1.0 < mu < 1.0 and 0.0 < f < 1.0 and 0.0 < phi < 360.0:
		return 0.0
	return -np.inf

def lnprob(theta, CQ, CU, CV, sigmaNoise):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, CQ, CU, CV, sigmaNoise)

dat = io.readsav('/scratch1/aasensio/HINODE/MAP_LITES/weakField.idl')
CV1Map, CV2Map, CV3Map = dat['cv1'], dat['cv2'], dat['cv3']
CQ1Map, CQ2Map, CQ3Map = dat['cq1'], dat['cq2'], dat['cq3']
CU1Map, CU2Map, CU3Map = dat['cu1'], dat['cu2'], dat['cu3']

#pool = MPIPool()
#if not pool.is_master():
	#pool.wait()
	#sys.exit(0)

nX, nY = CV1Map.shape
for posX in range(1):
	data = []
	for posY in range(1):
		print posX, posY		

		CV = [CV1Map[posX,posY],CV2Map[posX,posY],CV3Map[posX,posY]]
		CQ = [CQ1Map[posX,posY],CQ2Map[posX,posY],CQ3Map[posX,posY]]
		CU = [CU1Map[posX,posY],CU2Map[posX,posY],CU3Map[posX,posY]]
		noise = 1.1e-3

# Estimate the max-likelihood value of the parameters using
# the weak-field approximation and a random value for the ff
		BParML = 0.5 * CV[2] / CV[1]
		BPerpML = np.sqrt(0.5*np.sqrt(CQ[2]**2 + CU[2]**2) / CQ[1])
		phiML = 0.5 * np.arctan2(CU[2], CQ[2])
		ffML = np.random.rand()
		BML = np.sqrt(BParML**2 + BPerpML**2) / ffML
		if (BML > 2500.0):
			BML = 2400.0
		muML = np.cos(np.arctan2(BPerpML,BParML))
		
		maxL = [BML, muML, ffML, phiML]

		ndim, nwalkers = 4, 100
		pos = [maxL + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(CQ, CU, CV, noise))
		sampler.run_mcmc(pos, 500)

		#res = sampler.chain[:, 490:, :].reshape((-1, ndim))
		#res = res[np.newaxis,:,:]
		#data.append(res)
				
	#data = np.vstack(data)
	#np.save('chains/scan{0}'.format(posX),data)