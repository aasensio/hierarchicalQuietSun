import numpy as np
import matplotlib.pyplot as pl
import scipy.special as sp
from matplotlib.ticker import MaxNLocator
from scipy.integrate import simps
import scipy.signal as sg

def logNormalAvgPrior(x, mu, sigma):
	pf = np.zeros(len(x))
	for i in range(len(x)):
		logy = -np.log(sigma) - np.log(x[i]) - (np.log(x[i]) - mu)**2 / (2.0*sigma**2)
		pf[i] = np.mean(np.exp(logy))
	return pf

def betaAvgPrior(x, alpha, beta, left, right):
	Beta = sp.beta(alpha, beta)
	pf = np.zeros(len(x))
	for i in range(len(x)):
		ylog = ( (1.0-alpha-beta) * np.log(right-left) - (sp.gammaln(alpha) + sp.gammaln(beta) - sp.gammaln(alpha+beta)) +
			(alpha-1.0) * np.log(x[i] - left) + (beta-1.0) * np.log(right - x[i]) )		
		pf[i] = np.mean(np.exp(ylog))
	return pf

samples = np.load('samplesHyperPar.npy')
ch = samples.T
ch = ch[:,np.random.permutation(ch.shape[1])]
for i in range(4):
	ch[i,:] = sg.medfilt(ch[i,:],kernel_size=7)
	
pl.close('all')

fig1 = pl.figure(num=1, figsize=(17,8))
pl.clf()

loop = 1
nTicks = 5
labels = [r'$\alpha_B$',r'$\beta_B$',r'$\alpha_\mu$',r'$\beta_\mu$',r'$\alpha_f$',r'$\beta_f$']
bellotOrozco = np.loadtxt('bellot_orozco.dat')

for i in range(4):
	ax = fig1.add_subplot(2,5,loop)
	ax.plot(ch[i,:], color='#969696')
	ax.set_xlabel('Iteration')
	ax.set_ylabel(labels[i])
	ax.xaxis.set_major_locator(MaxNLocator(nTicks))
	loop += 1
	ax = fig1.add_subplot(2,5,loop)
	ax.hist(ch[i,:], color='#507FED', normed=True, bins=20)
	#ax.hist(ch[i,:], color='#507FED', normed=True, cumulative=True, alpha=0.5)
	ax.set_xlabel(labels[i])
	ax.set_ylabel('p('+labels[i]+'|D)')
	ax.xaxis.set_major_locator(MaxNLocator(nTicks))
	loop += 1
	print np.mean(ch[i,:]), np.std(ch[i,:])
	if ((i+1) % 2 == 0):
		loop += 1
		
EX = np.exp(ch[0,:]+ch[1,:]**2/2.0)
EX2 = np.sqrt(np.exp(2.0*ch[0,:]+2.0*ch[1,:]**2))

print "E(B)={0} +- {1}".format(np.mean(EX),np.std(EX))
print "sqrt(E(B^2))={0} +- {1}".format(np.mean(EX2),np.std(EX2))

EX = np.mean((ch[2,:]-ch[3,:])/(ch[2,:]+ch[3,:]))
	
# Magnetic field strength
B = np.linspace(0.1,800,500)
pB = np.zeros(500)
alpha = ch[0,:]
beta = ch[1,:]
pB = logNormalAvgPrior(B, alpha, beta)
ax = fig1.add_subplot(2,5,5)
ax.plot(B,pB, color='#507FED', linewidth=2)
#pBTypeII = IGAvgPrior(B, np.mean(ch[0,:]), np.mean(ch[1,:]))
#ax.plot(B,pBTypeII, '--', color='#969696', linewidth=2)
ax.set_xlabel(r'B [G]')
ax.set_ylabel(r'$\langle$ p(B|D) $\rangle$')
ax.xaxis.set_major_locator(MaxNLocator(nTicks))

# Inclination
left = -1.0
right = 1.0
mu = np.linspace(left + 1e-2,right - 1e-2,100)
pmu = np.zeros(100)
alpha = ch[2,:]
beta = ch[3,:]
pmu = betaAvgPrior(mu, alpha, beta, left, right)
ax = fig1.add_subplot(2,5,10)
ax.plot(mu,pmu, color='#507FED', linewidth=2)
#pBTypeII = betaAvgPrior(mu, np.mean(ch[2,:]), np.mean(ch[3,:]), left, right)
#ax.plot(mu,pBTypeII, '--', color='#969696', linewidth=2)
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\langle$ p($\mu$|D) $\rangle$')
ax.xaxis.set_major_locator(MaxNLocator(nTicks))

# Filling factor
#left = 0.0
#right = 1.0
#f = np.linspace(left + 1e-4, right - 1e-4, 100)
#pf = np.zeros(100)
#alpha = ch[4,:]
#beta = ch[5,:]
#pf = betaAvgPrior(f, alpha, beta, left, right)
#ax = fig1.add_subplot(3,5,15)
#ax.plot(f,pf)
#ax.set_xlabel('f')
#ax.set_ylabel('p(f)')
#ax.xaxis.set_major_locator(MaxNLocator(6))

fig1.tight_layout()

fig1.savefig("posteriorHyper.pdf")

pl.close('all')
theta = np.linspace(0,180,100)
fig = pl.figure(num=1)
ax = fig.add_subplot(111)
normaliz = simps(np.sin(theta*np.pi/180.0) * pmu[::-1], x=theta)
ax.plot(theta, np.sin(theta*np.pi/180.0) * pmu[::-1] / normaliz, color='#507FED', linewidth=2)
normaliz = simps(np.sin(theta*np.pi/180.0), x=theta)
ax.plot(theta, np.sin(theta*np.pi/180.0) / normaliz, color='#969696', linewidth=2)
ax.plot(bellotOrozco[:,0], bellotOrozco[:,1], '--', color='#969696', linewidth=2)
ax.set_xlabel(r'$\theta$ [deg]')
ax.set_ylabel(r'$\langle$ p($\theta$|D) $\rangle$')
ax.axvline(90.0,color='k',linestyle='--', linewidth=2)
fig.savefig("posteriorInclination.pdf")