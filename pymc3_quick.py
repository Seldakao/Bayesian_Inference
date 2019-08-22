# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:06:48 2019

@author: Ying-Fang.Kao

Pymc3 quick start
"""

%matplotlib inline
import numpy as np
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('notebook')
plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

#%%

with pm.Model() as model:
    # unobserved RV
    mu = pm.Normal('mu', mu = 0, sigma = 1)
    # observed RV, needs data to be passed into it
    obs = pm.Normal('obs', mu = mu, sigma = 1, observed = np.random.randn(100))

# print variables    
model.basic_RVs  
model.free_RVs
model.observed_RVs 

#%% logp
model.logp({'mu': 0})

## the second case is not faster as the guide suggested
# logp is good when dynamic
%timeit model.logp({mu: 0.1})
logp = model.logp
%timeit model.logp({mu:0.1})

#%% Probability Distributions
help(pm.Normal)
dir(pm.distributions.mixture)
dir(model.mu)

#%% Deterministic transforms
# freely do the algebra
with pm.Model():
    x = pm.Normal('x', mu = 0, sigma = 1)
    y = pm.Gamma('y', alpha = 1, beta = 1)
    summed = x + y
    squared = x**2
    sined = pm.math.sin(x)
    
# to keep track of a transformed variable, use pm.Deterministic
with pm.Model():
    x = pm.Normal('x', mu = 0, sigma = 1)
    plus_2 = pm.Deterministic('x plus 2', x + 2)    
    
#%% Automatic transforms of bounded RVs
#In order to sample models more efficiently, PyMC3 automatically transforms bounded RVs to be unbounded.
with pm.Model() as model:
    x = pm.Uniform('x', lower = 0, upper = 1)
    
model.free_RVs
model.deterministics

#we can trun transfroms off
with pm.Model() as model:
    x = pm.Uniform('x', lower = 0, upper = 1, transform = None)
    
print(model.free_RVs)

# or specify different transformation other than the default
import pymc3.distributions.transforms as tr

with pm.Model() as model:
    # use the default log transformation
    x1 = pm.Gamma('x1', alpha = 1, beta = 1)
    # specified a different transformation
    x2 = pm.Gamma('x2', alpha = 1, beta = 1, transform = tr.log_exp_m1)

print('The default transformation of x1 is: ' + x1.transformation.name)
print('The user specified transformation of x2 is: ' + x2.transformation.name)
    

#%% Transformed distributions and changes of variables
class Exp(tr.ElemwiseTransform):
    name = "exp"

    def backward(self, x):
        return tt.log(x)

    def forward(self, x):
        return tt.exp(x)

    def jacobian_det(self, x):
        return -tt.log(x)


with pm.Model() as model:
    x1 = pm.Normal('x1', 0., 1., transform=Exp())
    x2 = pm.Lognormal('x2', 0., 1.)

lognorm1 = model.named_vars['x1_exp__']
lognorm2 = model.named_vars['x2']

_, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.linspace(0., 10., 100)
ax.plot(
    x,
    np.exp(lognorm1.distribution.logp(x).eval()),
    '--',
    alpha=.5,
    label='log(y) ~ Normal(0, 1)')
ax.plot(
    x,
    np.exp(lognorm2.distribution.logp(x).eval()),
    alpha=.5,
    label='y ~ Lognormal(0, 1)')
plt.legend();

#%% Ordered RV: x1, x2 ~ unifrom(0,1) and x1<x2
Order = tr.Ordered()
Logodd = tr.LogOdds()
chain_tran = tr.Chain([Logodd, Order])

with pm.Model() as m0:
    x = pm.Uniform(
        'x', 0., 1., shape=2,
        transform=chain_tran,
        testval=[0.1, 0.9])
    trace = pm.sample(5000, tune=1000, progressbar=False)

_, ax = plt.subplots(1, 2, figsize=(10, 5))
for ivar, varname in enumerate(trace.varnames):
    ax[ivar].scatter(trace[varname][:, 0], trace[varname][:, 1], alpha=.01)
    ax[ivar].set_xlabel(varname + '[0]')
    ax[ivar].set_ylabel(varname + '[1]')
    ax[ivar].set_title(varname)
plt.tight_layout()

#%% List of RVs higher-dimensional RVs

# bad example
with pm.Model():
    x = [pm.Normal('x_{}'.format(i), mu = 0, sigma = 1) for i in range(10)]
    
# good example
with pm.Model() as model:
    x = pm.Normal('x', mu, sigma = 1, shape = 10)

# we can index into x and do linear algebra
with model:
    y = x[0] * x[1]
    x.dot(x.T)

#%% Initialisation with test_values
#While PyMC3 tries to automatically initialize models it is sometimes helpful to define initial values for RVs. This can be done via the testval kwarg:
with pm.Model():
    x = pm.Normal('', mu = 0, sigma = 1, shape = 5)

print(x.tag.test_value)    

with pm.Model():
    x = pm.Normal('x', mu = 0, sigma = 1, shape = 5, testval= np.random.randn(5))
    
print(x.tag.test_value)

#%% Inference - Sampling and Variational 
with pm.Model() as model:
    mu = pm.Normal('mu', mu =0, sigma = 1)
    obs = pm.Normal('obs', mu = mu, sigma = 1, observed = np.random.randn(100))
    
    trace = pm.sample(1000, tune = 500)

#%%multiple chains in parallel using the cores kwarg
with pm.Model() as model:
    mu = pm.Normal('mu', mu = 0, sigma = 1)
    obs = pm.Normal('obs', mu = mu, sigma = 1, observed= np.random.randn(100)) 
    
    trace = pm.sample(cores=4)
trace['mu'].shape
#%%
# number of chans
trace.nchains
# get values of a single chain
trace.get_values('mu', chains = 1).shape

#%% Other sampler
# show only the methods with upper case in the beginning
list(filter(lambda x: x[0].isupper(), dir(pm.step_methods)))

# sampling methods can be passed to sample
with pm.Model() as model:
    mu = pm.Normal('mu', mu = 0, sigma = 1)
    obs = pm.Normal('obs', mu = mu, sigma = 1, observed = np.random.randn(100))
    
    step = pm.Metropolis()
    trace = pm.sample(1000, step = step)
    
#%% assign variables to different step methods
with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=1)
    sd = pm.HalfNormal('sd', sigma=1)
    obs = pm.Normal('obs', mu=mu, sigma=sd, observed=np.random.randn(100))

    step1 = pm.Metropolis(vars=[mu])
    step2 = pm.Slice(vars=[sd])
    trace = pm.sample(10000, step=[step1, step2], cores=4)    
    
#%% Analyze sampling results

# need to have arviz library for the plots
# traceplot
pm.traceplot(trace)
plt.show()
# gelman_rubin (R-hat)
pm.gelman_rubin(trace)
# forestplot
pm.forestplot(trace)
# plot_posterior 
pm.plot_posterior(trace)
# energyplot

with pm.Model() as model:
    x = pm.Normal('x', mu = 0, sigma = 1, shape = 100)
    trace = pm.sample(cores=4)
    
pm.energyplot(trace)    
#%% Variational inference
# this is much faster but less accurate - with pm.fit()
with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=1)
    sd = pm.HalfNormal('sd', sigma=1)
    obs = pm.Normal('obs', mu=mu, sigma=sd, observed=np.random.randn(100))

    approx = pm.fit()

approx.sample(500)    

#%% full-rank ADVI (Automatic Diffrentiation Variational Inference)
mu = pm.floatX([0., 0.])
cov = pm.floatX([[1, .5], [.5, 1.]])
with pm.Model() as model:
    pm.MvNormal('x', mu=mu, cov=cov, shape=2)
    approx = pm.fit(method='fullrank_advi')  
    
# equivalently, using object-oriented interface
with pm.Model() as model:
    pm.MvNormal('x', mu=mu, cov=cov, shape=2)
    approx = pm.FullRankADVI().fit()
    
plt.figure()    
trace = approx.sample(10000)
sns.kdeplot(trace['x'][:,0], trace['x'][:,1])

#%% Stein Variational Gradient Descent (SVGD) uses particles to estimate the posterior

w = pm.floatX([.2, .8])
mu = pm.floatX([-.3, .5])
sd = pm.floatX([.1, .1])
with pm.Model() as model:
    pm.NormalMixture('x', w=w, mu=mu, sigma =sd)
    approx = pm.SVGD(n_particles = 200, jitter = 1).fit()
    
plt.figure()
trace = approx.sample(10000)
sns.distplot(trace['x']);
    
#%% Posterior Predictive Sampling     
# The sample_posterior_predictive() function performs prediction on hold-out data and posterior predictive checks

data = np.random.randn(100)
with pm.Model() as model:
    mu = pm.Normal('mu', mu = 0, sigma = 1)
    sd = pm.HalfNormal('sd', sigma = 1)
    obs = pm.Normal('obs', mu = mu, sigma = sd, observed = data)
    
    trace = pm.sample()
    
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples = 5000)
    
# sample_posterior_predictive() returns a dict with a key for every observed node
post_pred['obs'].shape

fig, ax = plt.subplots()
sns.distplot(post_pred['obs'].mean(axis = 1), label = 'Posterior predictive means', ax = ax)
ax.axvline(data.mean(), ls = '--', color = 'r', label = 'True mean')
ax.legend()

#%% Predicting on hold-out data
# rely on theano.shared variable. These are theano tensors whose values can be changed later

import theano
x = np.random.randn(100)
y = x > 0
x_shared = theano.shared(x)
y_shared = theano.shared(y) 

with pm.Model() as model:
    coeff = pm.Normal('x', mu = 0, sigma = 1)
    logistic = pm.math.sigmoid(coeff * x_shared)
    pm.Bernoulli('obs', p=logistic, observed = y_shared)
    trace = pm.sample()
    
x_shared.set_value([-1, 0, 1])
y_shared.set_value([0,0,0]) # dummy values
#%%
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples = 500)    
    
post_pred['obs'].mean(axis=0)
    