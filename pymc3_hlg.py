# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:02:16 2019

@author: Ying-Fang.Kao

Pymc3: GLM: Hierarchical Linear Regression 
"""

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import theano

data = pd.read_csv(pm.get_data('radon.csv'))
data['log_radon'] = data['log_radon'].astype(theano.config.floatX)
county_names = data.county.unique()
county_idx = data.county_code.values

n_counties = len(data.county.unique())

data[['county', 'log_radon', 'floor']].head()

#%%% Probabilistic Programming

# Unpooled/non-hierarchical model

with pm.Model() as unpooled_model:
    # Independent parameters for each county
    a = pm.Normal('a', 0, sigma = 100, shape = n_counties)
    b = pm.Normal('b', 0, sigma = 100, shape = n_counties)
    
    # Model error
    eps = pm.HalfCauchy('eps', 5)
    
    # Model prediction of radon level
    # a[county_idx] translates to a[0, 0, 0, 1, 1, ...], we thus link multiple household measures of a county
    # to its cooefficients.
    radon_est = a[county_idx] + b[county_idx]*data.floor.values

    # Data likelihood
    y = pm.Normal('y', radon_est, sigma = eps, observed = data.log_radon)
    
with unpooled_model:
    unpooled_trace = pm.sample(200)
    
# Hierarchical Model
'''Instead of creating models separately,
 the hierarchical model creates group parameters that consider the countys not as completely
 different but as having an underlying similarity. These distributions  are subsequently used to influence the 
 distribution of each county's alpha and beta''' 
 
with pm.Model() as hierarchical_model:
    # Hyperpriors for group nodes
    mu_a = pm.Normal('mu_a' , mu = 0, sigma = 100)
    sigma_a = pm.HalfNormal('sigma_a', 5)
    mu_b = pm.Normal('mu_b', mu = 0, sigma = 100)
    sigma_b = pm.HalfNormal('sigma_b', 5)
    
    # Intercept for each county, distributed around group mean mu_a
    # Above we just set mu and sd to a fixed value while here we
    # plug in a common group distribution for all a and b (which are vectors of lenght n_counties)
    a = pm.Normal('a', mu = mu_a, sigma = sigma_a, shape = n_counties)
    b = pm.Normal('b', mu = mu_b, sigma = sigma_b, shape = n_counties)
    
    # Model error
    eps = pm.HalfCauchy('eps', 5)
    
    radon_est = a[county_idx] + b[county_idx]*data.floor.values
    
    # Data likelihood
    radon_like = pm.Normal('radon_like', mu = radon_est,
                           sigma = eps, observed = data.log_radon)
    
with pm.Model() as hierarchical_model:
    hierarchical_trace = pm.sample(2000, tune = 2000, target_accept = .9)
    