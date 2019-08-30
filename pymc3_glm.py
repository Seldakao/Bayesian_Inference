# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:03:43 2019

@author: Ying-Fang.Kao

(Generalised Linear and Hierarchical Linear Models in Pymc3)
"""

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from pymc3 import  *
import theano
import pandas as pd
from statsmodels.formula.api import glm as glm_sm
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

#%% Linear Regression

# generate data with known slopt and intercept
size = 50
true_intercept = 1
true_slope = 2
x = np.linspace(0,1,size)
y = true_intercept + x*true_slope + np.random.normal(scale=.5, size = size)
data = {'x': x, 'y': y}

# The glm.linear_component() function can be used to generate the output variable y_est and coefficients of the specified linear model
with Model() as model:
    lm = glm.LinearComponent.from_formula('y ~ x', data) # from pymc3
    sigma = Uniform('sigma', 0, 20)
    y_obs = Normal('y_obs', mu = lm.y_est, sigma = sigma, observed = y)
    trace = sample(2000, cores = 2)


plt.figure(figsize = (5,5))
plt.plot(x, y, 'x')
plot_posterior_predictive_glm(trace)

#%%# The code below produces the same outcome
'''Since there are a couple of general linear models that are being used over and over again 
(Normally distributed noise, logistic regression etc), 
the glm.glm() function simplifies the above step by creating the likelihood (y_obs) and its priors (sigma) for us. 
Since we are working in the model context, the random variables are all added to the model behind the scenes. 
This function also automatically finds a good starting point which it returns.'''
with Model() as model:
    GLM.from_formula('y ~ x', data)
    trace = sample(2000, cores = 2)

plt.figure(figsize = (5,5))
plt.plot(x , y, 'x')
plot_posterior_predictive_glm(trace)
#%% Robust GLM
# same model with few outliers in the data

x_out = np.append(x, [.1, .15, .2])
y_out = np.append(y, [8, 6, 9])
data_outlier = dict(x = x_out, y = y_out)

with Model() as modelL:
    GLM.from_formula('y ~ x', data_outlier)
    trace = sample(2000, cores = 2)

plt.figure(figsize=(5, 5))
plt.plot(x_out, y_out, 'x')
plot_posterior_predictive_glm(trace)
# Because the normal distribution does not have a lot of mass in the tails, an outlier will affect the fit strongly

'''Instead, we can replace the Normal likelihood with a student T distribution which has heavier tails 
and is more robust towards outliers. While this could be done with the linear_compoment() function and 
manually defining the T likelihood we can use the glm() function for more automation. 
By default this function uses a normal likelihood. 
To define the usage of a T distribution instead we can pass a family object that contains 
information on how to link the output to y_est (in this case we explicitly use the 
Identity link function which is also the default) and what the priors for the T distribution are.
Here we fix the degrees of freedom nu to 1.5.'''
#%%
with Model() as model_robust:
    family = glm.families.StudentT(link=glm.families.Identity(),
                                   priors={'nu': 1.5,
                                           'lam': Uniform.dist(0, 20)})
    GLM.from_formula('y ~ x', data_outlier, family=family)
    trace = sample(2000, cores=2)

plt.figure(figsize=(5, 5))
plt.plot(x_out, y_out, 'x')
plot_posterior_predictive_glm(trace)
#%% Hierarchical GLM

set_data = pd.read_csv(get_data('Guber1999data.txt'))

with Model() as model_sat:
    grp_mean = Normal('grp_mean', mu = 0, sigma = 10)
    grp_sd = Uniform('grp_sd', 0, 200)
    # Difine priors for intercept and regression coefficients
    # using mean of the depedent variable as the prior 
    priors = { 'Intercept': Normal.dist(mu = sat_data.sat_t.mean(), sigma = sat_data.sat_t.std()),
              'spend': Normal.dist(mu = grp_mean, sigma = grp_sd),
              'stu_tea_rat': Normal.dist(mu = grp_mean, sigma = grp_sd),
              'salary': Normal.dist(mu = grp_mean, sigma = grp_sd),
              'prcnt_take': Normal.dist(mu = grp_mean, sigma = grp_sd)
            }
    GLM.from_formula(
            'sat_t ~ spend + stu_tea_rat + salary + prcnt_take', sat_data, priors = priors
            )
    
    trace_sat = sample(2000, cores = 2)
    
scatter_matrix(trace_to_dataframe(trace_sat), figsize = (12,12))

#%% 
with Model() as model_sat:
    grp_mean = Normal('grp_mean', mu = 0, sigma = 10)
    grp_prec = Gamma('grp_prec', alpha = 1, beta = .1, testval = 1)
    slope = StudentT.dist(mu = grp_mean, lam = grp_prec, nu = 1)
    intercept = Normal.dist(mu = sat_data.sat_t.mean(), sigma = sat_data.sat_t.std())
    GLM.from_formula('sat_t ~ spend + stu_tea_rat + salary + prcnt_take', sat_data,
                     priors = {'Intercept': intercept, 'Regressor': slope})
    trace_sat = sample(100, cores = 2)
    
scatter_matrix(trace_to_dataframe(trace_sat), figsize = (12,12))

#%%
tdf_gain = 5 # what is this??
with Model() as model_sat:
    grp_mean = Normal('grp_mean', mu=0, sigma=10)
    grp_prec = Gamma('grp_prec', alpha=1, beta=.1, testval=1.)
    slope = StudentT.dist(mu=grp_mean, lam=grp_prec, nu=1) #grp_df)
    intercept = Normal.dist(mu=sat_data.sat_t.mean(), sigma=sat_data.sat_t.std())
    GLM.from_formula('sat_t ~ spend + stu_tea_rat + salary + prcnt_take', sat_data,
                priors={'Intercept': intercept, 'Regressor': slope})
    trace_sat = sample(500, cores = 2)
    
scatter_matrix(trace_to_dateframe(trace_sat), figsize = (12, 12))
# this code does not work, too little samples??

#%%%%%% Logistic Regression
htwt_data = pd.read_csv(get_data('HtWt.csv'))
print(htwt_data.head())

m = glm_sm('male ~ height + weight', htwt_data, family = sm.families.Binomial()).fit()
print(m.summary()) 

with Model() as model_htwt:
    GLM.from_formula('male ~ height + weight', htwt_data, family = glm.families.Binomial())
    trace_htwt = sample(100, cores = 2, init = "adapt_diag") # default init with jitter can cause prolem
trace_df = trace_to_dataframe(trace_htwt)    
print(trace_df.describe().drop('count').T)
scatter_mattrix(trace_df, figsize = (8,8))
print("P(weight < 0) = ", (trace_df['weight'] < 0).mean())
print("P(height < 0) = ", (trace_df['height'] < 0).mean())


#%% Bayesian Logistic Lasso
lp = Laplace.dist(mu = 0, b = 0.05)
x_eval = np.linespace(-5, .5, 300)
plt.plot(x_eval, theano.tensor.exp(lp.logp(x_eval)).eval())
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Laplace distribution')

with Model() as model_lasso:
    # Define priors for intercept and regression coefficients
    priors = { 'Intercept': Normal.dist(mu = 0, sigma = 50),
              'Regressor': Laplace.dist(mu = 0, b = 0.05)
            }
    GLM.from_formula('male ~ height + weight', htwt_data, family = glm.families.Binomail(), priors = priors)
    trace_lasso = sample(100, cores = 2, init = 'adapt_diag')
    
trace_df = trace_to_dataframe(trace_lasso)
scatter_matrix(trace_df, figsize = (8, 8));
print(trace_df.describe().drop('count').T)    