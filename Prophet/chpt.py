from typing import List, Union

import multiprocessing as mp
try:
    mp.set_start_method('fork')     # forkserver does not work
except RuntimeError:
    pass

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from collections import defaultdict
from capacity_planning.PyMC3.pmprophet.pymc_prophet import plot as p_plt
from capacity_planning.PyMC3.pmprophet.pymc_prophet import utilities as ut


def set_times(data):
    max_ds = data['ds'].max()
    min_ds = data['ds'].min()
    ts_scale = (max_ds - min_ds).days
    dt = (data['ds'] - data['ds'].min()) / (data['ds'].max() - data['ds'].min())  # ratio of time deltas is a float
    return max_ds, min_ds, ts_scale, None, dt

def stick_breaking(beta_):
    _remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta_)[:-1]])
    return beta_ * _remaining


cpf = data[['ds', 'y']].set_index('ds').resample('7D').sum().reset_index()
_, _, _, _, cpf['t'] = set_times(cpf)  # add 't'
g = np.gradient(cpf['y'].values)  # trend
cpf['w_trend'] = np.abs(g) / np.sum(np.abs(g))  # changepoint density at each point
cpf['trend'] = g

K = len(cpf)   # max breakpoints
P = 1  #len(cpf)
with pm.Model() as model:      # The DP priors to obtain w, the cluster weights
    alpha = pm.Gamma('alpha', 1.0, 1.0, shape=1)
    beta = pm.Beta('beta', 1, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))

    # psi = pm.Uniform('psi', shape=(P, 1))
    # prob = pm.Uniform('prob', shape=(P, 1))
    # zb = pm.ZeroInflatedBinomial('zb', psi=psi, n=P, p=prob, shape=(P, K))
    # obs = pm.Mixture('obs', w, zb, shape=(P, 1), observed=cpf['w_trend'][:, None])

    # Prior on Bernoulli parameters, use Jeffrey's conjugate-prior
    # theta = pm.Beta('theta', 0.5, 0.5, shape=(P, K))
    theta = pm.Beta('theta', alpha=10.0, beta=0.5, shape=(P, K))
    obs = pm.Mixture('obs', w, pm.Bernoulli.dist(theta, shape=(P, K)), shape=(P, 1), observed=cpf['w_trend'][:, None])

    step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
    cpt_trace = pm.sample(1000, chains=None, step=step_method, tune=1000)
    cpt_smry = pm.summary(cpt_trace)
    pm.traceplot(cpt_trace)
    spp = pm.sample_posterior_predictive(cpt_trace, samples=1000, progressbar=False, var_names=['w', 'theta', 'alpha'])  #, 'alpha', 'obs'])
    # spp = pm.sample_posterior_predictive(cpt_trace, samples=1000, progressbar=False, var_names=['w', 'zb', 'prob', 'psi'])

K = 30
P =  len(cpf)  #
with pm.Model() as model:
    # The DP priors to obtain w, the cluster weights
    alpha = pm.Gamma('alpha', 1., 1.)
    beta = pm.Beta('beta', 1, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))

    # Prior on Bernoulli parameters
    theta = pm.Beta('theta', 0.5, 0.5, shape=(P, K))
    obs = pm.Mixture('obs', w, pm.Bernoulli.dist(theta, shape=(P, K)), shape=(P,1), observed=cpf['w_trend'][:, None])

    # sample
    step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
    cpt_trace = pm.sample(1000, chains=None, step=step_method, tune=1000)
    cpt_smry = pm.summary(cpt_trace)
    pm.traceplot(cpt_trace)
    spp = pm.sample_posterior_predictive(cpt_trace, samples=1000, progressbar=False, var_names=['w', 'theta', 'obs'])

n = len(cpf)
cpf = data[['ds', 'y']].set_index('ds').resample('7D').sum().reset_index()
_, _, _, _, cpf['t'] = set_times(cpf)  # add 't'
g = np.gradient(cpf['y'].values)  # trend
cpf['w_trend'] = np.abs(g) / np.sum(np.abs(g))  # changepoint density at each point
cpf['trend'] = g

alpha = 1.0 / cpf['w_trend'].mean()
beta = 1.0 / cpf['w_trend'].std()
t = np.range(0, n)
with pm.Model() as my_model:
    switchpoint = pm.DiscreteUniform("switchpoint", lower=0, upper=cpf['t'].max(), shape=10)
    for i in range(n):
        mu_name = 'mu_' + str(i)
        setattr(my_model, mu_name, pm.Exponential(mu_name, alpha))
        sd_name = 'sd_' + str(i)
        setattr(my_model, sd_name, pm.Exponential(sd_name, beta))

    t = 0
    for i in range(n):
        mu = pm.switch(switchpoint >= t, mu_1, mu_2 )
        var = pm.switch(switchpoint >= t, sd_1, sd_2 )
    obs = pm.Normal('x',mu=mu,sd=sd,observed=data)

with model:
    step1 = pm.NUTS( [mu_1, mu_2, sd_1, sd_2] )
    step2 = pm.Metropolis( [switchpoint] )
    trace = pm.sample( 10000, step=[step1,step2] )

traceplot(trace,varnames=['switchpoint','mu_1','mu_2', 'sd_1', 'sd_2'])
plt.show()

n = len(cpf)
with pm.Model():
    theta = pm.Beta('theta', 0.5, 0.5, shape=n)
    mu = pm.Bernoulli('mu', theta, shape=n)
    sig = pm.Deterministic('sig', mu * (1 - mu))
    obs = pm.Normal('obs', mu=mu, sigma=sig, observed=cpf['w_trend'][:, None])

    # sample
    step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
    cpt_trace = pm.sample(1000, chains=None, step=step_method, tune=1000)
    cpt_smry = pm.summary(cpt_trace)
    pm.traceplot(cpt_trace)
    spp = pm.sample_posterior_predictive(cpt_trace, samples=1000, progressbar=False, var_names=['mu', 'theta', 'sig'])


n = len(cpf)
cpf = data[['ds', 'y']].set_index('ds').resample('7D').sum().reset_index()
_, _, _, _, cpf['t'] = set_times(cpf)  # add 't'
g = np.gradient(cpf['y'].values)  # trend
cpf['w_trend'] = np.abs(g) / np.sum(np.abs(g))  # changepoint density at each point
cpf['trend'] = g

alpha = 1.0 / cpf['w_trend'].mean()
beta = 1.0 / cpf['w_trend'].std()
t = range(0, n)
N = len(cpf)
K = 5
with pm.Model() as my_model:
    # The DP priors to obtain w, the cluster weights
    # alpha = pm.Gamma('alpha', 1., 1.)
    # beta = pm.Beta('beta', 1, alpha, shape=K)
    # w = pm.Deterministic('w', stick_breaking(beta))

    sw_arr = [pm.DiscreteUniform('tau_0', lower=0, upper=N, shape=1)]
    for idx in range(1, len(cpf)):
        lwr = pm.Deterministic('lwr_' + str(idx), 1 + sw_arr[-1])
        sw_arr += [pm.DiscreteUniform('tau_' + str(idx), lower=lwr, upper=N, shape=1)]

    p = pm.Beta('p', 0.5, 0.5, shape=K + 1)
    rates = [pm.math.switch(sw_arr[i] >= t, p[i], p[i + 1]) for i in range(K)]
    k = pm.Binomial('k', p=rates, n=[1] * len(rates), observed=cpf['w_trend'].values)
    # mu_arr = [pm.Exponential('mu_' + str(idx), alpha, shape=1) for idx in range(K + 1)]
    # sd_arr = [pm.Exponential('sd_' + str(idx), beta, shape=1) for idx in range(K + 1)]

    # tau_mu = [pm.math.switch(sw_arr[0] >= t, mu_arr[0], mu_arr[1])]
    # tau_sd = [pm.math.switch(sw_arr[0] >= t, sd_arr[0], sd_arr[1])]
    # for idx in range(1, len(cpf)):
    #     tau_mu = [pm.math.switch(sw_arr[idx] >= t, mu_arr[idx - 1], mu_arr[idx])]
    #     tau_sd = [pm.math.switch(sw_arr[idx] >= t, sd_arr[idx - 1], sd_arr[idx])]
    # obs = pm.Normal('obs', mu=tau_mu[-1], sd=tau_sd[-1], observed=cpf['y'].values)



    # sample
    step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
    cpt_trace = pm.sample(1000, chains=None, step=step_method, tune=1000)
    cpt_smry = pm.summary(cpt_trace)
    pm.traceplot(cpt_trace)

    spp = pm.sample_posterior_predictive(cpt_trace, samples=1000, progressbar=False, var_names=['mu', 'theta', 'sig'])


n = len(cpf)
cpf = data[['ds', 'y']].set_index('ds').resample('7D').sum().reset_index()
_, _, _, _, cpf['t'] = set_times(cpf)  # add 't'
g = np.gradient(cpf['y'].values)  # trend
cpf['w_trend'] = np.abs(g) / np.sum(np.abs(g))  # changepoint density at each point
cpf['trend'] = g

alpha = 1.0 / cpf['w_trend'].mean()
beta = 1.0 / cpf['w_trend'].std()
t = range(0, n)
N = len(cpf)
K = 5
n_segs = K + 1
t = range(0, n)
with pm.Model() as sw_model:
    # define switch points
    tau = [t[0]]
    for n in range(n-1):
        tau.append(pm.DiscreteUniform('tau'+str(n+1), lower=eval('tau'+str(n)), upper=max(t)))
    tau.append(max(t))

    seg = np.zeros(K)
    for n in range(n_segs):
        if n==0:
            seg[(t>=tau[n])&(t<=tau[n+1])] = n
        else:
            seg[(x>tau[n])&(x<=tau[n+1])] = n
    seg = seg.astype(int)
    pi = HalfNormal('pi', sd = 0.8, shape = n_segs-1)
    b = HalfNormal('b', sd = 1, shape = n_segs-1)
    d = HalfNormal('d', sd = 4, shape = n_segs-1)


    sigma = HalfCauchy('sigma', beta=10)

    x_bar = x - tau[seg.astype(int)]
    likelihood = Normal('y', mu= pi[seg] / (1 + decline_rate[seg] * b[seg] * x_bar) ** (1/b[seg]), sd=sigma, observed=y)

    trace = sample(500, njobs=4,progressbar=True)




mu = np.log(y.mean())
sd = np.log(y.std())
t_ = np.linspace(0., 1., t)
nbreak = 3

with pm.Model() as m:
    lambdas = pm.Normal('lambdas', mu, sd=sd, shape=nbreak)
    trafo = Composed(pm.distributions.transforms.LogOdds(), Ordered())
    b = pm.Beta('b', 1., 1., shape=nbreak - 1, transform=trafo, testval=[0.4, 0.6])
    # index_t = tt.switch(tt.gt(t_, b[0]) * tt.lt(t_, b[1]), 1, 0) + tt.switch(tt.gt(t_, b[1]), 2, 0)

    index_t = tt.switch(tt.gt(t_, b[0]) * tt.lt(t_, b[1]), 1, 0)
    for idx in range(1, nbreak - 1):
        index_t +=  tt.switch(tt.gt(t_, b[1]), 2, 0)

    theta_ = pm.Deterministic('theta', tt.exp(lambdas[index_t]))
    obs = pm.Poisson('obs', theta_, observed=y)

    # sample
    step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
    cpt_trace = pm.sample(1000, chains=None, step=step_method, tune=1000)
    cpt_smry = pm.summary(cpt_trace)
    pm.traceplot(cpt_trace)

https://gist.github.com/junpenglao/f7098c8e0d6eadc61b3e1bc8525dd90d
t = 1000
n_cpt = 3
tbreak = np.sort(np.random.randint(100, 900, n_cpt + 1))
theta = np.random.exponential(25, size=n_cpt + 1)
theta_t = np.zeros(t)
theta_t[:tbreak[0]] = theta[0]
print(str(0) + ' ' + str(0) + ' ' + str(tbreak[0]) + ' ' + str(theta[0]))
for i in range(1, n_cpt + 1):
    print(str(i) + ' ' + str(tbreak[i-1]) + ' ' + str(tbreak[i]) + ' ' + str(theta[i]))
    theta_t[tbreak[i-1]:tbreak[i]] = theta[i]
theta_t[tbreak[n_cpt]:] = theta[n_cpt]
print(str(n_cpt) + ' ' + str(tbreak[n_cpt]) + ' ' + str(t) + ' ' + str(theta[n_cpt]))
y = np.random.poisson(theta_t)

_, ax = plt.subplots(1, 1, figsize=(15, 3))
ax.plot(range(t), y)
ax.plot(theta_t)

def logistic(L, x0, k=500, t_=np.linspace(0., 1., 1000)):
    return L / (1 + tt.exp(-k * (t_ - x0)))
import pymc3.distributions.transforms as tr

with pm.Model() as m2:
    lambda0 = pm.Normal('lambda0', mu, sd=sd)
    lambdad = pm.Normal('lambdad', 0, sd=sd, shape=n_cpt)
    xform = tr.Chain([tr.LogOdds(), tr.Ordered()])
    tv = np.random.uniform(low=0.0, high=1.0, size=n_cpt)
    b = pm.Beta('b', 1., 1., shape=n_cpt, transform=xform , testval=tv)
    xx = lambda0
    for i in range(n_cpt):
        xx += logistic(lambdad[i], b[i])
    dx = tt.exp(xx)
    theta_ = pm.Deterministic('theta', dx)
    obs = pm.Poisson('obs', theta_, observed=y)

    # sample
    step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
    cpt_trace = pm.sample(1000, chains=None, step=step_method, tune=1000)
    cpt_smry = pm.summary(cpt_trace)
    pm.traceplot(cpt_trace)


_, ax = plt.subplots(1, 1, figsize=(15, 3))
ax.plot(range(t), y)
ax.plot(cpt_trace['theta'].T, alpha=.01, color='k')
ax.plot(theta_t)
