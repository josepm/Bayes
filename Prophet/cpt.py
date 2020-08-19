"""

"""
import multiprocessing as mp
try:
    mp.set_start_method('fork')     # forkserver does not work
except RuntimeError:
    pass

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import theano
from Utilities import sys_utils as s_ut


def set_model(y, t_last, t_start=0, exp_scale=None):
    print(str(t_start) + ' ' + str(t_last) + ' ' + str(exp_scale))
    y_ = np.copy(y[t_start:t_last])
    if exp_scale is None:
        exp_scale = y_.mean()
    t_ = theano.shared(np.array(list(range(len(y_)))))   # range(t_start, t_last)
    y_obs = theano.shared(y_)

    with pm.Model() as model:
        # Exponential priors
        lambda_1 = pm.Exponential('lambda_1', lam=1/exp_scale)
        lambda_2 = pm.Exponential('lambda_2', lam=1/exp_scale)

        # Change point
        changepoint = pm.DiscreteUniform('changepoint', lower=0, upper=t_last - t_start - 1, testval=t_last // 2)

        # First distribution is strictly before the other
        lamda_selected = tt.switch(t_ < changepoint, lambda_1, lambda_2)
        lambda_diff = pm.Deterministic('lambda_diff', lambda_2 - lambda_1)

        # Observations come from Poisson distributions with one of the priors
        obs = pm.Poisson('obs', mu=lamda_selected, observed=y_obs)
    return model


def get_changepoints(y, t_step=10, samples=1000, verbose=False):
    t_last = t_step
    t_start = 0
    changepoints = list()
    while t_last < len(y):
        exp_scale = y[0:t_last].mean()
        model = set_model(y, t_last, t_start=t_start, exp_scale=exp_scale)
        with model:
            # with s_ut.suppress_stdout_stderr():
                step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
                try:
                    trace = pm.sample(samples, step=step_method, progressbar=True, tune=1000)
                except:  # pm.SamplingError:
                    trace = pm.sample(samples, step=step_method, progressbar=True, tune=1000,init = 'adapt_diag')

        lwr = np.percentile(trace['lambda_diff'], 2.5)
        upr = np.percentile(trace['lambda_diff'], 97.5)
        if verbose:
            lbda1 = trace['lambda_1'].mean()
            lbda2 = trace['lambda_2'].mean()
            print('t_start: ' + str(t_start) + ' t_last: ' + str(t_last) + ' trace: ' + str(np.shape(trace))
                  + ' lbda1: ' + str(lbda1) + ' lbda2: ' + str(lbda2) + ' upr: ' + str(upr) + ' lwr: ' + str(lwr))

        if lwr * upr > 0:  # 0 is not included in [lwr, upr]
            changepoints.append(t_last - 1)
            t_start = t_last
            if verbose:
                print('==================================================== Changepoint detected at time: ' + str(changepoints[-1]))
        t_last += t_step
    return changepoints


# ##################################################################################
# ##################################################################################
# ##################################################################################
# ##################################################################################
# generate data
# https://gist.github.com/junpenglao/f7098c8e0d6eadc61b3e1bc8525dd90d
t = 100
t_step = 10
n_cpt = 1
tbreak = np.sort(np.random.randint(2 * t_step, t - 2 * t_step, n_cpt + 1))
theta = np.random.exponential(5, size=n_cpt + 1)
theta_t = np.zeros(t)
theta_t[:tbreak[0]] = theta[0]
print(str(0) + ' ' + str(0) + ' ' + str(tbreak[0]) + ' ' + str(theta[0]))
for i in range(1, n_cpt + 1):
    print(str(i) + ' ' + str(tbreak[i-1]) + ' ' + str(tbreak[i]) + ' ' + str(theta[i]))
    theta_t[tbreak[i-1]:tbreak[i]] = theta[i]
theta_t[tbreak[n_cpt]:] = theta[n_cpt]
print(str(n_cpt) + ' ' + str(tbreak[n_cpt]) + ' ' + str(t) + ' ' + str(theta[n_cpt]))
y = np.random.poisson(theta_t)         # may not handle the case where ll values are 0 in y_???

timesteps = range(t)
_, ax = plt.subplots(1, 1, figsize=(15, 3))
ax.plot(range(t), y)
ax.plot(theta_t)
plt.show()

cpts = get_changepoints(y, t_step=t_step, verbose=True)
print('actuals: ' + str(tbreak))
print('detected: ' + str(cpts))
