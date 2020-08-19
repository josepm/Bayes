"""
change point detection
based on https://github.com/peterroelants/notebooks/blob/master/probabilistic_programming/Changepoint%20detection.ipynb
- level change detection
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


def level_model(y_):
    exp_scale = y_.mean()
    y_obs = theano.shared(y_)

    with pm.Model() as model:
        # Priors
        w = pm.Dirichlet('w', a=[1.0, 1.0])
        lambda_ = pm.Exponential('lambda', lam=1.0 / exp_scale, shape=2)
        components = pm.Poisson.dist(mu=lambda_, shape=2)
        diff = pm.Deterministic('diff', lambda_[0] - lambda_[1])
        obs = pm.Mixture('obs', w=w, comp_dists=components, observed=y_obs)
    return model


def level_changepoints(y, t_step=10, samples=1000, verbose=False, ci=0.95):
    t_last = t_step
    t_start = 0
    changepoints = list()
    while t_last < len(y):
        y_ = np.copy(y[t_start:t_last])
        model = level_model(y_)
        with model:
            step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
            with s_ut.suppress_stdout_stderr():
                trace = pm.sample(samples, step=step_method, progressbar=True, tune=1000)

        alpha = (1.0 - ci) / 2.0
        lwr = np.quantile(trace['diff'], alpha)
        upr = np.quantile(trace['diff'], 1 - alpha)
        if verbose:
            v1 = trace['lambda'][0].mean()
            v2 = trace['lambda'][1].mean()
            print('t_start: ' + str(t_start) + ' t_last: ' + str(t_last)
                  + ' v1: ' + str(v1) + ' v2: ' + str(v2) + ' upr: ' + str(upr) + ' lwr: ' + str(lwr))

        if lwr * upr > 0:  # 0 is not included in [lwr, upr]
            changepoints.append(t_last - 1)
            t_start = t_last
            if verbose:
                print('\t\t==================================================== Level changepoint detected at time: ' + str(changepoints[-1]))
        t_last += t_step
    return changepoints


def trend_model(y_):
    g = np.gradient(y_)                     # observed trend
    t_dens = np.abs(g) / np.sum(np.abs(g))  # changepoint density at each point
    mu = g.mean()
    sigma = max(1.0, g.std())  # must be > 0!!
    y_obs = theano.shared(y_)
    ts = np.array(range(1, 1 + len(y_)))
    t_arr = np.array([ts, ts]).T

    with pm.Model() as model:
        w = pm.Dirichlet('w', a=[1.0, 1.0])
        mu = pm.Normal('mu', mu, sigma, shape=2)
        diff = pm.Deterministic('diff', mu[1] - mu[0])
        mu_t = pm.Deterministic('mu_t', t_arr * mu)

        tau = pm.Gamma('tau', 1.0, 1.0, shape=2)
        obs = pm.NormalMixture('obs', w, mu_t, tau=tau, observed=y_obs)
    return model


def trend_changepoints(y, t_step=10, samples=1000, verbose=False, ci=0.95):
    t_last = t_step
    t_start = 0
    changepoints = list()
    while t_last < len(y):
        y_ = np.copy(y[t_start:t_last])
        model = trend_model(y_)
        with model:
            step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
            with s_ut.suppress_stdout_stderr():
                trace = pm.sample(samples, step=step_method, progressbar=True, tune=1000, init="adapt_diag")

        alpha = (1.0 - ci) / 2.0
        lwr = np.quantile(trace['diff'], alpha)
        upr = np.quantile(trace['diff'], 1 - alpha)
        if verbose:
            v1 = trace['mu'][0].mean()
            v2 = trace['mu'][1].mean()
            print('t_start: ' + str(t_start) + ' t_last: ' + str(t_last) + ' v1: ' + str(v1) + ' v2: ' + str(v2) + ' upr: ' + str(upr) + ' lwr: ' + str(lwr))

        if lwr * upr > 0:  # 0 is not included in [lwr, upr]
            changepoints.append(t_last - 1)
            t_start = t_last
            if verbose:
                print('\t\t==================================================== Trend changepoint detected at time: ' + str(changepoints[-1]))
        t_last += t_step
    return changepoints


# ##################################################################################
# ##################################################################################
# ##################################################################################
# ##################################################################################
# generate test data

def level_change_data(n_pts, t_step=10, n_cpt=1, noise_level=0.0, slow_bleed=0.0):
    # generate data for level changes
    # n_pts: number of points
    # t_step: points between checks
    # n_cpt: number of level changes
    # noise level: relative noise level around Poisson rate. 0.0 <= noise_level < 1.0
    # slow_bleed: relative Poisson rate change over time. -1.0 <= slow_bleed << 1.0. Note slow_bleed != 0 forces n_cpt = 0 and t_step = 1
    # https://gist.github.com/junpenglao/f7098c8e0d6eadc61b3e1bc8525dd90d
    if slow_bleed != 0.0:
        n_cpt = 0
        t_step = 1
    tbreak_ = np.sort(np.random.randint(t_step, n_pts - 2 * t_step, n_cpt))
    t_ = np.array(range(n_pts))

    theta = np.random.exponential(5, size=n_cpt + 1)
    if slow_bleed == 0.0:
        thetat_ = set_breaks(n_pts, tbreak_, theta)
    else:
        thetat_ = theta[0] * np.array([(1 + slow_bleed) ** n for n in t_])
    if noise_level > 0.0:
        noise_mult = np.random.choice([noise_level, -noise_level], size=n_pts)
        noise = thetat_ * noise_mult
    else:
        noise = np.zeros(n_pts)
    y_ = np.random.poisson(thetat_ + noise)
    return y_, tbreak_, thetat_


def trend_change_data(n_pts, t_step=10, n_cpt=1, noise_level=0.0, slow_bleed=0.0):
    # generate data for trend changes
    # t: number of points
    # t_step: points between checks
    # n_cpt: number of trend changes
    # noise level: relative noise level. 0.0 <= noise_level < 1.0
    # slow_bleed: relative trend change over time. -1.0 << slow_bleed << 1.0. Note slow_bleed != 0 forces n_cpt = 0 and t_step = 1
    # https://gist.github.com/junpenglao/f7098c8e0d6eadc61b3e1bc8525dd90d
    if slow_bleed != 0.0:
        n_cpt = 0
        t_step = 1
    tbreak_ = np.sort(np.random.randint(t_step, n_pts - 2 * t_step, n_cpt))
    t_ = np.array(range(t))

    mu, sigma = 0, 3
    theta = np.random.normal(mu, sigma,  size=n_cpt + 1)
    if slow_bleed == 0.0:
        thetat_ = set_breaks(n_pts, tbreak_, theta)
    else:
        thetat_ = theta[0] * np.array([(1 + slow_bleed) ** n for n in t_])
    y_ = np.cumsum(thetat_)
    noise = np.random.normal(0, sigma / noise_level, size=len(t_)) if noise_level > 0 else 0.0
    return y_ + noise, tbreak_, thetat_


def plot_change(y_, t_, thetat_, cpt):
    _, ax = plt.subplots(1, 1, figsize=(15, 3))
    ax.plot(t_, y_)
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(thetat_, color='g')
    for c in cpt:
        ax2.axvline(c, color='r', lw=1, ls='dashed')
    plt.show()


def set_breaks(n_pts, tbreak_, theta):
    thetat_ = np.zeros(n_pts)
    thetat_[:tbreak_[0]] = theta[0]
    print('index: ' + str(0) + ' start: ' + str(0) + ' end: ' + str(tbreak_[0]) + ' value: ' + str(theta[0]))
    for i in range(1, n_cpt):
        print('index: ' + str(i) + ' start: ' + str(tbreak_[i-1]) + ' end: ' + str(tbreak_[i]) + ' value: ' + str(theta[i]))
        thetat_[tbreak_[i-1]:tbreak_[i]] = theta[i]
    thetat_[tbreak_[n_cpt - 1]:] = theta[n_cpt]
    print('index: ' + str(n_cpt) + ' start: ' + str(tbreak_[n_cpt - 1]) + ' end: ' + str(t) + ' value: ' + str(theta[n_cpt]))
    return thetat_


# ##################################################################################
# ##################################################################################
# ##################################################################################
# ##################################################################################

# level change
t = 150
t_step = 10
n_cpt = 3
y, tbreak, theta_t = level_change_data(t, t_step=t_step, n_cpt=n_cpt, noise_level=0.0, slow_bleed=0.0)

cpts = level_changepoints(y, t_step=t_step, verbose=True)
print('level change actuals: ' + str(tbreak))
print('level changes detected: ' + str(cpts))
plot_change(y, np.array(range(t)), theta_t, cpts)

# trend change
t = 150
t_step = 10
n_cpt = 3
y, tbreak, theta_t = trend_change_data(t, t_step=t_step, n_cpt=n_cpt, noise_level=0.0, slow_bleed=0.0)

cpts = trend_changepoints(y, t_step=t_step, verbose=True)
print('trend change actuals: ' + str(tbreak))
print('trend changes detected: ' + str(cpts))
plot_change(y, np.array(range(t)), theta_t, cpts)
