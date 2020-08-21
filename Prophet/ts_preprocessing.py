"""
extract trend, seasonality and level
detect changepoints in trend
based on https://github.com/peterroelants/notebooks/blob/master/probabilistic_programming/Changepoint%20detection.ipynb
"""
import multiprocessing as mp
try:
    mp.set_start_method('fork')     # forkserver does not work
except RuntimeError:
    pass

import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import theano
from scipy import stats
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from Utilities import sys_utils as s_ut
from Bayes.Prophet import ts_generate as ts_gen


def level_model(y_old, y_new):
    # PyMC3 level changepoint model
    # level is modeled by Poisson RVs
    # y_old: older data points since the last changepoint
    # y_new: last win(10) datapoints
    mean_new = y_new.mean() if len(y_new) > 0 else None
    mean_old = y_old.mean() if len(y_old) > 0 else mean_new
    y_ = np.concatenate((y_old, y_new))
    y_obs = theano.shared(y_)
    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(2))
        lambda_ = pm.Exponential('lambda', lam=np.array([1.0 / mean_old, 1.0 / mean_new]), shape=(2,))
        components = pm.Poisson.dist(mu=lambda_, shape=(2, ))
        diff = pm.Deterministic('diff', lambda_[0] - lambda_[1])
        obs = pm.Mixture('obs', w=w, comp_dists=components, observed=y_obs)
    return model


def trend_model(y_old, y_new):
    # PyMC3 trend changepoint model
    # trend is modeled by Normal RVs
    # y_old: older data points since the last changepoint
    # y_new: last win(10) datapoints
    g_new = np.gradient(y_new)                     # observed trend
    g_old = np.gradient(y_old) if len(y_old) > 1 else g_new
    mu_new = g_new.mean() if len(g_new) > 0 else None
    mu_old = g_old.mean() if len(g_old) > 0 else mu_new
    sigma_new = max(1.0, g_new.std()) if len(g_new) > 0 else None
    sigma_old = max(1.0, g_old.std()) if len(g_old) > 0 else sigma_new
    y_ = np.concatenate((y_old, y_new))
    # g = np.gradient(y_)                     # observed trend
    # t_dens = np.abs(g) / np.sum(np.abs(g))  # empirical changepoint density at each point
    y_obs = theano.shared(y_)
    ts = np.array(range(1, 1 + len(y_)))  # start from 1 to deal with intercept
    t_arr = np.array([ts, ts]).T

    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(2))
        mu = pm.Normal('mu', np.array([mu_old, mu_new]), np.array([sigma_old, sigma_new]), shape=(2,))
        mu_t = pm.Deterministic('mu_t', t_arr * mu)
        tau = pm.Gamma('tau', 1.0, 1.0, shape=2)
        diff = pm.Deterministic('diff', mu[1] - mu[0])
        obs = pm.NormalMixture('obs', w, mu_t, tau=tau, observed=y_obs)
    return model


def get_changepoints(y, t_step=10, t_last=None, win=None, samples=1000, verbose=False, ci=0.95, type_='level'):
    # change point detector
    # for each new point(s) sample the model and get the 95% ci. If 0 is not in the ci bounds, it is a change point
    # y: time series
    # t_step: number of points added at each iteration
    # win: size of the new points considered
    # samples: PyMC3 samples
    # verbose: for printing
    # ci: confidence interval for a change
    # type_: level or trend
    if type_ == 'level':
        rate = 'lambda'
        if y.min() < 0.0:
            y -= y.min() + np.std(y) / 100.0  # level changes assume Poisson
    elif type_ == 'trend':
        rate = 'mu'
    else:
        print('ERROR: unknown type_: ' + str(type_))
        return list()

    if t_last is None:
        t_last = t_step
    if win is None:
        win = 10
    t_last = max(t_last, win)
    changepoints, t_start = list(), 0
    while t_last <= len(y):
        if len(y[t_start:t_last]) >= 2 * win:
            y_new, y_old = y[t_last - win:t_last], y[t_start:t_last - win]
        else:
            y_new, y_old = y[t_start:t_last], list()

        if type_ == 'level':
            model = level_model(y_old, y_new)
        elif type_ == 'trend':
            if len(y_new) > 1:
                model = trend_model(y_old, y_new)
            else:  # skip: no trend yet
                t_last += t_step
                continue
        else:
            print('ERROR: unknown type_: ' + str(type_))
            return None

        with model:
            step_method = pm.NUTS(target_accept=0.95, max_treedepth=15)
            with s_ut.suppress_stdout_stderr():
                trace = pm.sample(samples, step=step_method, progressbar=True, tune=1000)

        alpha = (1.0 - ci) / 2.0
        d_lwr = np.quantile(trace['diff'], alpha)
        d_upr = np.quantile(trace['diff'], 1 - alpha)
        w_lwr = np.quantile(trace['w'][:, 0], alpha) - 0.5
        w_upr = np.quantile(trace['w'][:, 0], 1 - alpha) - 0.5
        if verbose:
            v1 = trace[rate][0].mean()
            v2 = trace[rate][1].mean()
            w = trace['w'][:, 0].mean()
            print('t_start: ' + str(t_start) + ' t_last: ' + str(t_last) +
                  ' w0: ' + str(np.round(w, 4)) + ' rate1: ' + str(np.round(v1, 4)) + ' rate2: ' + str(np.round(v2, 4)) +
                  ' d_upr: ' + str(np.round(d_upr, 4)) + ' d_lwr: ' + str(np.round(d_lwr, 4)) +
                  ' w_upr: ' + str(np.round(w_upr, 4)) + ' w_lwr: ' + str(np.round(w_lwr, 4))
                  )

        if d_lwr * d_upr > 0 or w_lwr * w_upr > 0:  # 0 is not included in [d_lwr, d_upr] with <ci> confidence or same for w and 0.5
            if verbose:
                print('\t\t============================================= Level changepoint detected at time: ' + str(t_last - 1) + ' with t_step: ' + str(t_step))
            changepoints.append(t_last - 1)
            t_start = t_last
        t_last += t_step
    return changepoints


def stl_decompose(ts_, period, additive=True):
    # wrapper for the statsmodels function
    if additive is False:
        min_ = np.min(ts_)
        ts_ = np.log1p(ts_ - min_)
    result = STL(ts_, period=period, robust=True,
                 seasonal=7, trend=None, low_pass=None,
                 seasonal_deg=1, trend_deg=1, low_pass_deg=0,
                 seasonal_jump=1, trend_jump=1, low_pass_jump=1).fit()
    if additive is False:
        return np.exp(result.seasonal), np.exp(result.trend), np.exp(result.resid)
    else:
        return result.seasonal, result.trend, result.resid


def mstl_decompose(ts_, periods, additive=True):
    # multi-period STL
    # computes STL for each period in periods, builds an OLS from the single period STL decomposition and derives the trend, season and level from the OLS
    # ts_: time series
    # periods: list of periods in ts_
    if periods is None:
        periods = list(range(2, int(len(ts_) / 3)))
    periods = list(set(periods))
    stl_dict = dict()
    for p in periods:
        p = int(p)
        stl_dict['season' + str(p)], stl_dict['trend' + str(p)], level_ = stl_decompose(ts_, period=int(p), additive=additive)
        print('period: ' + str(p) + ' resid mean: ' + str(np.mean(level_)) + ' resid std: ' + str(np.std(level_)))

    df = pd.DataFrame(stl_dict)
    X = df.values
    results = sm.OLS(ts_, X).fit()
    stl_level = results.resid   # by definition
    stl_season = np.sum(np.array([X[:, i] * results.params[i] for i in range(0, 2 * len(periods), 2)]), axis=0)
    stl_trend = np.sum(np.array([X[:, i] * results.params[i] for i in range(1, 2 * len(periods), 2)]), axis=0)
    print('periods: ' + str(periods) + ' resid mean: ' + str(np.mean(stl_level)) + ' resid std: ' + str(np.std(stl_level)))
    return stl_season, stl_trend, stl_level


def plot_decompose(stl_season, stl_trend, stl_level, season, trend, level, title='STL Decomposition'):
    # plot STL decompositions
    fig, axes = plt.subplots(nrows=3, ncols=1)
    t = len(stl_season)
    axes[0].plot(np.arange(t), trend, 'b-', np.arange(t), stl_trend, 'g-')
    axes[0].legend(['trend', 'STL_trend'], frameon=False)
    axes[0].grid(True)
    axes[1].plot(np.arange(t), season, 'b-', np.arange(t), stl_season, 'g-')
    axes[1].legend(['season', 'STL_season'], frameon=False)
    axes[1].grid(True)
    axes[2].plot(np.arange(t), level, 'b-', np.arange(t), stl_level, 'g-')
    axes[2].legend(['level', 'STL_level'], frameon=False)
    axes[2].grid(True)
    fig.suptitle(title)
