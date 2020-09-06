"""
extract trend, seasonality and level
seasonalities are given
detect changepoints in trend
based on https://github.com/peterroelants/notebooks/blob/master/probabilistic_programming/Changepoint%20detection.ipynb

TODO: mult/additive test
TODO: detect data gaps and fill them
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
import sklearn.linear_model as l_mdl
import statsmodels.api as sm
import itertools
import theano.tensor as tt
import theano
import pwlf
from scipy import stats
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from scipy.signal import butter, sosfilt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from Utilities import sys_utils as s_ut


def data_gaps(sy, fill_data=True):
    """
    find data gaps and fill them. Data imputation needs to factor in serial correlations. Should not randomly sample from y-values
    :param sy: time series of y-values
    :param fill_data: when true impute
    :return: np array with imputed values
    """
    z = pd.DataFrame({'is_nan': sy.isnull().values * 1})
    z.reset_index(inplace=True, drop=True)
    if z.loc[z.index.max(), 'is_nan'] == 1:  # add non-NaN at the end to bound the last sequence of NaNs
        z = pd.concat([z, pd.DataFrame({'is_nan': [0]})], axis=0)
        z.reset_index(inplace=True, drop=True)
    if z.loc[z.index.min(), 'is_nan'] == 1:  # add non-NaN at the start to bound the first sequence of NaNs
        z = pd.concat([pd.DataFrame({'is_nan': [0]}), z], axis=0)
        z.reset_index(inplace=True, drop=True)
    z['cs'] = z['is_nan'].cumsum()
    z['d'] = z['cs'].diff()  # each 0 is the end of a missing window or the continuation of non-NaNs
    z.fillna(0, inplace=True)  # fill first row
    w = z[z['d'] == 0].drop_duplicates()  # ends of missing windows only (use keep='first')
    w['gap_sz'] = w['cs'].diff()  # gap sizes
    print('WARNING: missing ' + str(sy.isnull().sum()) + ' y-values. max gap: ' + str(w['gap_sz'].max()))
    if fill_data:  # data imputation from the distribution will miss serial correlations
        print('filling missing values')
        sy.interpolate(limit_direction='both', inplace=True, method='spline', order=3)
    return sy.values


def fourier_series(n_pts, seasonality, do_beta=True):
    tm = np.arange(n_pts)
    p, n = seasonality[1:]
    x = 2 * np.pi * np.arange(1, n + 1) / p              # 2 pi n / p
    x = x * tm[:, None]                                  # (2 pi n / p) * t
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)   # n_pts X 2 * n
    if do_beta:  # return a random combination of the fourier components
        beta = np.random.normal(size=2 * n)
        return x * beta
    else:
        return x


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
    w_changepoints = list()
    r_changepoints, t_start = list(), 0
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
        r_lwr = np.quantile(trace['diff'], alpha)
        r_upr = np.quantile(trace['diff'], 1 - alpha)
        w_lwr = np.quantile(trace['w'][:, 0], alpha) - 0.5
        w_upr = np.quantile(trace['w'][:, 0], 1 - alpha) - 0.5
        if verbose:
            v1 = trace[rate][0].mean()
            v2 = trace[rate][1].mean()
            w = trace['w'][:, 0].mean()
            print('t_start: ' + str(t_start) + ' t_last: ' + str(t_last) +
                  ' w0: ' + str(np.round(w, 4)) + ' rate1: ' + str(np.round(v1, 4)) + ' rate2: ' + str(np.round(v2, 4)) +
                  ' r_upr: ' + str(np.round(r_upr, 4)) + ' r_lwr: ' + str(np.round(r_lwr, 4)) +
                  ' w_upr: ' + str(np.round(w_upr, 4)) + ' w_lwr: ' + str(np.round(w_lwr, 4))
                  )

        # detect changepoint by rate change
        if r_lwr * r_upr > 0:  # 0 is not included in [r_lwr, r_upr] with <ci> confidence
            if verbose:
                print('\t\t============================================= ' + type_ + ' rate changepoint detected at time: ' + str(t_last - 1) + ' with t_step: ' + str(t_step))
            r_changepoints.append(t_last - 1)
            t_start = t_last

        # detect changepoint by mixture change
        if w_lwr * w_upr > 0:  # 0.5 is not included in [w_lwr, w_upr] with <ci> confidence
            if verbose:
                print('\t\t============================================= ' + type_ + ' mixture changepoint detected at time: ' + str(t_last - 1) + ' with t_step: ' + str(t_step))
            w_changepoints.append(t_last - 1)
            t_start = t_last

        t_last += t_step
    return r_changepoints


def stl_decompose(ts_, period):
    # wrapper for the statsmodels STL function
    seasonal = period + (period % 2 == 0)
    result = STL(ts_, period=period, robust=True,
                 seasonal=seasonal, trend=None, low_pass=None,
                 seasonal_deg=1, trend_deg=1, low_pass_deg=0,
                 seasonal_jump=1, trend_jump=1, low_pass_jump=1).fit()
    return result.seasonal, result.trend, result.resid


def mstl_decompose(ts_, periods, mode, f_order=20):
    if mode is None:
        print('ERROR: invalid mode: ' + str(mode))
        return None
    elif mode.lower()[0] == 'a':
        return _mstl_decompose(ts_, periods, f_order=f_order)
    elif mode.lower()[0] == 'm':
        if np.min(ts_) <= 0.0:
            print('ERROR: multiplicative time series must be positive')
            return None
        else:
            a_stl_season, a_stl_trend, a_stl_level = _mstl_decompose(np.log(ts_), periods, f_order=f_order)        # log(ts) = a_stl_trend + a_stl_season + a_stl_level
            m_stl_tr = np.exp(a_stl_trend)                                                                         # ts = m_stl_trend_ * m_stl_season * m_stl_level with m_stl_* = exp(a_stl_*)
            m_stl_trend = sm.nonparametric.lowess(m_stl_tr, range(len(m_stl_tr)), frac=0.5, return_sorted=False)   # smooth the m_trend
            if np.min(m_stl_trend) < 0.0:
                m_stl_trend += (1.0e-6 - np.min(m_stl_trend))
            a_stl_season, a_stl_level = _mstl_sl(np.log(ts_), np.log(m_stl_trend), periods, f_order)
            m_stl_season = np.exp(a_stl_season)
            m_stl_level = np.exp(a_stl_level)
            return m_stl_season, m_stl_trend, m_stl_level
    else:
        print('ERROR: invalid mode: ' + str(mode))
        return None


def _mstl_decompose(ts_, periods, f_order=20):
    """
    multi-period STL decomposition:
    computes STL for each period in periods by building an OLS from the single period STL decompositions
    and derives the trend, season and level (residual, noise) from the OLS
    ts_: time series
    periods: list of periods to include in ts_. If None, do all periods <= len(ts_) / 3
    """
    periods = list(set([p for p in periods if p < len(ts_) / 2]))
    stl_dict = dict()
    for p in periods:
        p = int(p)
        _, stl_dict['trend_' + str(p)], _ = stl_decompose(ts_, period=int(p))

    df = pd.DataFrame(stl_dict)  # residuals are derived from the OLS
    X = df.values
    results = sm.OLS(ts_, X).fit()
    stl_trend = np.sum(np.array([X[:, ix] * par for ix, par in enumerate(results.params)]), axis=0)

    # the trend fits well but some seasonality information leask through.
    # We set a more accurate model for seasons and trend:
    # - remove seasonality component from trend with a LP filter
    # - using Fourier orders that minimize the average lag correlations that are non-zero
    # _, stl_trend = sm.tsa.filters.cffilter(stl_trend, low=min(periods) / 2, high=max(periods), drift=False)
    stl_trend = sm.nonparametric.lowess(stl_trend, range(len(stl_trend)), frac=1.0 / 2.0, return_sorted=False)  # smooth the m_trend
    stl_season, stl_level = _mstl_sl(ts_, stl_trend, periods, f_order)
    return stl_season, stl_trend, stl_level


def _mstl_sl(ts_, stl_trend, periods, f_order):
    c = itertools.combinations_with_replacement(range(1, f_order + 1), len(periods))
    d = [list(set(itertools.permutations(cx))) for cx in c]
    dall = [x for dx in d for x in dx]                       # all possible combinations of orders. e.g. dall[0] = [w0, w1,...] with w0 Fourier order for p0, w1 order for p1, ...
    y_train = ts_ - stl_trend   # seasons + noise (level)
    min_val, stl_season, order_opt, stl_level = np.inf, 0.0, None, 0.0
    for f_orders in dall:       # extract best seasonality model
        stl_seas, resi, val = _mstl_season(f_orders, periods, y_train)
        if val < min_val:       # save best f_orders
            min_val = val
            order_opt = f_orders
            stl_season = stl_seas
            stl_level = resi
        # print('orders: ' + str(f_orders) + ' val: ' + str(val))
    print('orders: ' + str(order_opt) + ' min_val: ' + str(min_val) + ' resid: ' + str(np.mean(stl_level)))
    return stl_season, stl_level


def _mstl_season(f_orders, periods, y_train):
    # mstl for periods and f_orders (y_train ~ seasons + noise)
    zf = dict()
    for ix, order in enumerate(f_orders):
        p = periods[ix]
        p_str = str(int(p))
        fx = fourier_series(len(y_train), [p_str, p, order], do_beta=False)  # len(y_train) X (2 * order)
        for w in range(2 * order):
            zf[p_str + '_' + str(w)] = fx[:, w]   # from 0 to order - 1, cos() components and from order to 2 * order - 1, sin() components
    sf = pd.DataFrame(zf)
    X_train = sf.values
    results = sm.OLS(y_train, X_train, missing='drop').fit()
    stl_season = results.predict(X_train)
    resid = results.resid
    # print(str(f_orders) + ' len: ' + str(len(sf)) + ' sf nulls: ' + str(sf.isnull().sum().sum()) +
    #       ' season: ' + str(np.shape(stl_season)) + ' season nulls: ' + str(pd.Series(stl_season).isnull().sum()) +
    #       ' resid: ' + str(np.shape(resid)) + ' resid nulls: ' + str(pd.Series(resid).isnull().sum())
    #       )
    val = nz_acf(resid, periods)
    return stl_season, resid, val


def nz_acf(resid, periods):
    # compute the RMS value of non-zero acf values in residuals
    acfv, ci = acf(resid, nlags=int(max(periods)), fft=True, alpha=0.05)
    p = ci[:, 0] * ci[:, 1]   # if neg, the acf is 0 with (1 - alpha) confidence
    nz_ = acfv[p > 0][1:]     # non-zero correlations (ignore 0 lag)
    return np.sqrt(np.mean(nz_ ** 2)) if len(nz_) > 0 else 0.0


def plot_decompose(stl_season, stl_trend, stl_level, season, trend, level, ts, title='STL Decomposition'):
    # plot STL decompositions
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
    if ts is not None:
        t = len(ts)
        axes[0].plot(np.arange(t), ts, 'b-')
        axes[0].legend(['time series'], frameon=False)
    axes[0].grid(True)

    legend = list()
    if trend is not None:
        t = len(trend)
        axes[1].plot(np.arange(t), trend, 'b-')
        legend += ['trend']
    if stl_trend is not None:
        t = len(stl_trend)
        axes[1].plot(np.arange(t), stl_trend, 'g-')
        legend += ['STL_trend']
    if len(legend) == 2:
        legend = [legend[0]]
        rmse = np.sqrt(np.mean((trend - stl_trend) ** 2))
        legend.append('STL_trend::rmse: ' + str(np.round(rmse, 2)))
    axes[1].legend(legend, frameon=False)
    axes[1].grid(True)

    legend = list()
    if season is not None:
        t = len(season)
        axes[2].plot(np.arange(t), season, 'b-')
        legend += ['season']
    if stl_season is not None:
        t = len(stl_season)
        axes[2].plot(np.arange(t), stl_season, 'g-')
        legend += ['STL_season']
    if len(legend) == 2:
        legend = [legend[0]]
        rmse = np.sqrt(np.mean((season - stl_season) ** 2))
        legend.append('STL_season::rmse: ' + str(np.round(rmse, 2)))
    axes[2].legend(legend, frameon=False)
    axes[2].grid(True)

    legend = list()
    if level is not None:
        t = len(level)
        axes[3].plot(np.arange(t), level, 'b-')
        legend += ['level']
    if stl_level is not None:
        t = len(stl_level)
        legend += ['STL_level']
        axes[3].plot(np.arange(t), stl_level, 'g-')
    if len(legend) == 2:
        legend = [legend[0]]
        rmse = np.sqrt(np.mean((level - stl_level) ** 2))
        legend.append('STL_level::rmse: ' + str(np.round(rmse, 2)))
    axes[3].legend(legend, frameon=False)
    axes[3].grid(True)
    fig.suptitle(title)


def plot_change(t_, y_, thetat_, cpt_hat, cpt):
    # y_: trend, level TS
    # t_: time
    # theta_t: parameter TS
    # cpt: list of change points
    _, ax = plt.subplots(1, 1, figsize=(15, 3))
    ax.plot(t_, y_)
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(thetat_, color='g')
    for c in cpt:
        ax2.axvline(c, color='k', lw=1, ls='dashed')
    for c in cpt_hat:
        ax2.axvline(c, color='r', lw=1, ls='dashed')
    plt.show()

