"""
pre-processing of a TS before forecast.
Assumes that the seasonality periods are given.
The function process_TS() encompasses all the steps below
1) extract trend, seasonalities and level (STL decompositions)
    - the TS is converted to additive through multiple YJ transforms
    - seasonalities (periods, fourier order) are used in the final forecast
3) detect changepoints in trend using the trend in (1) (multiple methods)
4) return changepoints and sesonalities config
The data returned can be used to configure a forecasting engine (FB Prophet or PyMC3 prophet) for prediction.

Notes:
    - steps (1), (2) and (3) are done on YJ transformed data
    - For the forecast we use the actual TS but assume that the seasonality and change points do not change much with the transforms

changepoint detection solutions
- tt_changepoint(): fast, simple, generates more changepoints than actuals in general. Accuracy OK
- qr_changepoint(): fast, simple, generates more changepoints than actual in general. OK accuracy
- mcmc_changepoint: very slow, too many parameters but seems pretty accurate
- cu_changepoint(): change point detection based on cusum. no generic way to set up threshold and drift. Not very accurate
- rpt_changepoint(): change point detection based on ruptures. no generic way to set up penalty. Not very accurate
- pw_changepoint(): change point detection based on ruptures. no good way to regularize.


TODO: detect data gaps and fill them before STL decomposition
TODO: fill NaN with data_gaps() in pre_changepoint()
"""
import multiprocessing as mp
try:
    mp.set_start_method('fork')     # forkserver does not work
except RuntimeError:
    pass

import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import itertools
import theano
import pwlf
from scipy import stats as sps
from statsmodels.tools.eval_measures import aicc_sigma
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import PowerTransformer as BoxCox
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from Utilities import sys_utils as s_ut
import ruptures as rpt
from scipy import signal
from detecta import detect_cusum
from collections import defaultdict


def process_TS(ts, periods, f_order=20, do_plot=False):
    # returns stl decomposition, trend, seasonalities config (periods and fourier order) and changepoints
    # here >>>>> detect data gaps and fill then goes here <<<<<<<<<
    stl_season, stl_trend, stl_level, y, f, season_cfg = mstl_decompose(ts, periods, f_order=f_order)
    changepoints = tt_changepoint(f, window=25, alpha=0.01, do_plot=do_plot)  # qr_changepoint(f, window=25, alpha=0.01, do_plot=do_plot)
    return stl_season, stl_trend, stl_level, y, changepoints, season_cfg

# ############################################################
# ################ Utilities  ################################
# ############################################################


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


def rel_ext(acf_):
    # find relative extrema of the acf
    # the first non-empty array of extrema will contain the main period, if any
    rel_idx, sz = list(), len(acf_)
    while len(rel_idx) == 0 and sz > 0:
        rel_idx = signal.argrelextrema(acf_, np.greater, order=sz)[0]
        sz -= 1
    print('extrema periods: ' + str(rel_idx))
    return np.min(rel_idx) if len(rel_idx) > 0 else None


def pre_changepoint(ts_, do_plot=False):
    # clean the trend diff() TS of remaining frequencies
    f = pd.DataFrame({'trend': ts_})
    # f = pd.DataFrame({'trend': stl_trend})

    # remove noise from dtrend, which should be constant between level changes
    f['dtrend'] = f['trend'].diff()
    f['dtrend'].fillna(f.loc[1, 'dtrend'], inplace=True)
    acf_df = pd.DataFrame({'period': range(int(len(f) / 3))})
    acf_df['acf'] = acf_df['period'].apply(lambda x: f['dtrend'].autocorr(lag=int(x)))
    period = rel_ext(acf_df['acf'].values)
    print('main period: ' + str(period))
    f['dtrend_diff'] = f['dtrend'].diff(period)
    f['period'] = period
    f.dropna(inplace=True)  # but keep index, i.e. do not reset_index!!!!

    # ewa re-intoduces lagged correlations! nice try!
    # w = 0.05 if period is None else 1.0 / period
    # f['dtrend_dewa'] = f[['dtrend_diff']].ewm(alpha=w, adjust=False).mean()
    acf_df = pd.DataFrame({'period': range(int(len(f) / 3))})
    acf_df['acf'] = acf_df['period'].apply(lambda x: f['dtrend'].autocorr(lag=int(x)))
    acf_df['diff_acf'] = acf_df['period'].apply(lambda x: f['dtrend_diff'].autocorr(lag=int(x)))
    # acf_df['acf_dewa'] = acf_df['period'].apply(lambda x: f['dtrend_dewa'].autocorr(lag=int(x)))
    acf_df.set_index('period', inplace=True)
    if do_plot:
        acf_df.plot(grid=True, style='-+')
    return f


def to_additive(ts_):
    # turn a TS into additive by applying YJ transforms
    zf = pd.DataFrame({'y': ts_})
    last_diff, diff, ctr, y, best_ycol = np.inf, np.inf, 0, ts_[:, None], None
    while diff > 0.1 and ctr < 5:
        yj_obj = BoxCox(method='yeo-johnson')
        y = yj_obj.fit_transform(y)
        lbda = yj_obj.lambdas_[0]
        ycol = 'yj_' + str(np.round(lbda, 3))
        print('Yeo-Johnson lambda: ' + str(lbda) + ' ctr: ' + str(ctr) + ' ycol: ' + ycol)
        zf[ycol] = y[:, 0]
        diff = np.abs(1.0 - lbda)
        if diff < last_diff:
            best_ycol = ycol
            last_diff = diff
        ctr += 1
    return zf, best_ycol

# ############################################################
# ################ MCMC change detection #####################
# ############################################################


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
    y_obs = theano.shared(y_)
    ts = np.array(range(1, 1 + len(y_)))  # start from 1 to deal with intercept
    t_arr = np.array([ts, ts]).T

    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(2))
        mu = pm.Normal('mu', np.array([mu_old, mu_new]), np.array([sigma_old, sigma_new]), shape=(2,))
        mu_t = pm.Deterministic('mu_t', t_arr * mu)
        tau = pm.Gamma('tau', 1.0, 1.0, shape=2)
        diff = pm.Deterministic('diff', mu[1] - mu[0])                    # needed for PyMC3 model
        obs = pm.NormalMixture('obs', w, mu_t, tau=tau, observed=y_obs)   # needed for PyMC3 model
    return model


def mcmc_changepoint(y, t_step=10, t_last=None, win=None, samples=1000, verbose=False, ci=0.95, type_='level', detect='w'):
    # https://github.com/peterroelants/notebooks/blob/master/probabilistic_programming/Changepoint%20detection.ipynb
    # change point detector
    # for each new point(s) sample the model and get the 95% ci. If 0 is not in the ci bounds, it is a change point
    # y: time series
    # t_step: number of points added at each iteration
    # win: size of the new points considered
    # samples: PyMC3 samples
    # verbose: for printing
    # ci: confidence interval for a change
    # type_: level or trend
    # detect: 'w' (mixture weight) or 'r' (rate)
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

        if detect == 'r':                  # detect changepoint by rate change
            if r_lwr * r_upr > 0:  # 0 is not included in [r_lwr, r_upr] with <ci> confidence
                if verbose:
                    print('\t\t============================================= ' + type_ + ' rate changepoint detected at time: ' + str(t_last - 1) + ' with t_step: ' + str(t_step))
                changepoints.append(t_last - 1)
                t_start = t_last
        elif detect == 'w':                # detect changepoint by mixture change
            if w_lwr * w_upr > 0:  # 0.5 is not included in [w_lwr, w_upr] with <ci> confidence
                if verbose:
                    print('\t\t============================================= ' + type_ + ' mixture changepoint detected at time: ' + str(t_last - 1) + ' with t_step: ' + str(t_step))
                changepoints.append(t_last - 1)
                t_start = t_last
        else:
            print('ERROR: invalid detect parameter: ' + str(detect))
            return None
        t_last += t_step
    return changepoints


# ############################################################
# ################ T-Test change detection #####################
# ############################################################


def tt_changepoint(f, window=25, alpha=0.01, do_plot=False):
    # change point detection based on t-test
    # f, window = pre_changepoint(ts_, window=window)
    # ycol = 'dtrend_diff'                # used to detect changes in trend
    # ts = f[ycol].values
    ts = f['dtrend_diff'].values
    if f['period'].isnull().sum() == 0:
        window = f.loc[f.index[0], 'period']

    if do_plot:
        ax = plt.figure().add_subplot(111)
        f[['dtrend', 'dtrend_diff']].plot(grid=True, style='+-', ax=ax)
    start0, ctr = 0, 0
    end0 = start0 + window
    shift, start1, end1, cpt_arr = 1, start0, end0, list()
    while end1 <= len(ts):
        yk0 = ts[start0:end0]
        start1 += shift
        end1 += shift
        yk1 = ts[start1:end1]
        _, pval = sps.ttest_ind(yk0, yk1)  # compare the means
        if pval < alpha:
            cpt_arr.append(end1)
            print('end: ' + str(end1) + ' start: ' + str(start1) + ' pval: ' + str(np.round(pval, 4)) +
                  ' mean0: ' + str(np.round(np.mean(yk0), 4)) + ' mean1: ' + str(np.round(np.mean(yk1), 4)))
            if do_plot:
                ax.axvline(end1, color='r')
            ctr += 1
            start0 = end1
            end0 = start0 + window
            start1 = start0
            end1 = end0
    print('cpts: ' + str(ctr))
    return cpt_arr

# ############################################################
# ################ CUSUM change detection ####################
# ############################################################


def cu_changepoint(f, window=25):
    # change point detection based on cusum
    # no generic way to set up threshold and drift
    # f, window = pre_changepoint(ts_, window=window)
    # ycol = 'dtrend_diff'  # dtrend               # used to detect changes in trend
    # ts = f[ycol].values
    ts = f['dtrend_diff'].values
    if f['period'].isnull().sum() == 0:
        window = f.loc[f.index[0], 'period']
    min_ix = f.index.min()
    m = 8.0
    threshold = m * f['dtrend_diff'].std()
    thres = threshold
    while thres >= threshold / m:
        drift = f['dtrend_diff'].std()
        while drift > f['dtrend_diff'].std() / 10:
            ta, tai, taf, amp = detect_cusum(ts, threshold=thres, drift=drift, ending=False, show=False)
            print('thres: ' + str(thres) + ' drift: ' + str(drift) + ' ta: ' + str(min_ix + ta) + ' tai: ' + str(min_ix + tai) + ' taf: ' + str(taf))
            drift /= 2
        thres /= 2

# ############################################################
# ################ ruptures change detection #################
# ############################################################


def rpt_changepoint(f, window=25):
    # changepoint detection based on ruptures
    # NOT recommended: no guidance for penalty (pen)
    # f, window = pre_changepoint(ts_, window=window)
    # ycol = 'dtrend_diff'                # used to detect changes in trend
    # ts = f[ycol].values
    ts = f['dtrend_diff'].values
    if f['period'].isnull().sum() == 0:
        window = f.loc[f.index[0], 'period']
    algo = rpt.Pelt(model="rbf").fit(ts)
    for p in [0, 5, 10, 20, 50, 100]:
        result = algo.predict(pen=p)
        print(str(p) + ' ' + str(result))
    return None

# ############################################################
# ############ q-regression change detection #################
# ############################################################


def QR_fit(y):
    # QR fit for dtrend: should be constant, i.e. no slope in dtrend
    X = np.ones(len(y))
    res = sm.QuantReg(y, X).fit(0.5)
    slope = res.params[0]
    return slope


def qr_changepoint(f, window=25, alpha=0.01, do_plot=False):
    # changepoint detection based on quantile regression
    # f, window = pre_changepoint(ts_, window=window)
    # ycol = 'dtrend_diff'                # used to detect changes in trend
    # ts = f[ycol].values
    ts = f['dtrend_diff'].values
    if f['period'].isnull().sum() == 0:
        window = f.loc[f.index[0], 'period']

    if do_plot:
        ax = plt.figure().add_subplot(111)
        f[['dtrend', 'dtrend_diff']].plot(grid=True, style='+-', ax=ax)
    start0, ctr = 0, 0
    end0 = start0 + window
    shift, start1, end1, cpt_arr = 1, start0, end0, list()
    yk0 = ts[start0:end0]
    slope = QR_fit(yk0)
    y_upr0 = np.sum(yk0 > slope)
    while end1 <= len(ts):
        start1 += shift
        end1 += shift
        yk1 = ts[start1:end1]
        y_upr = np.sum(yk1 > slope)
        pval = sps.binom_test(y_upr, window, 0.5, alternative='two-sided')
        if pval < alpha:
            cpt_arr.append(end1)
            print('end: ' + str(end1) + ' start: ' + str(start1) + ' pval: ' + str(np.round(pval, 4)) +
                  '  init_ratio: ' + str(np.round(y_upr0 / window, 4)) + '  ratio: ' + str(np.round(y_upr / window, 4)))
            if do_plot:
                ax.axvline(end1, color='r')
            ctr += 1
            start0 = end1
            end0 = start0 + window
            yk0 = ts[start0:end0]
            slope = QR_fit(yk0)
            y_upr0 = np.sum(yk0 > slope)
            start1 = start0
            end1 = end0
    print('cpts: ' + str(ctr))
    return cpt_arr

# ############################################################
# #################### pwlf change detection #################
# ############################################################


def pw_changepoint(f, max_cpt=6, window=None):
    # ts is a STL trend from an addtive TS
    # best number of cpts is when the aic drop is max as nb increases
    # Not recommended: not clear how to regularize
    ts = f['dtrend_diff'].values
    if f['period'].isnull().sum() == 0:
        window = f.loc[f.index[0], 'period']
    if window is not None:
        f = pd.Series(ts)
        ts_ = f.rolling(window).mean()
        ts_.fillna(pd.Series(ts[:window]), inplace=True)
        ts = ts_.values
    nobs = len(ts)
    opt_val, opt_cp, aic_arr, cpt_arr = 0.0, list(), list(), list()
    my_pwlf = pwlf.PiecewiseLinFit(np.array(range(len(ts))), ts)
    d_res = defaultdict(list)
    for nb in range(2, max_cpt + 1):
        res = my_pwlf.fitfast(nb, pop=5)
        ssr = my_pwlf.ssr
        npars = my_pwlf.n_parameters
        sig2 = ssr / (nobs - npars)
        aic = aicc_sigma(sig2, nobs, nb, islog=False)
        d_res['nb'].append(nb - 1)
        d_res['ssr'].append(ssr)
        d_res['npars'].append(npars)
        d_res['sig2'].append(sig2)
        d_res['aic'].append(aic)
        aic_arr.append(aic)
        cpt = [int(x) for x in res]
        d_res['cpt'].append(cpt[1:-1])
        diff = np.inf if len(aic_arr) <= 1 else aic_arr[-1] - aic_arr[-2]
        cpt_arr.append(cpt)
        print('nb: ' + str(nb-1) + ' pars: ' + str(npars) + ' ssr: ' + str(ssr) + ' aic: ' + str(aic) + ' cpts: ' + str(cpt) + ' aicdiff: ' + str(diff))
    return pd.DataFrame(d_res)  # return list of estimated CPs

# ############################################################
# #################### STL decomposition  ####################
# ############################################################


def stl_decompose(ts_, period):
    # wrapper for the statsmodels STL function
    seasonal = period + (period % 2 == 0)
    result = STL(ts_, period=period, robust=True,
                 seasonal=seasonal, trend=None, low_pass=None,
                 seasonal_deg=1, trend_deg=1, low_pass_deg=0,
                 seasonal_jump=1, trend_jump=1, low_pass_jump=1).fit()
    return result.seasonal, result.trend, result.resid


def mstl_decompose(ts_, periods, f_order=20):
    # multi-period STL decomposition
    zf, ycol = to_additive(ts_)      # convert TS to an additive TS before the STL decomposition
    stl_season, stl_level, f, season_cfg = _mstl_decompose(zf[ycol].values[:, None], periods, f_order=f_order)

    # return STL values of the aditve TS, the additve TS amd the list of YJ lambdas used to turn input TS to an additve TS
    stl_trend = f['dtrend_diff'].values
    return stl_season, stl_trend, stl_level, zf['y'].values, f, season_cfg


def _mstl_decompose(ts_, periods, f_order=20):
    """
    multi-period STL decomposition:
    computes STL for each period in periods by building an OLS from the single period STL decompositions
    and derives the trend, season and level (residual, noise) from the OLS
    ts_: time series
    periods: list of periods to include in ts_. If None, do all periods <= len(ts_) / 3
    """
    periods = list(set([int(np.round(p, 0)) for p in periods if p < len(ts_) / 3]))
    stl_dict = dict()
    for p in periods:
        _, stl_dict['trend_' + str(p)], _ = stl_decompose(ts_, period=p)

    df = pd.DataFrame(stl_dict)
    X = df.values
    results = sm.OLS(ts_, X).fit()
    stl_trend = results.predict(X)      # the trend fits well but some seasonality information leaks through.
    f = pre_changepoint(stl_trend, do_plot=False)
    stl_trend = f['dtrend_diff']
    sl_resi = ts_ - stl_trend   # results.resid
    stl_season, stl_level, season_cfg = _mstl_sl(sl_resi, periods, f_order)
    return stl_season, stl_level, f, season_cfg


def _mstl_sl(sl_resi, periods, f_order):
    # find the fourier orders least correlated with the residual (on average)
    # return the result seasonality and level
    # sl_resi = TS - trend = season + level
    # f_order: max fourier order
    # periods: list of periods

    # build the tree of all fourier order combinations for the all the periods
    c = itertools.combinations_with_replacement(range(1, f_order + 1), len(periods))
    d = [list(set(itertools.permutations(cx))) for cx in c]
    dall = [x for dx in d for x in dx]                       # all possible combinations of orders. e.g. dall[0] = [w0, w1,...] with w0 Fourier order for p0, w1 order for p1, ...

    min_val, stl_season, order_opt, stl_level = np.inf, 0.0, None, 0.0
    for f_orders in dall:       # extract best seasonality model
        stl_seas, resi, val = _mstl_season(f_orders, periods, sl_resi)
        if val < min_val:       # save best f_orders
            min_val = val
            order_opt = f_orders
            stl_season = stl_seas
            stl_level = resi
    season_cfg = [(p, order_opt[ix]) for ix, p in enumerate(periods)]
    print('orders: ' + str(order_opt) + ' min_val: ' + str(min_val) + ' resid: ' + str(np.mean(stl_level)))
    return stl_season, stl_level, season_cfg


def _mstl_season(f_orders, periods, y_train):
    # mstl for a specific fourier order value for each period and f_orders
    # y_train ~ seasons + noise
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
    val = nz_acf(resid, periods)        # measures the avg non-zero correlation in the residuals, we want to minimize that
    return stl_season, resid, val


def nz_acf(resid, periods):
    # compute the RMS value of non-zero acf values in residuals
    acfv, ci = acf(resid, nlags=int(max(periods)), fft=True, alpha=0.05)
    p = ci[:, 0] * ci[:, 1]   # if p <= 0, the acf is 0 with (1 - alpha) confidence
    nz_ = acfv[p > 0][1:]     # non-zero correlations (ignore 0 lag)
    return np.sqrt(np.mean(nz_ ** 2)) if len(nz_) > 0 else 0.0

# ############################################################
# #################### plots  ################################
# ############################################################


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


def plot_change(t_, y_, thetat_, cpt, cpt_hat):
    # y_: trend, level TS
    # t_: time
    # theta_t: parameter TS
    # cpt: list of change points
    # cpt_hat: lis of estimated change points
    _, ax = plt.subplots(1, 1, figsize=(15, 3))
    ax.plot(t_, y_)
    ax.grid(True)
    if thetat_ is not None:
        ax2 = ax.twinx()
        ax2.plot(thetat_, color='g')
    else:
        ax2 = ax
    for c in cpt:
        ax2.axvline(c, color='k', lw=1, ls='dashed')
    for c in cpt_hat:
        ax2.axvline(c, color='r', lw=1, ls='dashed')
    plt.show()


