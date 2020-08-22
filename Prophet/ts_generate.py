"""
functions to generate fake time series
"""
import sys
import numpy as np
import pandas as pd


def gen_ts_(n_pts, type_, n_cpt=1, noise_level=0.0, slow_bleed=0.0):
    # generate data for trend changes
    # n_cpt: number of points
    # type_: 'level' or 'trend'
    # n_cpt: number of trend changes
    # noise level: relative noise level. 0.0 <= noise_level < 1.0
    # slow_bleed: relative trend/level change over time. -1.0 << slow_bleed << 1.0.
    # returns ts, change times and a ts with the model param values
    w_step = 10
    tbreak_ = np.sort(np.random.randint(w_step, n_pts - w_step, n_cpt))

    if len(tbreak_) == 0:
        tbreak_ = np.array([0])
    t_ = np.array(range(n_pts))
    sigma = 5
    theta = get_theta(type_, n_cpt, sigma)
    print('\nthetat_::::' + str(type_))
    thetat_ = set_breaks(n_pts, tbreak_, theta, slow_bleed)
    if noise_level > 0.0:
        noise = np.random.normal(0, sigma * noise_level, size=n_pts)
        if type_ == 'level':
            noise = np.abs(noise)  # half-normal
        thetat_ += noise
    y_ = np.cumsum(thetat_) if type_ == 'trend' else np.random.poisson(thetat_)
    return y_, tbreak_, thetat_


def get_theta(type_, n_cpt, sigma):
    # generate model parameters
    if type_ == 'trend':
        return np.random.normal(0, sigma / 10, size=n_cpt + 2)  # array with slopes
    elif type_ == 'level':
        return np.random.exponential(sigma, size=n_cpt + 2)  # array with rates
    else:
        print('ERROR: invalid type_')
        sys.exit(0)


def gen_ts(n_pts, additive=True, seasonalities=None, trend_cpt=-1, level_cpt=-1, noise_level=0.0, slow_bleed=0.0):
    # generate data for trend changes
    # t: number of points
    # trend_cpt: no trend if -1 otherwise have a trend with trend_cpt change points
    # level_cpt: no level if -1 otherwise have a level with level_cpt change points
    # seasonalities: list with season info (name, period, fourier order)
    # noise level: relative noise level. 0.0 <= noise_level < 1.0
    # slow_bleed: relative trend change over time. -1.0 << slow_bleed << 1.0.
    # returns the new ts, seasons_ts, trend_ts, level_ts, change times for each trend and level, model param ts for each trend and level
    n_val = 0.0 if additive is True else 1.0
    seasons_ = seasonal_data(n_pts, seasonalities) if seasonalities is not None else n_val
    trend_ = gen_ts_(n_pts, 'trend', n_cpt=trend_cpt, noise_level=noise_level, slow_bleed=slow_bleed) if trend_cpt >= 0 else [n_val, None, None]
    level_ = gen_ts_(n_pts, 'level', n_cpt=level_cpt, noise_level=noise_level, slow_bleed=slow_bleed) if level_cpt >= 0 else [n_val, None, None]
    ts_ = seasons_ + trend_[0] + level_[0] if additive else trend_[0] * seasons_[0] * level_[0]
    return ts_, seasons_, trend_[0], level_[0], trend_[1], level_[1], trend_[2], level_[2]


def fourier_series(n_pts, seasonality, do_beta=True):
    tm = np.arange(n_pts)
    p, n = seasonality[1:]
    x = 2 * np.pi * np.arange(1, n + 1) / p       # 2 pi n / p
    x = x * tm[:, None]                           # 2 pi n / p * t
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    if do_beta:  # random combination of the fourier components
        beta = np.random.normal(size=2 * n)
        return x * beta
    else:
        return x


def seasonal_data(n_pts, seasonalities):
    if n_pts < 2 * max([x[1] for x in seasonalities]):
        print('ERROR: not enough data points')
        return None
    terms = {s[0]: fourier_series(n_pts, s).sum(axis=1) for s in seasonalities}
    df = pd.DataFrame(terms)
    df['total'] = df.sum(axis=1)
    return df['total'].values


def set_breaks(n_pts, tbreak_, theta, slow_bleed):
    # builds the time series of model parameters
    # n_pts: nbr of TS points
    # tbreak_: times of changepoints
    # theta: model parameter values after each change point
    # slow_bleed: (small) geom rate change of model parameter between change points
    n_cpt = len(tbreak_)
    thetat_ = np.zeros(n_pts)
    thetat_[:tbreak_[0]] = theta[0]
    print('index: ' + str(0) + ' start: ' + str(0) + ' end: ' + str(tbreak_[0]) + ' value: ' + str(theta[0]))
    for i in range(1, n_cpt):
        print('index: ' + str(i) + ' start: ' + str(tbreak_[i-1]) + ' end: ' + str(tbreak_[i]) + ' value: ' + str(theta[i]))
        thetat_[tbreak_[i-1]:tbreak_[i]] = theta[i] * np.array([(1 + slow_bleed) ** n for n in range(tbreak_[i] - tbreak_[i - 1])])
    thetat_[tbreak_[n_cpt - 1]:] = theta[n_cpt] * np.array([(1 + slow_bleed) ** n for n in range(n_pts - tbreak_[n_cpt - 1])])
    print('index: ' + str(n_cpt) + ' start: ' + str(tbreak_[n_cpt - 1]) + ' end: ' + str(n_pts) + ' value: ' + str(theta[n_cpt]))
    return thetat_
