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

import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from Bayes.Prophet import ts_generate as ts_gen
from Bayes.Prophet import ts_preprocessing as ts_pre
from Bayes.Prophet import loess

##################################################################################
##################################################################################
##################################################################################
##################################################################################

# examples
# level change
t = 200
n_cpt = 5
y, tbreak, theta_t = ts_gen.gen_ts_(t, 'level', n_cpt=n_cpt, noise_level=0.0, slow_bleed=0.0)

t_step = 1
cpts = ts_pre.get_changepoints(y, t_step=t_step, t_last=10, type_='level', verbose=True)
print('level change actuals: ' + str(tbreak))
print('level changes detected: ' + str(cpts))
ts_pre.plot_change(y, np.array(range(t)), theta_t, cpts)

# trend change
t = 200
n_cpt = 5
y, tbreak, theta_t = ts_gen.gen_ts_(t, 'trend', n_cpt=n_cpt, noise_level=0.0, slow_bleed=0.0)

t_step = 1
cpts = ts_pre.get_changepoints(y, t_step=t_step, t_last=10, type_='trend', verbose=True)
print('trend change actuals: ' + str(tbreak))
print('trend changes detected: ' + str(cpts))
ts_pre.plot_change(y, np.array(range(t)), theta_t, cpts)

t_step = 1
cpts = ts_pre.get_changepoints(np.gradient(y), t_step=t_step, t_last=10, type_='level', verbose=True)
print('trend change actuals: ' + str(tbreak))
print('trend changes detected: ' + str(cpts))
ts_pre.plot_change(y, np.array(range(t)), theta_t, cpts)

# generate a TS with trend, seasonality and level changes (variance changes)
# additive vs multiplicative????
# assume seasonalities are known
# extract trend and level (noise)
# find change points in trend
seasonalities_ = [('monthly', 365.25/12, 3), ('weekly', 7, 2), ('yearly', 365.25, 5)]
periods = [s[1] for s in seasonalities_]
t = 1024
trend_cpt = 3
level_cpt = 0
mix = 0.0
if mix == 1.0:
    mode = 'a'
elif mix == 0.0:
    mode = 'm'
else:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR: not implemented !!!!!!!!!!!!!!!!!!!!!!!!!')
    mode = None

# generate TS
ts_obj = ts_gen.TimeSeries(t, seasonalities=seasonalities_, events=None, mix=mix, trend_cpt=trend_cpt, level_cpt=level_cpt,
                           trend_noise_level=0.0, trend_slow_bleed=0.0, level_noise_level=0.0, level_slow_bleed=0.0)
stl_season, stl_trend, stl_level = ts_pre.mstl_decompose(ts_obj.ts, periods, mode)

if mode.lower()[0] == 'a':
    season = ts_obj.a_seasons
    trend = ts_obj.a_trend
    level = ts_obj.a_level
else:
    season = ts_obj.m_seasons
    trend = ts_obj.m_trend
    level = ts_obj.m_level

ts_pre.plot_decompose(stl_season, stl_trend, stl_level, season, trend, level, ts_obj.ts, title='All')

# a_stl_season, a_stl_trend, a_stl_level = ts_pre._mstl_decompose(np.log(ts_obj.ts), periods, f_order=20)
# ts_pre.plot_decompose(None, None, None, season, trend, level, ts_obj.ts, title='All-Mult')
# ts_pre.plot_decompose(None, None, None, np.log(season), np.log(trend), np.log(level), np.log(ts_obj.ts), title='All-add')
# ts_pre.plot_decompose(stl_season, stl_trend, stl_level, np.log(season), np.log(trend), np.log(level), np.log(ts_obj.ts), title='All-Mult')

m_df = pd.DataFrame({'m_ts': ts_obj.ts, 'm_stl_trend': stl_trend, 'm_trend': ts_obj.m_trend})
m_df.plot(grid=True)
a_df = np.log(m_df)
a_df.rename(columns={'m_ts': 'a_ts', 'm_stl_trend': 'a_stl_trend', 'm_trend': 'a_trend'}, inplace=True)
a_df.plot(grid=True)

aic_arr, rss_arr, pars_arr, bic_arr = list(), list(), list(), list()
for f in np.linspace(0.1, 0.9, 9):
    m_stl_trend = sm.nonparametric.lowess(m_stl_trend_, range(len(m_stl_trend_)), frac=f, return_sorted=False)
    df = pd.DataFrame({'in': m_stl_trend_, 'out': m_stl_trend})
    rss = np.sum((m_stl_trend - m_stl_trend_) ** 2)
    nobs = len(m_stl_trend)
    pars = 2 * nobs / f   # capture total overfit with f = 0 (y_loess = y) and f = 1, linear approx
    aic = 2 * pars + nobs * np.log(rss)  # aic: 2, bic: log(n)
    bic = np.log(nobs) * pars + nobs * np.log(rss / nobs)  # aic: 2, bic: log(n)
    aic_arr.append(aic)
    bic_arr.append(bic)
    rss_arr.append(rss)
    pars_arr.append(pars)
    # df.plot(grid=True, title=str(f) + ':: err: ' + str(np.round(rss, 2)) + ' aic: ' + str(np.round(aic, 2)))
xf = pd.DataFrame({'f': np.linspace(0.1, 0.9, 9), 'aic': aic_arr, 'bic': bic_arr, 'rss': rss_arr, 'pars': pars_arr})
xf.set_index('f', inplace=True)
xf[['aic', 'bic']].plot(grid=True)
# xf[['rss']].plot(grid=True)

a_stl_season, a_stl_trend, a_stl_level = ts_pre._mstl_decompose(np.log(ts_obj.ts), periods, f_order=f_order)
m_stl_trend_ = np.exp(a_stl_trend)
x = np.array(range(len(m_stl_trend_)))
# loess.loess_bandwith(x, m_stl_trend_, iter_=3, fmin=0.1, fmax=0.9, npts=10)
yhat = loess.opt_loess(x, m_stl_trend_, iter_=3, fmin=0.1, fmax=0.9, npts=10)
xf = pd.DataFrame({'x': x, 'yhat': yhat, 'm_trend_': m_stl_trend_})
xf.set_index('x', inplace=True)
xf.plot(grid=True)

get_bandwith(np.array(range(len(m_stl_trend_))), m_stl_trend_)
m_stl_trend = sm.nonparametric.lowess(m_stl_trend_, range(len(m_stl_trend_)), frac=0.5, return_sorted=False)
if np.min(m_stl_trend) < 0.0:
    m_stl_trend += (1.0e-6 - np.min(m_stl_trend))
a_stl_season, a_stl_level = ts_pre._mstl_sl(np.log(ts_obj.ts), np.log(m_stl_trend), periods, f_order)

X_train = sf.values
results = sm.OLS(y_train, X_train, missing='drop').fit()
stl_season = results.predict(X_train)


# trend changes
if trend_cpt >= 0:
    print('\nStarting trend change points ....')
    t_step = 1
    trend_cpts = ts_pre.get_changepoints(stl_trend, t_step=t_step, t_last=10, type_='trend', verbose=True)
    print('trend change actuals: ' + str(trend_tbreak))
    print('trend changes detected: ' + str(trend_cpts))
    ts_pre.plot_change(np.array(range(t)), stl_trend, np.cumsum(trend_thetat), trend_cpts, trend_tbreak)

if level_cpt >= 0:
    print('\nStarting level change points ....')
    t_step = 1
    level_cpts = ts_pre.get_changepoints(stl_level, t_step=t_step, t_last=10, type_='level', verbose=True)
    print('level change actuals: ' + str(level_tbreak))
    print('level changes detected: ' + str(level_cpts))
    ts_pre.plot_change(np.array(range(t)), stl_level, level_thetat, level_cpts, level_tbreak)

