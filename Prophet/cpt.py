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

# generate a TS with trend, seasonality and level changes (noise leve)
# assume seasonalities are known
# extract trend and level (noise)
# find change points in trend
seasonalities_ = [('monthly', 365.25/12, 5), ('weekly', 7, 3), ('yearly', 365.25, 2)]
periods = [s[1] for s in seasonalities_]
t = 1200
trend_cpt = 3
level_cpt = 0

# generate TS
ts_obj = ts_gen.TimeSeries(t, seasonalities=seasonalities_, event=None, t_size=0, trend_cpt=trend_cpt, level_cpt=level_cpt,
                           trend_noise_level=0.0, trend_slow_bleed=0.0, level_noise_level=0.0, level_slow_bleed=0.0)
f_order = 20
stl_season, stl_trend, stl_level, y = ts_pre.mstl_decompose(ts_obj.ts, periods, f_order=f_order)
cpt_arr = ts_pre.qr_changepoint(stl_trend, window=10, alpha=0.05)
print(ts_obj.trend_tbreak_)

df = ts_pre.pw_changepoint(stl_trend)
df['cpt_in'] = [ts_obj.trend_tbreak_ for i in range(len(df))]
print('exact: ' + str(ts_obj.trend_tbreak_) + ' f0: ' + str(f0))  # + ' f7: ' + str(f7))



cpt_hat, lbda_arr = ts_pre.process_TS(ts_obj.ts, periods, f_order=20, t_step=1)
ts_pre.plot_change(ts_obj.trend, np.array(range(t)), None, ts_obj.trend_tbreak_, cpt_hat)


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
