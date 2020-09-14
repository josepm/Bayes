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
from Utilities import ts_generate as ts_gen
from Bayes.Prophet import ts_preprocessing as ts_pre

##################################################################################
##################################################################################
##################################################################################
##################################################################################

# generate a TS with trend, seasonality and level changes (noise leve)
# assume seasonalities are known
# extract trend
# find change points in trend
seasonalities_ = [('monthly', 365.25/12, 5), ('weekly', 7, 3), ('yearly', 365.25, 2)]
periods = [s[1] for s in seasonalities_]
t = 1200
trend_cpt = 3
level_cpt = 0

# generate TS
ts_obj = ts_gen.TimeSeries(t, seasonalities=seasonalities_, event=None, t_size=-3, trend_cpt=trend_cpt, level_cpt=level_cpt,
                           trend_noise_level=0.0, trend_slow_bleed=0.0, level_noise_level=0.0, level_slow_bleed=0.0)
fy = pd.DataFrame({'y': ts_obj.ts})
fy.plot(grid=True, title='Time Series')
f_order = 20
stl_season, stl_trend, stl_level, y, f, season_cfg = ts_pre.mstl_decompose(ts_obj.ts, periods, f_order=f_order, do_plot=True)
# changepoints = ts_pre.tt_changepoint(f, window=25, alpha=0.01, a_cpts=ts_obj.trend_tbreak_, do_plot=True)
# print('original" ' + str(ts_obj.trend_tbreak_) + ' tt_detected: ' + str(changepoints))
changepoints = ts_pre.qr_changepoint(f, window=25, alpha=0.01, a_cpts=ts_obj.trend_tbreak_, eps=0.1, do_plot=True)
print('original: ' + str(ts_obj.trend_tbreak_) + ' rates: ' + str(ts_obj.trend_theta_[:-1]) + ' qr_detected: ' + str(changepoints))

