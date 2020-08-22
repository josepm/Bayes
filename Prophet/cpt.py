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
# assume
seasonalities_ = [('monthly', 365.25/12, 3), ('weekly', 7, 2)]
seasonalities_ = [('weekly', 7, 1)]
periods = [s[1] for s in seasonalities_]
t = 256
additive = True
ts, season, trend, level, trend_tbreak, level_tbreak, trend_thetat, level_thetat = ts_gen.gen_ts(t, seasonalities=seasonalities_,
                                                                                                 additive=additive, trend_cpt=3, level_cpt=2,
                                                                                                 noise_level=0.0, slow_bleed=0.0)


stl_season, stl_trend, stl_level = ts_pre.mstl_decompose(ts, periods=[365.25/12], additive=additive)
ts_pre.plot_decompose(stl_season, stl_trend, stl_level, season, trend, level, title='All')

# trend changes
print('\nStarting trend change points ....')
t_step = 1
trend_cpts = ts_pre.get_changepoints(stl_trend, t_step=t_step, t_last=10, type_='trend', verbose=True)
print('trend change actuals: ' + str(trend_tbreak))
print('trend changes detected: ' + str(trend_cpts))
ts_pre.plot_change(np.array(range(t)), stl_trend, np.cumsum(trend_thetat), trend_cpts, trend_tbreak)

print('\nStarting level change points ....')
t_step = 1
level_cpts = ts_pre.get_changepoints(stl_level, t_step=t_step, t_last=10, type_='level', verbose=True)
print('level change actuals: ' + str(level_tbreak))
print('level changes detected: ' + str(level_cpts))
ts_pre.plot_change(np.array(range(t)), stl_level, level_thetat, level_cpts, level_tbreak)

# #######################################
# finding the periods automatically
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm

ss = stl_season
ts0 = ss - ss.mean()
f_s = 1
X = fftpack.fft(ts0)
freqs = fftpack.fftfreq(len(ts0)) * f_s
fig, ax = plt.subplots()
xabs = np.sqrt(np.abs(X))

ax.stem(freqs, xabs)
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(0, f_s / 2)

from scipy import signal
f, Pxx = signal.welch(ts0, f_s, nperseg=len(ts0), detrend='linear', )
fig, ax = plt.subplots()
xabs = np.sqrt(Pxx)
ax.stem(f, xabs)
# plt.plot(f, Pxx)
# plt.semilogy(f, Pxx)
# plt.ylim([0.5e-3, 1])
ax.set_xlabel('frequency')
ax.set_ylabel('PSD')
ax.grid(True)
plt.show()

sm.graphics.tsa.plot_acf(ts0, fft=True, alpha=0.05, use_vlines=True )
plt.grid(True)
plt.show()
sm.graphics.tsa.plot_pacf(ts0, alpha=0.05, use_vlines=True )
plt.grid(True)
plt.show()
