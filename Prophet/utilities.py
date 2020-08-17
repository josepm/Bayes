"""

"""
import os
import sys
import numpy as np
import pandas as pd


def mase(f, shift_=1, y_col='y', yhat_col='yhat'):                  # len(f) < season
    # f is a DF with col names: y (actuals) and yhat (for the fancy forecast)
    # compare fcast error to the error of the naive forecaster yhat(t) = yhat(t|t-shift_) = y(t - shift_)
    # shift >=1
    fc = f.copy()
    if shift_ <= 0:
        print('ERROR: invalid shift: ' + str(shift_))
        return np.nan
    fc.dropna(inplace=True, subset=[y_col, yhat_col])
    if len(f) <= np.floor(shift_ / 2):
        print('WARNING: not enough data for mase: len(f) = ' + str(len(f)) + ' and shift: ' + str(shift_))
        return np.nan
    fs = fc.shift(-shift_)
    num = (fc[yhat_col] - fc[y_col]).abs().mean()        # MASE (for a window < season and factoring in upr and lwr)
    den = (fs[y_col] - f[y_col]).abs().mean()          # MASE (for a window < season and factoring in upr and lwr)
    if den == 0.0:
        return np.inf if num != 0 else np.nan
    else:
        return num / den          # MASE (for a window < season and factoring in upr and lwr)


def h_mase(f_, horizon, cutoff_date, freq='D', periods=None, t_col='ds', y_col='y', yhat_col='yhat'):
    # avg mase from 1 to horizon
    # assumes no data gaps
    # improvement over naive predictor if value < 1
    # if cu = cutoff_date,
    # n(h, s) = (1/#) sum_{t + h <= cu} |y_t - y_{t - h * s}|      # avg h-step ahead naive avg prediction error at period s (no period, s = 1)
    # n(h) = (1/#) sum_s n(h, s)                                   # avg h-step naive forecast error (avg naive errors across all seasonalities)
    # mase_H = (1/H) sum_{1 <= h <= H} |yhat_{cu + h} - y_{cu + h}| / n(h)  # cu = cutoff date
    f = f_.copy()
    ff = f[(f[t_col] > cutoff_date) & (f[t_col] <= cutoff_date + pd.to_timedelta(horizon, unit=freq))].copy()
    ferr = (ff[yhat_col] - ff[y_col]).abs()   # forecast error
    ferr.reset_index(inplace=True, drop=True)
    if periods is None:
        periods = [1]

    # naive forecast
    h_list = list()
    for h in range(1, horizon + 1):
        dlist = list()
        err = ferr.loc[h - 1]
        for s in periods:                      # naive forecast for period s and horizon h
            fn = f[f[t_col] <= cutoff_date + pd.to_timedelta(s, unit=freq)].copy()
            yns = fn[yhat_col].shift(-h * s)   # naive forecast at time t - h * s
            n = yns - fn[y_col]                # naive forecast error
            if n.isnull().sum() < len(n):
                den = n.abs().mean()
                dlist.append(den)
        if len(dlist) > 0:
            h_list.append(err / np.nanmean(np.array(dlist)))
    return np.nanmean(np.array(h_list))
        

def mape(f, y_col='y', yhat_col='yhat'):
    # f is a DF
    # computed on all the DF
    # assumes col names: y and yhat
    if len(f[(f[y_col] == 0.0) & (f[yhat_col] == 0.0)]) > 0:
        print('WARNNG: dropping zero/zero from mape computation')
        return mape(f[~((f[y_col] == 0.0) & (f[yhat_col] == 0.0))], y_col=y_col, yhat_col=yhat_col)
    elif len(f[(f[y_col] == 0.0) & (f[yhat_col] != 0.0)]) > 0:
        print('WARNNG: dropping zero from mape computation')
        return mape(f[~((f[y_col] == 0.0) & (f[yhat_col] != 0.0))], y_col=y_col, yhat_col=yhat_col)
    else:
        return 100.0 * (1 - f[yhat_col] / f[y_col]).abs().mean() if len(f) > 0 else np.nan                     # MAPE


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


class Linearizer(object):
    # transforms logistic like data between floor and ceiling (floor < y < ceiling, no equality) into unbound (linear)
    # See Fisher-Prys transform or
    # http://newmaeweb.ucsd.edu/courses/MAE119/WI_2018/ewExternalFiles/Simple%20Substitution%20Model%20of%20Technological%20Change%20-%20Fischer%20Pry%201971.pdf
    # Assume F < y < C
    # z = log((y - F) / (C - y)) is between -inf and inf
    # inverse:
    # y = (F + C * exp(z)) /(1 + exp(z))
    # y'' = (C - F) * exp(z) * (1 -exp(z)) / (1 + exp(z))^3
    def __init__(self, ceiling, floor, eps):
        self.lmbda = None
        self.ceiling = ceiling
        self.floor = floor
        self.eps = eps
        self.data_in = None
        self.fitted = False
        if isinstance(self.ceiling, type(None)) or isinstance(self.floor, type(None)):
            raise Exception('ERROR: must have ceiling and floor for logistic transform')
        if self.ceiling < self.floor:
            raise Exception('ERROR: must have ceiling and floor for logistic transform')

    def transform(self, y):
        if self.fitted is True:
            raise Exception('linearizer already fitted')
        if np.max(y) > self.ceiling:
            raise Exception('ceiling is too low')
        if np.min(y) < self.floor:
            raise Exception('floor is too high')
        self.data_in = y
        self.fitted = True
        c = self.ceiling + self.eps
        f = self.floor - self.eps
        return np.log((y - f) / (c - y))  # t

    def inverse_transform(self, y, y_var=None):
        if y is None:
            return None

        if y_var is None:
            return self._b_inverse_transform(np.array(y))
        else:
            fy, d2fy = self._u_inverse_transform(np.array(y))
            return None if fy is None else (fy if d2fy is None else list(np.array(fy) + (y_var / 2.0) * d2fy))

    def _u_inverse_transform(self, y):
        # unbiased inverse transform
        fy = self._b_inverse_transform(y)
        z = pd.DataFrame({'y': list(y)})
        d2fy = z['y'].apply(lambda x: 0 if np.isinf(np.exp(x)) else (self.ceiling - self.floor) * np.exp(x) * (1.0 - np.exp(x)) / np.power(1.0 + np.exp(x), 3.0))
        return fy, d2fy.values

    def _b_inverse_transform(self, y):
        # biased inverse transform
        z = pd.DataFrame({'y': list(y)})
        z['yf'] = z['y'].apply(lambda x: self.ceiling if np.isinf(np.exp(x)) else (self.floor + self.ceiling * np.exp(x)) / (1.0 + np.exp(x)))
        return list(z['yf'].values)


class ScaledData(object):
    # min-max scaler for a pandas series floats only

    def __init__(self, name, pd_series, s_dict, floor=None, ceiling=None):
        if isinstance(pd_series.min(), (int, float)) is False:
            raise Exception('scaling invalid series: ' + str(name))

        self.name = name
        self.npoints = len(pd_series)
        if self.name not in s_dict.keys():
            self.data_in = pd_series.values
            self.min_val = pd_series.min() if floor is None else floor
            self.max_val = pd_series.max() if ceiling is None else ceiling
            self.scale = np.abs(self.max_val - self.min_val)
            if self.scale == 0.0:
                self.scale = 1.0
            self.scaled_data = ((pd_series - self.min_val) / self.scale).values
            s_dict[self.name] = self
            print('name: ' + str(self.name) + ' min: ' + str(self.min_val) + ' scale: ' + str(self.scale))
        else:
            print('WARNING: ' + str(self.name) + ' has already been scaled')

    def scale_vals(self, data_in):
        # scale data_in
        return (pd.Series(data_in) - self.min_val) / self.scale

    def descale_vals(self, scaled_data=None):
        # convert to the original scale
        # scaled_data may contain new values
        return self.data_in - self.min_val if scaled_data is None else scaled_data * self.scale

    def reset_vals(self, scaled_data=None):
        # convert to the original scale and shift min value
        # scaled_data may contain new values
        return self.data_in if scaled_data is None else scaled_data * self.scale + self.min_val

