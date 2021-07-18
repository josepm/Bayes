"""
__author__: josep ferrandiz

"""

import os
import sys
import pandas as pd
# import my_projects.ace.supersmoother as sm
import supersmoother as sm
import numpy as np
from scipy.interpolate import interp1d
import statsmodels.api as sm_api
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar, fminbound
import time

lowess = sm_api.nonparametric.lowess

N_JOBS = 4 if sys.platform == 'darwin' else os.cpu_count()


class CatSmooth(object):
    """
    categorical smoother
    :param x: given categorical values
    :param y: y values
    :return: E[y|x=a]
    """
    def __init__(self, x, y):
        self.vals = np.unique(x)
        self.brr = dict()
        for val in self.vals:
            args = np.argwhere(x == val)[0]
            den = len(args)
            num = np.sum(y[args])
            self.brr[self.vals] = num / den

    def predict(self, xarr):
        return np.array([self.brr.get(v, 0.0) for v in xarr])


class ContSmooth(object):
    """
     continuous smoother
     :param x: given values
     :param y: y values
     :param name: 'SuperSmoother' or 'Lowess'
     :return: E[y|x=a]
     """

    def __init__(self, x, y, name='SuperSmoother'):
        tic = time.time()
        if name == 'SuperSmoother':
            min_J = 5               # min points in a window (otherwise no convergence). Does no work with less than 5!
            s12 = 4                 # multiplier from min to mid span, ie mid_span / min_span
            s23 = 2.5               # multiplier from mid to max span, ie max_span / mid_span

            # min data points: we must have min_J/len * s12 * s23 <= 1
            min_len = min_J * s12 * s23
            if len(y) < min_len:
                print('ERROR: not enough data points: ' + str(len(y)) + '. Should be at least ' + str(min_len) + '. Try Lowess')
                sys.exit(-1)
            min_span = max(0.05, min_J / len(y))
            self.model = sm.SuperSmoother(primary_spans=(min_span, min_span * s12, min_span * s12 * s23), middle_span=min_span * s12, final_span=min_span)
            try:
                self.model.fit(x, y, dy=np.std(y))
            except ValueError as e:
                print('ERROR: ' + str(e) + '\nTry Lowess smoother')
                sys.exit(0)
        elif name == 'Lowess':
            self.model = LowessSmooth(x, y)
            self.model.fit(x, y, dy=1.0)   # get BW
        else:
            print('ERROR: invalid smoother: ' + str(name) + '. Valid smoothers are: SuperSmoother, Lowess')
            sys.exit(-1)
        # print('ContSmooth fit time: ' + str(np.round(time.time()-tic, 4)) + 'secs')

    def predict(self, xarr):
        return self.model.predict(xarr)


class LowessSmooth(object):
    def __init__(self, x, y):
        self.args = np.argsort(x)
        self.xs = x[self.args]
        self.ys = y[self.args]
        self.delta = 0.01 * (self.xs[-1] - self.xs[0])
        dxs = np.diff(self.xs)
        dargs = np.argwhere(dxs > self.delta)  # indices of the "lowess points"

        # loo cv only on the "lowess" points
        self.loo = list()
        for xtest in dargs:
            xtrain = np.delete(dargs, np.where(dargs == xtest))
            self.loo.append((xtrain, xtest[0]))
        self.it = 3
        self.method = 'fminbound'                    # 'loop', bounded, fminbound
        self.min_frac = max(0.05, 5 / len(x))

    def fit(self, x, y, dy=1.0):
        self.bw, min_mse, niter = self.get_bw(self.method)

        # prepare predict for Lowess
        yhat = lowess(self.ys, self.xs, frac=self.bw, is_sorted=True, delta=self.delta, it=self.it)[:, 1]
        self.fpred = interp1d(self.xs, yhat, fill_value='extrapolate', kind='linear')

    def lowess_cv(self, frac):
        # sqr_errs = [self.loocv_idx(train_index, test_index, frac) for train_index, test_index in self.loo]
        sqr_errs = Parallel(n_jobs=N_JOBS)(delayed(self.loocv_idx)(train_index, test_index, frac) for train_index, test_index in self.loo)
        mse = np.mean(np.array(sqr_errs))
        return mse

    def get_bw(self, method):
        if method == 'fminbound':
            return self._fminbound_bw()
        elif method == 'loop':
            return self._loop_bw()
        elif method == 'bounded':
            return self._bounded_bw()
        else:
            print('ERROR: invalid method: ' + str(method) + '. Defaulting to loop')
            return self._loop_bw()

    def _loop_bw(self):
        # exit after max_quit_ctr values larger than min_mse
        frac_list_ = np.arange(0.05, 1.0, 0.05)
        self.frac_list = [f for f in frac_list_ if f >= self.min_frac]
        min_mse, bw, ix = np.inf, 0.5, 0
        quit_ctr, max_quit_ctr = 0, 2
        for ix, frac in enumerate(self.frac_list):
            mse = self.lowess_cv(frac)
            if mse < min_mse:
                quit_ctr = 0  # reset quit_ctr
                min_mse = mse
                bw = frac
            else:
                quit_ctr += 1
                if quit_ctr == max_quit_ctr:  # break after we see max_quit_ctr values with mse larger than min_mse
                    break
        return bw, min_mse, 2 + ix if quit_ctr == max_quit_ctr else self._bounded_bw()  # use bounded if min_mse not found in the grid

    def _bounded_bw(self):
        res = minimize_scalar(self.lowess_cv, bracket=None, bounds=(self.min_frac, 1.0), method='bounded', options={'xatol': 1.0e-03, 'maxiter': 500})
        return res.x, res.fun, res.nfev

    def _fminbound_bw(self):
        # tic=time.time()
        fmin = fminbound(self.lowess_cv, self.min_frac, 1.0, xtol=1.0e-03, maxfun=500, full_output=True)
        # print('fmin::bw: ' + str(np.round(fmin[0], 2)) + ' min_mse: ' + str(np.round(fmin[1], 4)) + ' time: ' + str(np.round(time.time()-tic, 2)) + 'secs')
        return fmin[0], fmin[1], fmin[3]   # bw, min_mse, niter

    def loocv_idx(self, train_index, test_index, frac):
        xs_train, xs_test = self.xs[train_index], self.xs[test_index]
        ys_train, ys_test = self.ys[train_index], self.ys[test_index]
        yhat_i = lowess(ys_train, xs_train, frac=frac, is_sorted=True, delta=self.delta, it=self.it)[:, 1]
        f = interp1d(xs_train, yhat_i, fill_value='extrapolate', kind='linear')
        return (ys_test - f(xs_test)) ** 2

    def predict(self, xval):
        return self.fpred(xval)

