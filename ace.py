"""
__author__: josep ferrandiz

Y dependent variable
X independent variables
Find theta() and phi_i() that maximize correlation between theta(Y) and sum_{i=1}^p phi_i(X_i) where theta(Y) = sum_{i=1}^p phi_i(X_i)

set use to SuperSmoother or Lowess
if supersoother fails, switch to lowess

set method to 'loop' or 'bounded'
lowess with method = 'loop' converges faster than with method 'bounded'

use from my_projects.ace import ace
"""

import os
import sys
import pandas as pd
import supersmoother as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import statsmodels.api as sm_api
from sklearn.model_selection import LeaveOneOut
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
import time

lowess = sm_api.nonparametric.lowess

N_JOBS = 4 if sys.platform == 'darwin' else os.cpu_count()


def example_data(N=100, scale=1.0):
    X = np.transpose(np.array([np.random.uniform(-1, 1, size=N) for _i in range(0, 5)]))
    noise = np.random.normal(scale=scale, size=N)
    y = np.array(np.log(4.0 + np.sin(4 * X[:, 0]) + np.abs(X[:, 1]) + X[:, 2] ** 2 + X[:, 3] ** 3 + X[:, 4] + 0.1 * noise))
    return X, y


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


SMOOTHER_SWITCH = False


class ContSmooth(object):
    """
     continuous smoother
     :param x: given values
     :param y: y values
     :param use: 'SuperSmoother' or 'Lowess'
     :param method: (Lowess only) 'loop' or 'bounded'
     :return: E[y|x=a]
     """

    def __init__(self, x, y, use='SuperSmoother', method='loop'):
        if use == 'SuperSmoother' and SMOOTHER_SWITCH is False:
            self.model = sm.SuperSmoother()
            try:
                self.model.fit(x, y, dy=np.std(y))
            except ValueError as e:
                print('ERROR: ' + str(e) + '\nTry Lowess smoother')
                sys.exit(0)
        elif use == 'Lowess':
            self.model = LowessSmooth(x, y, method)
            self.model.fit()   # get BW
        else:
            print('ERROR: invalid smoother: ' + str(use) + '. Valid smoothers are: SuperSmoother, Lowess')
            sys.exit(-1)

    def predict(self, xarr):
        return self.model.predict(xarr)


class LowessSmooth(object):
    def __init__(self, x, y, method):
        # method: bounded or loop
        frac_list_ = [0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        self.min_frac = 3 / len(x)  # at least 3 points
        self.frac_list = [f for f in frac_list_ if f >= self.min_frac]
        self.args = np.argsort(x)
        self.xs = x[self.args]
        self.ys = y[self.args]
        self.delta = 0.01 * (self.xs[-1] - self.xs[0])
        self.it = 3
        self.method = 'loop' if method is None else method

    def fit(self):
        tic = time.time()
        self.bw, min_mse, niter = self.get_bw(self.method)
        print('lowess fit:: method: ' + str(self.method) +
              ' time: ' + str(np.round(time.time() - tic, 2)) +
              'secs frac: ' + str(np.round(self.bw, 2)) + ' min_mse: ' + str(np.round(min_mse, 4)) + ' niter: ' + str(niter))

        # prepare predict for Lowess
        yhat = lowess(self.ys, self.xs, frac=self.bw, is_sorted=True, delta=self.delta, it=self.it)[:, 1]
        self.fpred = interp1d(self.xs, yhat, fill_value='extrapolate', kind='linear')

    def lowess_cv(self, frac):
        loo = LeaveOneOut()
        # sqr_errs = [self.loocv_idx(train_index, test_index, frac) for train_index, test_index in loo.split(self.xs)]
        sqr_errs = Parallel(n_jobs=N_JOBS)(delayed(self.loocv_idx)(train_index, test_index, frac) for train_index, test_index in loo.split(self.xs))
        mse = np.mean(np.array(sqr_errs))
        return mse

    def get_bw(self, method):
        if method == 'loop':
            return self._loop_bw()
        elif method == 'bounded':
            return self._bounded_bw()
        else:
            print('ERROR: invalid method: ' + str(method) + '. Defaulting to loop')
            return self._loop_bw()

    def _loop_bw(self):
        # exit after max_quit_ctr values larger than min_mse
        min_mse, bw, ix = np.inf, 0.5, 0
        quit_ctr, max_quit_ctr = 0, 2
        for ix, frac in enumerate(self.frac_list):
            mse = self.lowess_cv(frac)
            # print('>>>>>>>>>>>>>>>>>>> ix: ' + str(ix) + ' frac: ' + str(frac) +
            #       ' mse: ' + str(np.round(mse, 4)) + ' min_mse: ' + str(np.round(min_mse, 4)) + ' diff: ' + str(np.round(mse - min_mse, 4)) +
            #       ' quit_ctr: ' + str(quit_ctr))
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
        res = minimize_scalar(self.lowess_cv, bracket=None, bounds=(self.min_frac, 0.99), method='bounded', options={'xatol': 1.0e-03, 'maxiter': 500})
        return res.x, res.fun, res.nfev

    def loocv_idx(self, train_index, test_index, frac):
        xs_train, xs_test = self.xs[train_index], self.xs[test_index]
        ys_train, ys_test = self.ys[train_index], self.ys[test_index]
        yhat_i = lowess(ys_train, xs_train, frac=frac, is_sorted=True, delta=self.delta, it=self.it)[:, 1]
        f = interp1d(xs_train, yhat_i, fill_value='extrapolate', kind='linear')
        return (ys_test - f(xs_test)) ** 2

    def predict(self, xval):
        return self.fpred(xval)


class ACE(object):
    def __init__(self, X, y, use, method, cat_X=None, cat_y=False, verbose=True):
        """
        ACE class
        non-categorical: no repeat values (may need to add noise)
        usage:
        obj = ACE(X, y)
        obj.fit()
        y_pred = obj.predict(y')
        df = obj.plot() : plots + data in DF
        :param X: (n, p) np.array of independent variables (p features)
        :param y: (n, ) np.array, dependent variable
        :param cat_X: list of categorical features (column indices in X)
        :param cat_y: True if y is categorical else continuous
        :param method: None, loop or bounded. None defaults to loop
        :param verbose: True to print
        """
        if self.check_data(X, 'X') is False:
            sys.exit(-1)
        if self.check_data(y, 'y') is False:
            sys.exit(-1)
        self.X_in = X                                                # self.X shape: (nrows, ncols) (features in columns)
        self.x_scaler = StandardScaler()                             # scaler for X
        self.X = self.x_scaler.fit_transform(X)                      # self.X shape: (nrows, ncols) (features in columns)
        self.y_in = y                                                # self.y shape: (nrows, 1)
        yy = np.reshape(y, (-1, 1))
        self.y = StandardScaler().fit_transform(yy)                  # self.y shape: (nrows, 1)
        self.theta = self.y                                          # theta(Y)
        self.phi = np.zeros_like(self.X)                             # initialize the phi_i functions
        self.nrows, self.ncols = np.shape(self.phi)
        self.cat_y = cat_y                                           # True/False
        self.cat_X = list() if cat_X is None else cat_X              # col idx that are categorical
        self.phi_sum = None                                          # sum_i phi_i(X_i)
        self.use = use                                               # smoother to use
        self.method = method                                         # BW method for Lowess
        self.max_cnt = 1000                                          # max ACE/AVAS iterations
        self.ctr = 0                                                 # iteration counter
        self.round = 4                                               # convergence identical digits in correlation
        self.max_same = 3                                            # convergence repeats
        self.same = 0                                                # convergence correlation repeat counter
        self.verbose = verbose
        self.is_fit = False
        self.corr_coef = -1.0
        self.corr_list = list()                                      # cyclic convergence list
        self.max_corr = None
        print('Initial Correlations')
        for j in range(self.ncols):
            corr = np.corrcoef(self.X[:, j], self.y[:, 0])[0, 1]
            print('\tcorr(y, X_' + str(j) + '): ' + str(np.round(corr, 4)))

    def fit(self):
        """
        finds theta() and phi_i()
        """
        self.max_corr = None
        while self.is_fit is False:
            self.X_loop()
            self.y_loop()

            # check for NaNs and inf
            if self.check_data(self.phi, 'phi') is False:
                break
            if self.check_data(self.theta, 'theta') is False:
                break
            self.is_fit = self.check_convergence()
            if self.ctr >= self.max_cnt:
                print('could not converge')
                break

        if self.is_fit is True:     # final smooths for prediction
            self.final_sm_objs = [self.obj_smooth(self.X[:, j], self.phi[:, j], j) for j in range(self.ncols)]

    def X_loop(self):
        """
        phi_sum_j = sum_{k!=j} phi_k
        updates phi_j = E[theta - phi_sum_j|X_j]
        """
        _ = [self.process_col(j) for j in range(self.ncols)]  # go through cols
        self.phi_sum = np.sum(self.phi, axis=1)               # shape: (nrows, )

    def y_loop(self):
        """
        updates theta = E[phi_sum|Y]
        """
        s_obj = self.obj_smooth(self.y[:, 0], self.phi_sum, None)
        theta = s_obj.predict(self.y[:, 0])
        theta = np.reshape(theta, (-1, 1))
        self.theta = StandardScaler().fit_transform(theta)  # shape: (nrows, 1)

    def obj_smooth(self, x, y, j):
        """
        :param x: independent value
        :param y: dependent variable (E[y|x])
        :param j: column number or None
        :return: a smoother object consistent with x (categorical or continuous)
        """
        if j is None:
            return CatSmooth(x, y) if self.cat_y is True else ContSmooth(x, y, use=self.use, method=self.method)
        else:
            return CatSmooth(x, y) if j in self.cat_X else ContSmooth(x, y, use=self.use, method=self.method)

    def process_col(self, j):
        """
        update an X-column
        :param j: column index
        """
        phij = np.delete(self.phi, j, axis=1)  # drop column j
        xj = self.X[:, j]                      # shape: (nrows, )
        zj = self.theta[:, 0] - np.sum(phij, axis=1)  # shape: (nrows, 1)
        s_obj = self.obj_smooth(xj, zj, j)
        self.phi[:, j] = s_obj.predict(xj)

    def predict(self, X):
        """
        predict the y's from some X
        smoother is more accurate than interpolation
        :param X: input values to predict (must have same cols -features- as input X, any rows)
        :return: predicted values (y_hat)
        """
        if self.is_fit:
            if np.shape(X)[1] != self.ncols:
                print('invalid input to predict:: columns: ' + str(np.shape(X)[1]) + ' and should be ' + str(self.ncols))
                return None
            if self.check_data(X, 'predict data') is False:
                return None
            Xs = self.x_scaler.transform(X)
            phi_sum = np.sum(np.transpose(np.array([self.final_sm_objs[j].predict(Xs[:, j]) for j in range(self.ncols)])), axis=1)
            s_obj = ContSmooth(self.phi_sum, self.y_in, use=self.use, method=self.method)                                              # phi_sum is always continuous
            ysm = s_obj.predict(phi_sum)
            return ysm
        else:
            print('must fit model first')
            return None

    def plots(self):
        if self.is_fit is False:
            print('must fit data first')
            return None
        df = pd.DataFrame(self.X_in)
        df.columns = ['X_' + str(c) for c in df.columns]
        df['phi_sum'] = self.phi_sum
        df['theta'] = self.theta[:, 0]
        df['y'] = self.y_in
        yhat = self.predict(self.X_in)
        df['yhat'] = yhat
        gf = pd.DataFrame(self.phi)
        gf.columns = ['phi_' + str(c) for c in gf.columns]
        df = pd.concat([df, gf], axis=1)

        corr = np.corrcoef(df['phi_sum'].values, df['theta'].values)[0, 1]
        df.plot(kind='scatter', grid=True, x='phi_sum', y='theta', title='Transformed Prediction\nRMSE: ' + str(np.round(corr, 4)))
        df[['phi_sum', 'theta']].plot(grid=True, style='x-', title='phi_sum and theta transforms\nCorr(theta, phi_sum): ' + str(np.round(corr, 4)))
        corr = np.corrcoef(df['y'].values, df['theta'].values)[0, 1]
        df.plot(kind='scatter', grid=True, x='y', y='theta', style='-x', title='theta transform\nCorr(y, theta): ' + str(np.round(corr, 4)))
        for j in range(self.ncols):
            corr = np.corrcoef(df['X_' + str(j)].values, df['phi_' + str(j)].values)[0, 1]
            df.plot(kind='scatter', x='X_' + str(j), y='phi_' + str(j), grid=True, title='phi_' + str(j) + ' transform. \nCorr(X_' + str(j) + ', phi_' + str(j) + ': ' + str(np.round(corr, 4)))
        return df

    @staticmethod
    def check_data(arr, lbl):
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print('invalid values in ' + lbl)
            return False
        else:
            return True

    def check_convergence(self):
        corr_coef = np.round(np.corrcoef(self.theta[:, 0], self.phi_sum)[0, 1], self.round)
        self.ctr += 1
        if self.verbose:
            print('>>>>> ctr: ' + str(self.ctr) + ' prev_corr: ' + str(self.corr_coef) +
                  ' corr: ' + str(corr_coef) + ' eq: ' + str(corr_coef == self.corr_coef) + ' same: ' + str(self.same))

        # correlation cycle??
        self.corr_list.append(corr_coef)
        ucorr = len(set(self.corr_list))
        if ucorr < len(self.corr_list) / 4:  # correlations are cycling (hopefully with close values)
            self.max_corr = np.max(np.array(self.corr_list))
            print('Cyclic correlations::: ' + str(self.ctr) + ' iterations with ' + str(ucorr) + ' unique correlation values and max_corr: ' + str(self.max_corr))
            return True if self.max_corr == corr_coef else False

        if np.abs(corr_coef - self.corr_coef) == 0.0:
            self.same += 1
            return True if self.same == self.max_same else False
        else:
            self.same = 0
            self.corr_coef = corr_coef
            return False


class AVAS(ACE):
    def __init__(self, X, y, use, method, cat_X=None, cat_y=False, verbose=True):
        """
        see ACE class
        """
        super().__init__(X, y, use, method, cat_X=cat_X, cat_y=cat_y, verbose=verbose)

    def y_loop(self):
        super().y_loop()

        # variance stabilization
        # Var(theta|phi_sum) ~= E[(theta - phi_sum)^2|phi_sum] because E[theta|phi_sum] ~= phi_sum
        # take log and exp later to avoid negative smooths
        res = self.theta[:, 0] - self.phi_sum
        res = np.where(np.abs(res) < 1.0e-06, 1.0e-06, res)  # avoid log(0)
        lr = np.log(np.sqrt(res ** 2))
        args, bargs = self.var_sort()                       # sort args and back-sort by y
        s_y = self.y[args, 0]                               # sorted y
        s_phi_sum = self.phi_sum[args]                      # y sorted phi_sum
        s_lr = lr[args]                                     # y sorted log|residuals|
        slr_hat = ContSmooth(s_phi_sum, s_lr, use=self.use, method=self.method).predict(s_lr)   # phi_sum is always continuous
        vu = np.exp(slr_hat)[args]                          # y sorted V(u) = Var(theta|phi_sum=u)
        vu = np.where(vu == 0.0, 1.0e-12, vu)               # avoid 1/0
        htheta = np.array([np.trapz(1.0 / np.sqrt(vu[:ix + 1]), s_y[:ix + 1]) for ix in range(self.nrows)])
        theta = np.reshape(htheta[bargs], (-1, 1))          # restore the original order
        self.theta = StandardScaler().fit_transform(theta)  # shape: (nrows, 1)

    def var_sort(self):
        args = np.argsort(self.y[:, 0])
        bargs = np.zeros_like(args)
        for i, si in enumerate(args):    # to restore original order
            bargs[si] = i
        return args, bargs
