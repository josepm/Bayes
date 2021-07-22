"""
__author__: josep ferrandiz

Y dependent variable
X independent variables
Find theta() and phi_i() that maximize correlation between theta(Y) and sum_{i=1}^p phi_i(X_i) where theta(Y) = sum_{i=1}^p phi_i(X_i)

set use to SuperSmoother or Lowess
if supersoother fails, switch to lowess

use from my_projects.ace import ace
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from my_projects.ace.supersmoother import ace_smoothers as asm
import time

N_JOBS = 4 if sys.platform == 'darwin' else os.cpu_count()


def example_data(N=100, scale=1.0, n=5):
    # some test data
    # N: points
    # scale: noise variance
    # n: number of independent components
    X = np.transpose(np.array([np.random.uniform(-1, 1, size=N) for _i in range(0, n)]))
    noise = np.random.normal(scale=scale, size=N)
    if n == 1:
        y = np.array(np.log(4.0 + np.sin(4 * X[:, 0])))
    elif n == 2:
        y = np.array(np.log(4.0 + np.sin(4 * X[:, 0]) + np.abs(X[:, 1])))
    elif n == 3:
        y = np.array(np.log(4.0 + np.sin(4 * X[:, 0]) + np.abs(X[:, 1]) + X[:, 2] ** 2))
    elif n == 4:
        y = np.array(np.log(4.0 + np.sin(4 * X[:, 0]) + np.abs(X[:, 1]) + X[:, 2] ** 2 + X[:, 3] ** 3 ))
    elif n == 5:
        y = np.array(np.log(4.0 + np.sin(4 * X[:, 0]) + np.abs(X[:, 1]) + X[:, 2] ** 2 + X[:, 3] ** 3 + X[:, 4]))
    else:
        print('at most 5 components')
        sys.exit(-1)
    return X, y + 0.1 * noise


class ACE(object):
    def __init__(self, X, y, smoother, cat_X=None, cat_y=False, verbose=True):
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
        self.phi_sum = np.array([0.0] * len(self.theta))             # sum_i phi_i(X_i)
        self.smoother = smoother                                     # smoother to use
        self.max_cnt = 1000                                          # max ACE/AVAS iterations
        self.ctr = 0                                                 # iteration counter
        self.round = 4                                               # convergence identical digits in correlation
        self.max_same = 3                                            # convergence repeats. Must be > 1
        self.same = 0                                                # convergence correlation repeat counter
        self.verbose = verbose
        self.is_fit = False
        self.var_dict = dict()                                      # cyclic convergence list
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
        self.is_fit = False
        retry = True
        print('starting with tol: 1.0e-0' + str(self.round))
        while self.is_fit is False:
            self.tic = time.time()
            self.X_loop()
            self.y_loop()
            # self.theta = StandardScaler().fit_transform(self.theta)  # shape: (nrows, 1)

            # check for NaNs and inf
            if self.check_data(self.phi, 'phi') is False:
                print('ERROR: invalid values for phi')
                retry = False
                break
            if self.check_data(self.theta, 'theta') is False:
                print('ERROR: invalid values for theta')
                retry = False
                break
            self.is_fit = self.check_convergence()
            if self.ctr >= self.max_cnt:
                print('too many iterations::could not converge with tol 1.0e-0' + str(self.round))
                break

        if self.is_fit is True:     # final smooths for prediction
            self.final_sm_objs = [self.obj_smooth(self.X[:, j], self.phi[:, j], j) for j in range(self.ncols)]
        else:
            self.fit_reset(retry)

    def fit_reset(self, retry):
        # try fit at lower tolerance
        if self.round > 2 and retry is True:
            print('WARNING: could not fit data with tol 1.0e-0' + str(self.round))
            self.round -= 1
            self.ctr = 0
            self.same = 0
            self.theta = self.y  # theta(Y)
            self.phi = np.zeros_like(self.X)
            self.phi_sum = np.array([0.0] * len(self.theta))
            self.var_dict = dict()
            self.fit()
        else:
            print('ERROR: could not fit data')
            sys.exit(-1)

    def X_loop(self):
        """
        phi_sum_j = sum_{k!=j} phi_k
        updates phi_j = E[theta - phi_sum_j|X_j]
        """
        _ = [self.process_col(j) for j in range(self.ncols)]  # go through cols
        self.phi_sum = np.sum(self.phi, axis=1)               # shape: (nrows, )
        self.phi_sum = (self.phi_sum - np.mean(self.phi_sum)) / np.std(self.phi_sum)

    def process_col(self, j):
        """
        update an X-column
        :param j: column index
        """
        phij = self.phi[:, j]
        zj = self.theta[:, 0] - (self.phi_sum - phij)
        xj = self.X[:, j]                      # shape: (nrows, )
        self.phi[:, j] = self.obj_smooth(xj, zj, j).predict(xj)
        self.phi[:, j] = (self.phi[:, j] - np.mean(self.phi[:, j])) / np.std(self.phi[:, j])
        self.phi_sum += (self.phi[:, j] - phij)

    def y_loop(self):
        """
        updates theta = E[phi_sum|Y]
        """
        s_obj = self.obj_smooth(self.y[:, 0], self.phi_sum, None)
        theta = s_obj.predict(self.y[:, 0])
        self.theta = np.reshape(theta, (-1, 1))
        self.theta = StandardScaler().fit_transform(self.theta)  # shape: (nrows, 1)

    def obj_smooth(self, x, y, j):
        """
        :param x: independent value
        :param y: dependent variable (E[y|x])
        :param j: column number or None
        :return: a smoother object consistent with x (categorical or continuous)
        """
        if j is None:
            return asm.CatSmooth(x, y) if self.cat_y is True else asm.ContSmooth(x, y, self.smoother)
        else:
            return asm.CatSmooth(x, y) if j in self.cat_X else asm.ContSmooth(x, y, self.smoother)

    def predict(self, X):
        """
        predict the y's from some X
        smoother is more accurate than interpolation
        :param X: input values to predict (must have same cols -features- as input X, any rows)
        :return: predicted values (y_hat, not theta(y))
        """
        if self.is_fit:
            if np.shape(X)[1] != self.ncols:
                print('invalid input to predict:: columns: ' + str(np.shape(X)[1]) + ' and should be ' + str(self.ncols))
                return None
            if self.check_data(X, 'predict data') is False:
                return None
            Xs = self.x_scaler.transform(X)
            phi_sum = np.sum(np.transpose(np.array([self.final_sm_objs[j].predict(Xs[:, j]) for j in range(self.ncols)])), axis=1)

            phi_sum = (phi_sum - np.mean(phi_sum)) / np.std(phi_sum)
            s_obj = asm.ContSmooth(phi_sum, self.y_in, self.smoother)                                              # phi_sum is always continuous
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
        df['phi_sum'] = (self.phi_sum - np.mean(self.phi_sum)) / np.std(self.phi_sum)   # scale like theta
        df['theta'] = self.theta[:, 0]
        df['y'] = self.y_in
        yhat = self.predict(self.X_in)
        df['yhat'] = yhat
        gf = pd.DataFrame(self.phi)
        gf.columns = ['phi_' + str(c) for c in gf.columns]
        df = pd.concat([df, gf], axis=1)

        corr = np.corrcoef(df['y'].values, df['yhat'].values)[0, 1]
        df.plot(kind='scatter', grid=True, x='y', y='yhat', title='Prediction: yhat vs. y\nCorr(y, yhat): ' + str(np.round(corr, 4)))
        corr = np.corrcoef(df['phi_sum'].values, df['yhat'].values)[0, 1]
        df.plot(kind='scatter', grid=True, x='phi_sum', y='yhat', title='Prediction from phi_sum\nCorr(phi_sum, yhat): ' + str(np.round(corr, 4)))
        corr = np.corrcoef(df['phi_sum'].values, df['theta'].values)[0, 1]
        df.plot(kind='scatter', grid=True, x='phi_sum', y='theta', title='Transformed Prediction\nCorr(phi_sum, theta): ' + str(np.round(corr, 4)))
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
        if np.isnan(corr_coef):
            print('ERROR: NaN correlation (variance = 0). Resetting fit')
            self.fit_reset(True)
            return self.is_fit
        return self.check_convergence_(corr_coef, discount=0.85)

    def check_convergence_(self, corr_coef, discount=0.99):
        # sometimes the algo cycles between a few (~close) variance values
        # we count how many times we see a var value and discount it
        # convergence if max discounted count > self.max_same
        # discount is used to manage cyclical (after round up) corr values
        # discount = 0.99 seems to work for periods <=10

        self.ctr += 1
        var = np.round(np.mean((self.theta[:, 0] - self.phi_sum) ** 2), self.round)
        for k in self.var_dict.keys():
            self.var_dict[k] **= discount              # discount old observed values
        if var not in self.var_dict.keys():
            self.var_dict[var] = 0
        self.var_dict[var] += 1.0
        dict_vals = np.array(list(self.var_dict.values()))
        args = np.argwhere(dict_vals > 2)               # recent repeating values must have a score > 2
        if self.verbose:
            print('>>>>> ctr: ' + str(self.ctr) +
                  ' curr_corr: ' + str(corr_coef) + ' var: ' + str(var) +
                  ' vars: ' + str({k: np.round(v, self.round) for k, v in self.var_dict.items() if v > 1}) +
                  ' time: ' + str(np.round(time.time() - self.tic, 4)))
        b = np.max(dict_vals[args]) > self.max_same if len(args) > 0 else False
        return bool(b)  # must have bool here


class AVAS(ACE):
    def __init__(self, X, y, smoother, cat_X=None, cat_y=False, verbose=True):
        """
        see ACE class
        """
        super().__init__(X, y, smoother, cat_X=cat_X, cat_y=cat_y, verbose=verbose)

    def y_loop(self):
        super().y_loop()

        # variance stabilization
        args, bargs = self.var_sort()                       # sort args and back-sort by y
        s_theta = self.theta[args, 0]                               # sorted by increasing y
        s_phi_sum = self.phi_sum[args]                      # phi_sum sorted by increasing y

        # res = self.theta[:, 0] - self.phi_sum             # does not work
        res = s_theta - np.mean(s_theta)
        res2 = res[args] ** 2                               # var(theta) = np.mean(res2)

        # take log and exp later to avoid negative smooths
        vu = asm.ContSmooth(s_phi_sum, np.log(res2), self.smoother).predict(s_phi_sum)   # phi_sum is always continuous. log(resr2) = fsmooth(phi_sum)

        # remove the max to avoid overflows: vu = vu_max - vu_tilde, vu_tilde >= 0 and exp(vu) = K * exp(-vu_tilde). We only integrate vu_tilde
        # 0 < vu_tilde < vu_max-vu_min => vu_min - vu_max < -vu_tilde < 0: exp will not blow up
        vu_max = np.max(vu)
        vu_tilde = vu_max - vu
        vu_ = np.exp(-vu_tilde)     # vu_tilde = var(theta|phi_sum) up to the vm constant which goes away at StandardScaler()

        # almost the integral in paper (up to 1/sqrt(vu_max))
        htheta = np.array([np.trapz(1.0 / np.sqrt(vu_[:ix + 1]), s_theta[:ix + 1]) for ix in range(self.nrows)])
        self.theta = np.reshape(htheta[bargs], (-1, 1))          # restore the original order and reshape
        self.theta = StandardScaler().fit_transform(self.theta)  # shape: (nrows, 1)

    def var_sort(self):
        args = np.argsort(self.theta[:, 0])
        bargs = np.zeros_like(args)
        for i, si in enumerate(args):    # to restore original order
            bargs[si] = i
        return args, bargs

