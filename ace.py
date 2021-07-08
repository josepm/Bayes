"""
__author__: josep ferrandiz

Y dependent variable
X independent variables
Find theta() and phi_i() that maximize correlation between theta(Y) and sum_{i=1}^p phi_i(X_i) where theta(Y) = sum_{i=1}^p phi_i(X_i)

use from my_projects.ace import ace
"""

import os
import sys
import pandas as pd
import supersmoother as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

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


class ContSmooth(object):
    """
     continuous smoother
     :param x: given values
     :param y: y values
     :return: E[y|x=a]
     """
    def __init__(self, x, y):
        self.model = sm.SuperSmoother()
        self.model.fit(x, y, dy=np.std(y))

    def predict(self, xarr):
        return self.model.predict(xarr)


class ACE(object):
    def __init__(self, X, y, cat_X=None, cat_y=False, verbose=True):
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
        self.X_in = X                                                 # self.X shape: (nrows, ncols) (features in columns)
        self.x_scaler = StandardScaler()
        self.X = self.x_scaler.fit_transform(X)                       # self.X shape: (nrows, ncols) (features in columns)
        self.y_in = np.reshape(y, (-1, 1))                            # self.y shape: (nrows, 1)
        self.y = StandardScaler().fit_transform(self.y_in)            # self.y shape: (nrows, 1)
        self.scaler = StandardScaler()
        self.theta = self.y
        self.phi = np.zeros_like(self.X)                              # initialize the phi functions
        self.nrows, self.ncols = np.shape(self.phi)
        self.cat_y = cat_y                                            # True/False
        self.cat_X = list() if cat_X is None else cat_X               # col idx that are categorical
        self.phi_sum = None
        self.max_cnt = 1000
        self.ctr = 0
        self.round = 4
        self.max_same = 3
        self.same = 0
        self.verbose = verbose
        self.is_fit = False
        self.corr_coef = -1.0
        self.corr_list = list()

    def fit(self):
        """
        finds theta() and phi_i()
        """
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
            return CatSmooth(x, y) if self.cat_y is True else ContSmooth(x, y)
        else:
            return CatSmooth(x, y) if j in self.cat_X else ContSmooth(x, y)

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
            sm_objs = [self.obj_smooth(self.X[:, j], self.phi[:, j], j) for j in range(self.ncols)]     # list of smoother objects
            phi_sum = np.sum(np.transpose(np.array([sm_objs[j].predict(Xs[:, j]) for j in range(self.ncols)])), axis=1)
            s_obj = ContSmooth(self.phi_sum, self.y[:, 0])                                              # phi_sum is always continuous
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
        df['y'] = self.y[:, 0]
        yhat = self.predict(self.X_in)
        df['yhat'] = yhat
        gf = pd.DataFrame(self.phi)
        gf.columns = ['phi_' + str(c) for c in gf.columns]
        df = pd.concat([df, gf], axis=1)

        rmse = np.sqrt(np.mean((yhat - self.y[:, 0]) ** 2))
        df.plot(kind='scatter', grid=True, x='y', y='yhat', title='Prediction\nRMSE: ' + str(np.round(rmse, 4)))
        corr = np.corrcoef(df['phi_sum'].values, df['theta'].values)[0, 1]
        df[['phi_sum', 'theta']].plot(grid=True, style='x-', title='phi_sum and theta transforms\nCorr(theta, phi_sum): ' + str(np.round(corr, 4)))
        corr = np.corrcoef(df['y'].values, df['theta'].values)[0, 1]
        df.plot(kind='scatter', grid=True, x='y', y='theta', style='-x', title='theta transform\nCorr(y, theta): ' + str(np.round(corr, 4)))
        for j in range(self.ncols):
            corr = np.corrcoef(df['X_' + str(j)].values, df['y'].values)[0, 1]
            df.plot(kind='scatter', x='X_' + str(j), y='phi_' + str(j), grid=True, title='phi_' + str(j) + ' transform. \nCorr(X_' + str(j) + ', y): ' + str(np.round(corr, 4)))
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
            print('ctr: ' + str(self.ctr) + ' prev_corr: ' + str(self.corr_coef) + ' corr: ' + str(corr_coef) + ' eq: ' + str(corr_coef == self.corr_coef) + ' same: ' + str(self.same))
        if np.abs(corr_coef - self.corr_coef) == 0.0:
            self.same += 1
            return True if self.same == self.max_same else False
        else:
            self.corr_list.append(corr_coef)
            if len(self.corr_list) > 3 * self.max_same:
                self.corr_list.pop(0)
                if len(set(self.corr_list)) <= self.max_same:  # correlations are cycling (hopefully with close values)
                    print(list(self.corr_list))
                    return True

            # no cycles
            self.same = 0
            self.corr_coef = corr_coef
            return False


class AVAS(ACE):
    def __init__(self, X, y, cat_X=None, cat_y=False, verbose=True):
        """
        see ACE class
        """
        super().__init__(X, y, cat_X=cat_X, cat_y=cat_y, verbose=verbose)

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
        slr_hat = ContSmooth(s_phi_sum, s_lr).predict(s_lr) # phi_sum is always continuous
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
