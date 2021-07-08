"""
__author__: josep ferrandiz

Y dependent variable
X independent variables
Find theta() and phi_i() that maximize correlation between theta(Y) and sum_{i=1}^p phi_i(X_i) where theta(Y) = sum_{i=1}^p phi_i(X_i)

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
        self.max_cnt = 100
        self.ctr = 0
        self.round = 2
        self.max_same = 3
        self.same = 0
        self.verbose = verbose
        self.is_fit = False
        self.corr_coef = -1.0

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
        updates phi_i()
        """
        _ = [self.process_col(j) for j in range(self.ncols)]  # go through cols
        self.phi_sum = np.sum(self.phi, axis=1)               # shape: (nrows, )

    def y_loop(self):
        """
        updates theta()
        """
        theta = self.cat_smooth(self.y[:, 0], self.phi_sum) if self.cat_y is True else self.cont_smooth(self.y[:, 0], self.phi_sum)
        theta = np.reshape(theta, (-1, 1))
        if self.__class__.__name__ == 'ACE':
            self.theta = StandardScaler().fit_transform(theta)  # shape: (nrows, 1)
        else:
            self.theta = theta

    @staticmethod
    def cat_smooth(x, y):
        """
        categorical smoother
        :param x: given categorical values
        :param y: y values
        :return: E[y|x=a]
        """
        brr = np.zeros_like(y)
        for val in np.unique(x):
            args = np.argwhere(x == val)[0]
            den = len(args)
            num = np.sum(y[args])
            brr[args] = num / den
        return brr

    @staticmethod
    def cont_smooth(x, y):
        """
        continuous smoother
        :param x: given values
        :param y: y values
        :return: E[y|x=a]
        """
        model = sm.SuperSmoother()
        model.fit(x, y, dy=np.std(y))
        xhat = model.predict(x)
        return xhat

    def process_col(self, j):
        """
        update an X-column
        :param j: column index
        """
        phij = np.delete(self.phi, j, axis=1)  # drop column j
        xj = self.X[:, j]                      # shape: (nrows, )
        zj = self.theta[:, 0] - np.sum(phij, axis=1)  # shape: (nrows, 1)
        self.phi[:, j] = self.cat_smooth(xj, zj) if j in self.cat_X else self.cont_smooth(xj, zj)

    def predict(self, X):
        """
        predict the y's from some X
        :param X: input values to predict (must have same cols as input X, any rows)
        :return: predicted values (y_hat)
        """
        if self.is_fit:
            if np.shape(X)[1] != self.ncols:
                print('invalid input to predict:: columns: ' + str(np.shape(X)[1]) + ' and should be ' + str(self.ncols))
                return None
            if self.check_data(X, 'predict data') is False:
                return None
            fint = [interp1d(self.X[:, j], self.phi[:, j], fill_value='extrapolate', kind='linear') for j in range(self.ncols)]
            Xs = self.x_scaler.transform(X)
            phi_sum = np.sum(np.transpose(np.array([fint[j](Xs[:, j]) for j in range(self.ncols)])), axis=1)
            fy = interp1d(self.phi_sum, self.y[:, 0], fill_value='extrapolate', kind='linear')
            return fy(phi_sum)
        else:
            print('must fit model first')
            return None

    def plot(self):
        if self.is_fit is False:
            print('must fit data first')
            return None
        df = pd.DataFrame(self.X_in)
        df.columns = ['X_' + str(c) for c in df.columns]
        df['phi_sum'] = self.phi_sum
        df['theta'] = self.theta[:, 0]
        df['y'] = self.y_in[:, 0]
        gf = pd.DataFrame(self.phi)
        gf.columns = ['phi_' + str(c) for c in gf.columns]
        df = pd.concat([df, gf], axis=1)

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
            print('ctr: ' + str(self.ctr) + ' corr: ' + str(corr_coef) + ' same: ' + str(self.same))
        if np.abs(corr_coef - self.corr_coef) == 0.0:
            self.same += 1
            return True if self.same == self.max_same else False
        else:
            self.same = 0
            self.corr_coef = corr_coef
            return False

    @staticmethod
    def example_data(N=100, scale=1.0):
        X = np.transpose(np.array([np.random.uniform(-1, 1, size=N) for _i in range(0, 5)]))
        noise = np.random.normal(scale=scale, size=N)
        y = np.array(np.log(4.0 + np.sin(4 * X[:, 0]) + np.abs(X[:, 1]) + X[:, 2] ** 2 + X[:, 3] ** 3 + X[:, 4] + 0.1 * noise))
        return X, y


class AVAS(ACE):
    def __init__(self, X, y, cat_X=None, cat_y=False, verbose=True):
        """
        see ACE class
        """
        super().__init__(X, y, cat_X=cat_X, cat_y=cat_y, verbose=verbose)
        if len(self.cat_X) > 0:
            print('categorical X is not implemented')

    def y_loop(self):
        super().y_loop()                      # self.theta[:, 0] is not 0 mean and unit var!

        # Var(theta|phi_sum) ~= E[(theta - phi_sum)^2|phi_sum] because E[theta|phi_sum] ~= phi_sum
        # take log and exp later to avoid negative smooths
        # variance stabilization
        res = self.theta[:, 0] - self.phi_sum
        res = np.where(np.abs(res) < 1.0e-06, 1.0e-06, res)  # avoid log(0)
        lr = np.log(np.sqrt(res ** 2))
        args, bargs = self.var_sort()                       # sort args and back-sort by y
        s_y = self.y[args, 0]                               # sorted y
        s_phi_sum = self.phi_sum[args]                      # y sorted phi_sum
        s_lr = lr[args]                                     # y sorted log|residuals|
        slr_hat = self.cont_smooth(s_phi_sum, s_lr)         # y sorted smoothed log|residuals|
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



