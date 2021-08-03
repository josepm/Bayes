"""
__author__: josep ferrandiz

Y dependent variable
X independent variables
Find theta() and phi_i() that maximize correlation between theta(Y) and sum_{i=1}^p phi_i(X_i) where theta(Y) = sum_{i=1}^p phi_i(X_i)

set use to SuperSmoother or Lowess
if supersoother fails, switch to lowess (but it's slow)
"""

import os
import sys
import pandas as pd
import numpy as np
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from my_projects.ace.supersmoother import ace_smoothers as asm
import my_projects.suggested_vehicle.utilities.utilities as ut
import time
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar, fminbound

N_JOBS = 4 if sys.platform == 'darwin' else os.cpu_count()
N_CPUS = 4 if sys.platform == 'darwin' else os.cpu_count()


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


def drop_cols(X, y, min_corr=0.1):
    # drop uncorrelated columns
    xcols = list()
    print('Initial Correlations')
    for j in range(np.shape(X)[1]):
        corr = np.corrcoef(X[:, j], y)[0, 1]
        print('\tcorr(y, X_' + str(j) + '): ' + str(np.round(corr, 4)))
        if np.abs(corr) >= min_corr:
            xcols.append(j)
    if len(xcols) == 0:
        print('ERROR: correlations are too low')
        sys.exit(-1)

    # drop low corr columns and check duplicates
    f = pd.DataFrame(X[:, xcols])
    f['y'] = np.array(y, dtype=np.float)
    f.drop_duplicates(inplace=True)
    print('final columns: ' + str(xcols))
    return f[[c for c in f.columns if c != 'y']].values, f['y'].values, xcols


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
        :param min_corr: drop X-cols with |Corr(y, C[:, col])| < min_corr
        """
        if self.check_data(X, 'X') is False:
            sys.exit(-1)
        if self.check_data(y, 'y') is False:
            sys.exit(-1)

        self.X_in = X
        self.x_scaler = StandardScaler()                             # scaler for X
        self.X = self.x_scaler.fit_transform(self.X_in)              # self.X shape: (nrows, ncols) (features in columns)
        self.phi = np.zeros_like(self.X)                             # initialize the phi_i functions
        self.nrows, self.ncols = np.shape(self.phi)
        self.cat_X = list() if cat_X is None else cat_X              # col idx that are categorical

        self.y_in = y                                                # self.y shape: (nrows, 1)
        yy = np.reshape(self.y_in, (-1, 1))
        self.y = StandardScaler().fit_transform(yy)                  # self.y shape: (nrows, 1)
        self.theta = self.y                                          # theta(Y)
        self.cat_y = cat_y                                           # True/False
        self.phi_sum = np.array([0.0] * len(self.theta))             # sum_i phi_i(X_i), shape (nrows,)

        self.smoother = smoother                                     # smoother to use
        self.max_cnt = 1000                                          # max ACE/AVAS iterations
        self.ctr = 0                                                 # iteration counter
        self.round = 3                                               # convergence identical digits in correlation
        self.max_same = 3                                            # convergence repeats. Must be > 1
        self.same = 0                                                # convergence correlation repeat counter
        self.verbose = verbose
        self.is_fit = False
        self.var_dict = dict()                                       # cyclic convergence list
        self.max_corr = None
        self.thres = None
        self.f_name = 'wfscore'

        # create noisy versions of the data to avoid division by 0
        if self.cat_y is False:
            self.use_ynoise = None  # set to true if we need y_noisy
            # self.y_noisy = self.add_noise(self.y[:, 0])
        else:
            self.use_ynoise = False
        self.X_noisy = np.zeros_like(X)
        self.use_Xnoise = [None for j in range(self.ncols)]
        for j in self.cat_X:
            self.use_Xnoise[j] = False

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

            # check for NaNs and inf
            if self.check_data(self.phi, 'phi') is False:
                print('ERROR: invalid values for phi')
                retry = False
                break
            if self.check_data(self.theta, 'theta') is False:
                print('ERROR: invalid values for theta')
                ut.to_parquet(pd.DataFrame({'theta': self.theta[:, 0]}), '~/my_data/theta.par')
                retry = False
                break
            self.is_fit = self.check_convergence()
            if self.ctr >= self.max_cnt:
                print('too many iterations::could not converge with tol 1.0e-0' + str(self.round))
                break

        if self.is_fit is True:     # final smooths for prediction
            self.predict_prep()
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

    def predict_prep(self):
        self.final_sm_objs = list()
        for j in range(self.ncols):
            try:
                sobj = self.obj_smooth(self.X[:, j], self.phi[:, j], j)
            except ValueError as e:
                if j not in self.cat_X:
                    xj = self.X_noisy[:, j]  # shape: (nrows, )
                    sobj = self.obj_smooth(xj, self.phi[:, j], j)
                else:
                    print('ERROR: cat smoother for X' + str(j) + ' should not fail')
                    sys.exit(-1)
            self.final_sm_objs.append(sobj)
        try:
            self.pred_smoother = asm.ContSmooth(self.phi_sum, self.y_in, self.smoother)
        except ValueError:
            self.pred_smoother = asm.ContSmooth(self.add_noise(self.phi_sum), self.y_in, self.smoother)

        self.thres = None
        if self.cat_y is True:
            if len(np.unique(self.y_in)) == 2:
                self.set_thres()

    def set_thres(self):
        # sets threshold for classification
        def f_precision(t):  # maximize recall
            # precision = (yhat >= t & y_in == 1) / yhat >= t
            y = np.where(y_hat >= t, 1, 0)
            num = np.sum(y * y_in)
            den = np.sum(y)
            return -(num / den)

        def f_wprecision(t):  # maximize recall
            y = np.where(y_hat >= t, 1, 0)
            num = np.sum(y_hat * y * y_in) / np.sum(y_hat)
            den = np.sum(y)
            return -(num / den)

        def f_recall(t):  # maximize recall
            # recall = (yhat >= t & y_in == 1) / yin == 1
            y = np.where(y_hat >= t, 1, 0)
            num = np.sum(y * y_in)
            den = np.sum(y_in)
            return -(num / den)

        def f_wrecall(t):  # maximize recall
            y = np.where(y_hat >= t, 1, 0)
            num = np.sum(y_hat * y * y_in) / np.sum(y_hat)
            den = np.sum(y_in)
            return -(num / den)

        def f_fscore(t):
            pre = f_precision(t)
            rec = f_recall(t)
            return (1 + beta ** 2) * pre * rec / ((beta ** 2) * pre + rec)

        def f_wfscore(t):
            pre = f_wprecision(t)
            rec = f_wrecall(t)
            return (1 + beta ** 2) * pre * rec / ((beta ** 2) * pre + rec)

        def f_rmse(t):  # rmse
            y = np.where(y_hat >= t, 1, 0)
            return np.mean((y - y_in) ** 2)

        y_hat = self.predict(self.X_in)
        y_in = self.y_in
        y_hat = y_hat[~np.isnan(y_hat)]  # used in func
        y_in = y_in[~np.isnan(y_hat)]  # used in func
        if self.f_name == 'recall':
            func = f_recall
        elif self.f_name == 'wrecall':
            func = f_wrecall
        elif self.f_name == 'precision':
            func = f_precision
        elif self.f_name == 'wprecision':
            func = f_wprecision
        elif self.f_name == 'f_fscore':
            beta = 1.0
            func = f_fscore
        elif self.f_name == 'f_wfscore':
            beta = 1.0
            func = f_wfscore
        else:
            func = f_rmse
        res = minimize_scalar(func, bounds=(0.0, 1.0), method='bounded')
        self.thres = res.x if res.status == 0 else None

    @staticmethod
    def add_noise(yarr, m=1.0e-06):  # prevent identical values in super_smoother
        if len(yarr) > len(np.unique(yarr)):
            std_dev = np.std(yarr) * m if np.std(yarr) > 0 else np.abs(np.mean(yarr * m))
            noise = np.random.normal(0, std_dev, np.shape(yarr))
            print('\tWARNING: adding noise::len: ' + str(len(yarr)) + ' unique: ' + str(len(np.unique(yarr))) +
                  ' yin mean: ' + str(np.mean(yarr)) + ' yout mean: ' + str(np.mean(yarr + noise)) +
                  ' yin std: ' + str((np.std(yarr))) + ' yout std: ' + str((np.std(yarr + noise))))
            return yarr + noise
        else:
            return yarr

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
        if self.use_Xnoise[j] is None:   # set Xnoise to use Xj or Xj noisy for the rest of the iteration
            try:
                xj = self.X[:, j]                      # shape: (nrows, )
                self.phi[:, j] = self.obj_smooth(xj, zj, j).predict(xj)
                self.use_Xnoise[j] = False
            except ValueError as e:
                if j not in self.cat_X:
                    self.X_noisy[:, j] = self.add_noise(self.X[:, j])
                    xj = self.X_noisy[:, j]            # shape: (nrows, )
                    self.phi[:, j] = self.obj_smooth(xj, zj, j).predict(xj)
                    self.use_Xnoise[j] = True
                    print('WARNING: using noisy version for column ' + str(j))
                else:
                    print('ERROR: categorical X smoother should not fail on value error')
                    sys.exit(-1)
        else:                              # Xnoise is set
            xj = self.X[:, j] if self.use_Xnoise[j] is False else self.X_noisy[:, j]  # shape: (nrows, )
            self.phi[:, j] = self.obj_smooth(xj, zj, j).predict(xj)

        self.phi[:, j] = (self.phi[:, j] - np.mean(self.phi[:, j])) / np.std(self.phi[:, j])
        self.phi_sum += (self.phi[:, j] - phij)

    def y_loop(self):
        """
        updates theta = E[phi_sum|Y]
        """
        if self.use_ynoise is None:       # set ynoise to use y or y_noisy for the rest of the iteration
            try:
                theta = self.obj_smooth(self.y[:, 0], self.phi_sum, None).predict(self.y[:, 0])
                self.use_ynoise = False
            except ValueError as e:
                if self.cat_y is False:
                    self.y_noisy = self.add_noise(self.y[:, 0])
                    theta = self.obj_smooth(self.y_noisy, self.phi_sum, None).predict(self.y[:, 0])
                    self.use_ynoise = True
                    print('WARNING: using noisy version for y ' + str(j))
                else:
                    print('ERROR: categorical y smoother should not fail on value error')
                    sys.exit(-1)
        else:                             # ynoise set
            x = self.y[:, 0] if self.use_ynoise is False else self.y_noisy
            theta = self.obj_smooth(x, self.phi_sum, None).predict(x)

        self.theta = np.reshape(theta, (-1, 1))
        self.theta = StandardScaler().fit_transform(self.theta)  # shape: (nrows, 1)

    def obj_smooth(self, x, y, j):
        """
        :param x: independent value
        :param y: dependent variable (E[y|x])
        :param j: column number or None
        :return: a smoother object consistent with x (categorical or continuous)
        """
        if j is None:  # y_loop
            return asm.CatSmooth(x, y) if self.cat_y is True else asm.ContSmooth(x, y, self.smoother)
        else:          # X_loop
            return asm.CatSmooth(x, y) if j in self.cat_X else asm.ContSmooth(x, y, self.smoother)

    def predict(self, X, thres=None):
        """
        predict the y's from some X, if thres = None, returns the y-score
        smoother is more accurate than interpolation
        :param X: input values to predict (must have same cols -features- as input X, any rows)
        :param thres: if None, predict scores. Otherwise predict classes (2 class case only)
        :return: predicted values (y_hat, not theta(y))
        """
        if self.is_fit:
            if np.shape(X)[1] != self.ncols:
                print('invalid input to predict:: columns: ' + str(np.shape(X)[1]) + ' and should be ' + str(self.ncols))
                return None
            if self.check_data(X, 'predict data') is False:
                return None
            Xs = self.x_scaler.transform(X)
            phi_sum_list = list()    # np.zeros_like(len(X), dtype=np.float)
            for j in range(self.ncols):
                try:
                    phi_sum_list.append(self.final_sm_objs[j].predict(Xs[:, j]))
                except ValueError as e:
                    xj = self.add_noise(Xs[:, j])
                    try:
                        phi_sum_list.append(self.final_sm_objs[j].predict(xj))
                    except ValueError:
                        print('ERROR: fatal for column ' + str(j))
                        s = pd.Series(xj)
                        print(s.describe())
                        print(s.nunique())
                        sys.exit(-1)
            phi_sum_arr = np.reshape(np.array(phi_sum_list, dtype=np.float), np.shape(Xs))
            phi_sum = np.sum(phi_sum_arr, axis=1)
            phi_sum = (phi_sum - np.mean(phi_sum)) / np.std(phi_sum)
            ysm = self.pred_smoother.predict(phi_sum)
            return ysm if thres is None else np.where(ysm >= thres, 1, 0)
        else:
            print('must fit model first')
            return None

    def smry_DF(self):
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
        return df

    def plots(self):
        df = self.smry_DF()
        if len(df) > 10000:
            df = df.sample(n=10000, axis=0)
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
            df.plot(kind='scatter', x='X_' + str(j), y='phi_' + str(j), grid=True, title='phi_' + str(j) + ' transform. \nCorr(X_' + str(j) + ', phi_' + str(j) + '): ' + str(np.round(corr, 4)))

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
        var = np.round(np.mean((self.theta[:, 0] - self.phi_sum) ** 2) / np.mean(self.theta[:, 0] ** 2), self.round)
        for k in self.var_dict.keys():
            self.var_dict[k] **= discount              # discount old observed values
        if var not in self.var_dict.keys():
            self.var_dict[var] = 0
        self.var_dict[var] += 1.0
        dict_vals = np.array(list(self.var_dict.values()))
        args = np.argwhere(dict_vals > 2)               # recent repeating values must have a score > 2
        if self.verbose:
            top_keys = dict(sorted(self.var_dict.items(), key=itemgetter(1), reverse=True)[:5])
            print('>>>>> ctr: ' + str(self.ctr) +
                  ' curr_corr: ' + str(corr_coef) + ' var: ' + str(var) +
                  ' vars: ' + str({k: np.round(v, self.round) for k, v in top_keys.items() if v > 1}) +
                  ' time: ' + str(np.round(time.time() - self.tic, 4)))
        b = np.max(dict_vals[args]) > self.max_same if len(args) > 0 else False
        return bool(b)  # must have bool here


class AVAS(ACE):
    def __init__(self, X, y, smoother, cat_X=None, cat_y=False, verbose=True):
        """
        see ACE class
        """
        if cat_y is True:
            raise NotImplementedError('cat_y True not implemented for AVAS')
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
        # htheta = np.array([np.trapz(1.0 / np.sqrt(vu_[:ix + 1]), s_theta[:ix + 1]) for ix in range(self.nrows)])
        ix_theta = Parallel(n_jobs=N_JOBS)(delayed(self.trap_sum)(ix, vu_, s_theta) for ix in range(self.nrows))
        ix_theta.sort(key=lambda x: x[0])
        htheta = np.array([x[1] for x in ix_theta])

        self.theta = np.reshape(htheta[bargs], (-1, 1))          # restore the original order and reshape
        self.theta = StandardScaler().fit_transform(self.theta)  # shape: (nrows, 1)

    def var_sort(self):
        args = np.argsort(self.theta[:, 0])
        bargs = np.zeros_like(args)
        for i, si in enumerate(args):    # to restore original order
            bargs[si] = i
        return args, bargs

    @staticmethod
    def trap_sum(ix, vy, vx):
        a = np.trapz(1.0 / np.sqrt(vy[:ix + 1]), vx[:ix + 1])
        return ix, a
