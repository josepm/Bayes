"""
from https://gist.github.com/agramfort/850437
also http://ml.stat.purdue.edu/hafen/preprints/Hafen_thesis.pdf
This module implements the Lowess function for nonparametric regression.
Functions:
lowess Fit a smooth nonparametric regression curve to a scatterplot.
For more information, see
William S. Cleveland: "Robust locally weighted regression and smoothing
scatterplots", Journal of the American Statistical Association, December 1979,
volume 74, number 368, pp. 829-836.
William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
approach to regression analysis by local fitting", Journal of the American
Statistical Association, September 1988, volume 83, number 403, pp. 596-610.

Usage:
y_loess = lowess(x, y, f=2./3., iter_=3)
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg
import statsmodels.api as sm
from statsmodels.tools.eval_measures import aic_sigma, aicc_sigma


class Loess(object):
    """
     Lowess smoother class: Robust locally weighted regression.
     Fits a nonparametric regression curve to a scatterplot.
     The arrays x and y contain an equal number of elements; each pair
     (x[i], y[i]) defines a data point in the scatterplot.
     The function returns the estimated (smooth) values of y.
     The smoothing span (bandwidth) is given by f.
     A larger value for f will result in a smoother curve.
     The number of robustifying iterations is given by iter_.
     The function will run faster with a smaller number of iterations.
     x: np.array
     y: np.array
     f: relative bandwidth (0 < f < 1)
     returns the smoothed values of y
     """
    def __init__(self, x, y, f, iter_=3, degree=1):
        self.x = x
        self.y = y
        self.f = f
        self.n = len(x)
        self.iter_ = iter_
        self.degree = degree
        self.L, self.Lambda = self.operator_matrix()
        self.pars = np.trace(np.dot(np.transpose(self.L), self.L))

    def loess_weights(self):
        r = int(np.ceil(self.f * self.n))
        h = [np.sort(np.abs(self.x - self.x[i]))[r] for i in range(self.n)]
        w = np.clip(np.abs((self.x[:, None] - self.x[None, :]) / h), 0.0, 1.0)
        w = (1 - w ** 3) ** 3
        return w  # the j column of w contains the weights of each point on the jth point

    @staticmethod
    def _set_delta(residuals):
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        return (1 - delta ** 2) ** 2

    def operator_matrix(self):
        # get the L-matrix and the Lambda-matrix, Lambda = (I-L)^T (I-L)
        w = self.loess_weights()
        delta = np.ones(self.n)
        L = None
        for iteration in range(self.iter_):
            L, delta = self._operator_matrix(w, delta)
        IL = np.identity(self.n) - L
        Lambda = np.dot(np.transpose(IL), IL)
        return L, Lambda

    def _operator_matrix(self, w, delta):
        l_rows = list()
        for i in range(self.n):  # ith row of L
            wi = delta * w[:, i]
            nzi = np.nonzero(wi)[0]
            x_cols_ = [(self.x - self.x[i]) ** n for n in range(self.degree+1)]
            x_cols = [c[nzi] for c in x_cols_]
            X = np.column_stack(x_cols)
            W = np.diag(wi[wi != 0.0])
            Xt = np.transpose(X)
            H = np.dot(np.dot(Xt, W), X)
            Hinv = linalg.inv(H)
            lr = np.dot(np.dot(Hinv, Xt), W)
            lx = np.zeros(self.n)
            lx[nzi] = lr[0]
            l_rows.append(lx)
        L = np.array(l_rows)
        IL = np.identity(self.n) - L
        residuals = np.dot(IL, self.y)
        delta = _set_delta(residuals)
        return L, delta

    def loess_ssr(self):
        resid = self.loess_resid()
        return np.sum(resid ** 2)

    def loess_resid(self):
        return self.y - np.dot(self.L, self.y)

    def loess_var(self):
        # loess model variance
        resid = self.loess_resid()
        return np.sum(resid ** 2) / np.trace(self.Lambda)

    def loess_mallows(self):
        ssr = self.loess_ssr()
        sig2 = self.loess_var()
        tr = np.trace(self.Lambda)
        return ssr / sig2 + self.pars - tr


def loess_bandwith(x, y, iter_=3, degree=1, fmin=0.1, fmax=0.9, npts=10):
    # returns best bandwidth
    f_arr = np.linspace(fmin, fmax, 1 + int((fmax - fmin) * npts))
    opt_f, opt_perf, opt_L = fmin, np.inf, None
    for f in f_arr:
        loess_obj = Loess(x, y, f, iter_=iter_, degree=degree)
        perf = loess_obj.loess_mallows()
        print('f: ' + str(f) + ' perf: ' + str(perf))
        if perf < opt_perf:
            opt_f = f
            opt_perf = perf
            opt_L = loess_obj.L
    return opt_f, opt_L


def opt_loess(x, y, iter_=3, degree=1, fmin=0.1, fmax=0.9, npts=10):
    f, L = loess_bandwith(x, y, iter_=iter_, degree=degree, fmin=fmin, fmax=fmax, npts=npts)
    return np.dot(L, y)


def loess(y, x, f=0.66, iter_=3):
    """
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot.
    The function returns the estimated (smooth) values of y.
    The smoothing span (bandwidth) is given by f.
    A larger value for f will result in a smoother curve.
    The number of robustifying iterations is given by iter_.
    The function will run faster with a smaller number of iterations.
    x: np.array
    y: np.array
    f: relative bandwidth (0 < f < 1)
    returns the smoothed values of y
    """
    n = len(x)
    w = loess_weights(x, f)
    yhat = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter_):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yhat[i] = beta[0] + beta[1] * x[i]
        delta = _set_delta(y - yhat)
    return yhat


def _set_delta(residuals):
    s = np.median(np.abs(residuals))
    delta = np.clip(residuals / (6.0 * s), -1, 1)
    return (1 - delta ** 2) ** 2


def loess_weights(x, f):
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    return w   # the j column of w contains the weights of each point on the jth point

# def operator_matrix(x, y, f, iter_=3):
    # get the L-matrix and the Lambda-matrix, Lambda = (I-L)^T (I-L)
    # w = loess_weights(x, f)
    # delta = np.ones(len(x))
    # L = None
    # for iteration in range(iter_):
    #     L, delta = _operator_matrix(x, y, w, delta)
    # IL = np.identity(len(x)) - L
    # Lambda = np.dot(np.transpose(IL), IL)
    # return L, Lambda
#
#
# def _operator_matrix(x, y, w, delta, degree=1):
#     l_rows = list()
#     n = len(x)
#     for i in range(n):  # ith row of L
#         wi = delta * w[:, i]
#         nzi = np.nonzero(wi)[0]
#         x_cols_ = [(x - x[i]) ** n for n in range(degree+1)]
#         x_cols = [c[nzi] for c in x_cols_]
#         X = np.column_stack(x_cols)
#         W = np.diag(wi[wi != 0.0])
#         Xt = np.transpose(X)
#         H = np.dot(np.dot(Xt, W), X)
#         Hinv = linalg.inv(H)
#         lr = np.dot(np.dot(Hinv, Xt), W)
#         lx = np.zeros(n)
#         lx[nzi] = lr[0]
#         l_rows.append(lx)
#     L = np.array(l_rows)
#     IL = np.identity(n) - L
#     residuals = np.dot(IL, y)
#     delta = _set_delta(residuals)
#     return L, delta


# def loess_ssr(x, y, f, L=None):
#     if L is None:
#         L, Lambda = operator_matrix(x, y, f)
#     resid = loess_resid(x, y, f, L=L)
#     return np.sum(resid ** 2)
#
#
# def loess_resid(x, y, f, L=None):
#     if L is None:
#         L, Lambda = operator_matrix(x, y, f)
#     return y - np.dot(L, y)
#
#
# def loess_pars(x, y, f, L=None):
#     if L is None:
#         L, _ = operator_matrix(x, y, f)
#     return np.trace(np.dot(np.transpose(L), L))
#
#
# def loess_var(x, y, f, L=None, Lambda=None):
#     loess model variance
    # if L is None or Lambda is None:
    #     L, Lambda = operator_matrix(x, y, f)
    # resid = loess_resid(x, y, f, L=L)
    # return np.sum(resid ** 2) / np.trace(Lambda)


#
# def bandwith_performance(x, y, f, iter_=3):
#     L, Lambda = operator_matrix(x, y, f, iter_=iter_)
#     sig2 = loess_var(x, y, f, L=L, Lambda=Lambda)
#     pars = loess_pars(x, y, f, L=L)
#     aic = aicc_sigma(sig2, len(x), pars)
#     m = loess_mallows(x, y, f, L=L, Lamba=Lambda)
#     print(str(f) + ' sig2: ' + str(sig2) + ' pars: ' + str(pars) + ' aicc: ' + str(aic))
#     return aic, L
#
#
# def loess_mallows(x, y, f, L=None, Lambda=None):
#     if L is None or Lambda is None:
#         L, Lambda = operator_matrix(x, y, f)
#     ssr = loess_ssr(x, y, f, L=L)
#     sig2 = loess_var(x, y, f, L=L, Lambda=Lambda)
#     pars = loess_pars(x, y, f, L=L)
#     tr = np.trace(Lambda)
#     return ssr / sig2 + pars - tr
#
#
# def loess_bandwith(x, y, iter_=3, fmin=0.1, fmax=0.9, npts=10):
#     returns best bandwidth
    # f_arr = np.linspace(fmin, fmax, 1 + int((fmax - fmin) * npts))
    # opt_f, opt_perf, opt_L = fmin, np.inf, None
    # for f in f_arr:
    #     perf, L = bandwith_performance(x, y, f, iter_=iter_)
    #     if perf < opt_perf:
    #         opt_f = f
    #         opt_perf = perf
    #         opt_L = L
    # return opt_f, opt_L
#
#
# def opt_loess(x, y, iter_=3, fmin=0.1, fmax=0.9, npts=10):
#     f, L = loess_bandwith(x, y, iter_=iter_, fmin=fmin, fmax=fmax, npts=npts)
#     return np.dot(L, y)
#


