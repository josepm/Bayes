"""
see https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/
see https://github.com/luke14free/pm-prophet
see https://github.com/facebook/prophet/tree/1053a6e9ce935ff29c8f69f56d0a6b3c3397520e/python/fbprophet
see https://docs.pymc.io/notebooks/dp_mix.html

"""

from typing import List, Union

import multiprocessing as mp
try:
    mp.set_start_method('fork')     # forkserver does not work
except RuntimeError:
    pass

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from collections import defaultdict
from capacity_planning.PyMC3.pmprophet.pymc_prophet import plot as p_plt
from capacity_planning.PyMC3.pmprophet.pymc_prophet import utilities as ut


class PyMCProphet(pm.Model):
    def __init__(
        self,
        data: pd.DataFrame,
        growth: str = None,
        floor: float = None,
        ceiling: float = None,
        name: str = None,
        changepoints: list = None,
        n_changepoints: int = 25,
        changepoints_prior_scale: float = 0.05,
        changepoint_range: float = 0.8,
        seasonality_prior_scale: float = 0.05,
        holidays_prior_scale: float = 2.5,
        regressors_prior_scale: bool = 2.5,
        sigma_tau: float = 0.5,
        growth_prior_scale: float = 5.0,
        offset_prior_scale: float = 5.0,
        positive_regressors_coefficients: bool = False,
        cutoff_date: str = None,
        freq: str = None,
        seasonalities: List = None,
        regressors: pd.DataFrame = None,
        holidays: pd.DataFrame = None,
        merge_seasonalities: bool = True,
        mcmc: bool = True,
        additive: list = None,        # regressors, seasonalities, holidays
        multiplicative: list = None   # regressors, seasonalities, holidays
    ):
        """
        PyMCProphet class
        :param data: DF with `ds` and `y` columns
        :param growth: type of trend growth. None, linear or logistic
        :param floor: lower bound for logistic growth otherwise None
        :param ceiling: upper bound for logistic growth otherwise None
        :param name: instance name
        :param changepoints: datetimes of known change points (as string)
        :param n_changepoints: number of changepoints
        :param changepoints_prior_scale: scale parameter for the changepoint prior
        :param changepoint_range: fraction of actuals used to learn about change points (default: 0.8)
        :param seasonality_prior_scale: scale parameter for the seasonality prior
        :param holidays_prior_scale: scale parameter for the holidays prior
        :param regressors_prior_scale: scale parameter for the regressors prior
        :param sigma_tau: scale parameter for the sigma prior
        :param growth_prior_scale: scale parameter for the growth rate prior
        :param offset_prior_scale: scale parameter for the trend intercept
        :param positive_regressors_coefficients: if True use positive coef for regressors. Default False
        :param cutoff_date: last date of actuals used in forecast
        :param freq: time series frequency (D, W, M, ...).
        :param seasonalities: seasonality tuples, each of the form (<name>, <period>, <fourier_order>)
        :param regressors: DF with regressors data. Has a `ds` column and a column for each regressor. Must extend to the forecast date
        :param holidays: DF with holidays with columns `holiday` (holiday name) and `ds` (holiday date). Adjacent dates to a holiday must be included in this format, e.g. 12/24
        :param merge_seasonalities: True: add all fourier components of a seasonality element. Default True
        :param mcmc: True, do MCMC. Default True. Otherwise MAP estimation
        :param additive: list of additive components. Default None. Must be in ['regressor', 'seasonality', 'holiday']
        :param multiplicative: list of multiplicative components. Default None. Must be in ['regressor', 'seasonality', 'holiday']
        """

        # data checks
        if 'y' not in data.columns:
            raise Exception('Target variable should be called `y` in the `data` dataframe')
        if 'ds' not in data.columns:
            raise Exception('Time variable should be called `ds` in the `data` dataframe')
        if name is None:
            raise Exception('Specify a model name through the `name` parameter')

        self.reserved_names = ['y', 'scaled_y', 'ds', 't', 'yhat', 'yhat_scaled']

        super().__init__()
        self.name = name
        self.mcmc = mcmc
        self.map = None
        self.my_model = pm.Model()
        self.fit_data = dict()

        # check the input data
        self.freq = freq
        data.sort_values(by='ds', inplace=True)
        data.reset_index(drop=True, inplace=True)
        data, self.cutoff_date = self.setup_data(data, cutoff_date)

        # set growth model
        self.linearizer = None
        self.floor = floor
        self.ceiling = ceiling
        self.growth = self.check_growth(growth)
        if self.growth == 'logistic':
            self.linearizer = ut.Linearizer(self.ceiling, self.floor, eps=1.0e-8)
            data['y'] = self.linearizer.transform(data['y'].values)

        # scale y values
        self.scaled_dict = dict()
        ut.ScaledData('y', data['y'], self.scaled_dict)

        # get float time (column t) and set dates
        self.max_ds, self.min_ds, self.ts_scale, self.cutoff_time, dt = self.set_times(data)

        # input data frame
        self.data = pd.DataFrame({'ds': data['ds'],
                                  't': dt,
                                  'scaled_y': self.scaled_dict['y'].scaled_data,
                                  'y': self.scaled_dict['y'].data_in})

        # set other instance components
        self.trace = {}
        self.n_changepoints = n_changepoints
        self.forecasting_periods = 0
        self.spp = None
        self.fcast_df = pd.DataFrame()
        self.changepoints_t = None
        self.changepoints_ds = None
        self.fitted = False
        self.seasons_df = None

        # priors stuff
        # Note: putting priors in a dictionary does not seem to work. Not sure why??
        self.growth_prior_scale = growth_prior_scale
        self.offset_prior_scale = offset_prior_scale
        self.changepoint_range = changepoint_range
        self.changepoints_prior_scale = changepoints_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.regressors_prior_scale = regressors_prior_scale
        self.sigma_tau = sigma_tau
        self.positive_regressors_coefficients = positive_regressors_coefficients

        # add model components
        self.seasonality_info = dict()
        self.merge_seasonalities = merge_seasonalities  # add the components of the same seasonality to a single value
        self.regressor_names = []
        self.regressor_components = []
        self.holiday_components = []
        self.growth_components = []
        self.seasonality_components = []
        self.seasonalities = seasonalities
        self.regressors = regressors
        self.holidays = holidays
        self.regressors_data = dict()
        self.holidays_data = dict()
        self.add_seasonalities()
        self.add_holidays()
        self.add_regressors()
        self.set_changepoints(changepoints)
        self.additive = list() if additive is None else additive
        self.additive_components = list()
        self.multiplicative = list() if multiplicative is None else multiplicative
        self.multiplicative_components = list()

        self.lambda_arr = list()
        self.new_cpt_cnt = list()
        self.delta_arr = list()
        self.mdl_components = list()

    def set_times(self, data):
        max_ds = data['ds'].max()
        min_ds = data['ds'].min()
        ts_scale = self.set_ts_scale(max_ds, min_ds)
        dt = (data['ds'] - data['ds'].min()) / (data['ds'].max() - data['ds'].min())                     # ratio of time deltas is a float
        cutoff_time = (self.cutoff_date - data['ds'].min()) / (data['ds'].max() - data['ds'].min())       # cutoff_date in float time
        return max_ds, min_ds, ts_scale, cutoff_time, dt

    def set_ts_scale(self, ds_max, ds_min):
        ts_scale = (ds_max - ds_min).days
        if self.freq == 'D':
            ts_scale /= 1.0
        elif self.freq == 'W':
            ts_scale /= 7.0
        elif self.freq == 'M':
            ts_scale /= 30.4375
        else:
            raise Exception('frequency not supported: ' + str(self.freq))
        if ts_scale == 0.0:
            ts_scale = 1.0
        return ts_scale

    def set_changepoints(self, changepoints):
        if self.n_changepoints > len(self.data) * self.changepoint_range:
            self.n_changepoints = int(np.floor(self.n_changepoints * len(self.data)))
            print('n_changepoints reset to ' + str(self.n_changepoints))

        cp_ds = list(pd.date_range(start=self.min_ds, end=self.max_ds, periods=self.n_changepoints + 2))[1:-1] if self.n_changepoints > 0 else list()
        cp_t = list(np.linspace(start=self.data['t'].min(), stop=self.data['t'].max(), num=self.n_changepoints + 2))[1:-1] if self.n_changepoints > 0 else list()
        c_ds, c_ts = self._set_cpts(changepoints)
        self.changepoints_ds = c_ds + cp_ds
        self.changepoints_ds.sort()
        self.changepoints_t = c_ts + cp_t
        self.changepoints_t.sort()

    def _set_cpts(self, changepoints):
        c_ds = list()
        if changepoints is not None:
            if len(changepoints) > 0:
                c_ds = [pd.to_datetime(x) for x in changepoints]
        c_ts = [(ds - self.min_ds) / (self.max_ds - self.min_ds) for ds in c_ds]
        return c_ds, c_ts

    def set_auto_changepoints1(self, changepoints):
        min_period = min([s[1] for s in self.seasonalities])
        cpf = self.data[['ds', 't', 'y']].set_index('ds').resample(str(min_period) + self.freq).sum().reset_index()
        g = np.gradient(cpf['y'].values)                # trend
        cpf['w_trend'] = np.abs(g) / np.sum(np.abs(g))  # changepoint density at each point
        cpf['trend'] = g
        ndata = len(cpf)
        bic_min, cpts = np.inf, None
        for k in range(1, self.n_changepoints):
            keep = cpf.nlargest(k, columns=['w_trend'])                                   # pick k-largest
            sse = np.sum(cpf.nsmallest(ndata - k, columns=['w_trend'])['w_trend'] ** 2)   # sse on ignored changepoints
            bic = -2.0 * ndata * np.log(sse / ndata) + k * np.log(ndata)
            if bic < bic_min:
                bic_min = bic
                cpts = keep
        print(':: n_changepoints: ' + str(len(cpts)))
        c_ds, c_ts = self._set_cpts(changepoints)
        self.changepoints_ds = c_ds + list(cpts['ds'].values)
        self.changepoints_ds.sort()
        self.changepoints_t = c_ts + list(cpts['t'].values)
        self.changepoints_t.sort()

    def set_auto_changepoints(self):
        def stick_breaking(beta_):
            _remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta_)[:-1]])
            return beta_ * _remaining

        min_period = min([s[1] for s in self.seasonalities])
        cpf = self.data[['ds', 'y']].set_index('ds').resample(str(min_period) + self.freq).sum().reset_index()
        _, _, _, _, cpf['t'] = self.set_times(cpf)  # add 't'
        g = np.gradient(cpf['y'].values)                 # trend
        cpf['w_trend'] = np.abs(g) / np.sum(np.abs(g))   # changepoint density at each point
        cpf['trend'] = g

        k = self.n_changepoints
        with pm.Model:
            alpha = pm.Gamma('alpha', 1.0, 1.0, shape=1)
            beta = pm.Beta('beta', 1.0, alpha, shape=k)
            w = pm.Deterministic('w', stick_breaking(beta))
            h = pm.Deterministic('constant', 1.0, shape=k)
            cpt = pm.Mixture(w=w, comp_dists=h, obs=cpf['w_trend'].values)
            step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
            cpt_trace = pm.sample(500, chains=None, step=step_method, tune=500)
            cpt_smry = pm.summary(cpt_trace)
            pm.traceplot(cpt_trace)
            spp = pm.sample_posterior_predictive(cpt_trace, samples=1000, progressbar=False)
            return cpt_smry

    def check_growth(self, growth):
        if growth is None:
            self.n_changepoints = 0
            return None
        elif growth == 'linear':
            return 'linear'
        elif growth == 'logistic':   # to-linear transform after scaling?
            if self.floor is None:
                raise Exception('logistic growth has no floor set')
            if self.ceiling is None:
                raise Exception('logistic growth has no ceiling set')
            if self.floor >= self.ceiling:
                raise Exception('invalid floor and ceiling values')
            return 'logistic'
        else:
            raise Exception('invalid growth ' + str(growth))

    def setup_data(self, data, cutoff_date):
        """
        convert ds to datetime
        impute y-values
        set cutoff date as datetime. default to max(ds) if None
        :param data: DF with ds and y columns
        :param cutoff_date: cutoff date if any
        :return: DF with ds and y values (imputed) and cutoff date
        """
        if 'ds' not in data.columns:
            raise Exception('input DF must contain a `ds` column')
        if 'y' not in data.columns:
            raise Exception('input DF must contain a `y` column')

        dta = data.copy()
        dta['ds'] = pd.to_datetime(dta['ds'].values)
        if dta['ds'].max() == dta['ds'].min():
            raise Exception('invalid date range')
        dr = pd.date_range(dta['ds'].min(), dta['ds'].max(), freq=self.freq)
        data_df = pd.DataFrame({'ds': dr}).merge(dta, on='ds', how='left').reset_index()  # may have NAs

        # find data gaps, max gap size and fill
        # fill: use interpolate because of serial correlations
        if data_df['y'].isnull().sum() > 0:
            data_df['y'] = ut.data_gaps(data_df['y'].copy(), fill_data=True)

        if cutoff_date is None:       # Note: cutoff_date = None does not generate a ValueError
            cutoff_date = data_df['ds'].max()
        try:
            cutoff_date = pd.to_datetime(cutoff_date)
        except ValueError:
            raise Exception('invalid cutoff date: ' + str(cutoff_date))

        if data_df['ds'].max() < cutoff_date:
            raise Exception('Cutoff date and max date inconsistent')
        else:
            df = data_df[data_df['ds'] <= cutoff_date].copy()
        return df, cutoff_date

    def set_ts(self, f, min_ds=None, max_ds=None):
        """
        maps ds (datetime) column to t (time as float) column
        :param f: ds column in datetime
        :param min_ds:
        :param max_ds:
        :return: normalized time as float
        """
        if min_ds is None:
            min_ds = self.min_ds
        if max_ds is None:
            max_ds = self.max_ds
        if min_ds == max_ds:
            raise Exception('min and max are identical. cannot scale')
        f = pd.to_datetime(f.values)
        return (f - min_ds) / (max_ds - min_ds)  # the ratio of timedeltas produces a float!

    def check_reserved(self, name):
        # make sure name has not been used before or is not a reserved name
        if name in self.reserved_names:
            raise Exception('reserved names:: name \'' + str(name) + '\' already in use')
        else:
            self.reserved_names.append(name)

    def actual_cpt(self, t_max=1.0, thres=0.25):
        # drop noisy changepoints when plotting
        if self.growth is None:
            return list()
        else:
            new_changepoints_ds, new_deltas = self.new_cpt_smry(t_max)
            cpt_arr = self.changepoints_ds + new_changepoints_ds
            if len(cpt_arr) == 0:
                return list()
            else:
                delta_arr = np.nanmean(self.trace['delta'], axis=0) if self.mcmc else self.map['delta']
                v = np.concatenate((np.abs(delta_arr), np.array(new_deltas)))
                d_thres = thres * np.mean(v)  # dynamic threshold
                return [cpt_arr[i] for i in range(len(cpt_arr)) if v[i] > d_thres]

    # ###########################################################
    # ###########################################################
    # seasonalities
    # ###########################################################
    # ###########################################################
    @staticmethod
    def fourier_series(dates, period, series_order):
        w = 2 * np.pi * np.arange(1, series_order + 1) / period         # 2 pi n / p
        x = w * dates[:, None]                                          # 2 pi n / p * t
        return np.concatenate(((1.0 + np.cos(x)) / 2.0, (1.0 + np.sin(x)) / 2.0), axis=1)   # scale between 0 and 1

    def add_seasonalities(self):
        """
        adds all seasonalities to the model
        :return:
        """
        if self.seasonalities is not None:
            _ = [self.add_seasonality(name=name, period=period, fourier_order=order) for name, period, order in self.seasonalities]
        else:
            print('no seasonality info provided. Skipping')
            return

    def add_seasonality(self, name: str, period: Union[int, float], fourier_order: int):
        """
        add a seasonal component to the model
        may merge all fourier elements into a single one
        sets prior
        :param name: seasonal component name
        :param period: period in self.freq units
        :param fourier_order: number of fourier components
        :return: order
        """
        component = self._seasonality_error_check(name, period)
        fs = self.fourier_series(self.data['t'], period / self.ts_scale, fourier_order)
        s_df = pd.DataFrame(fs, columns=[component + '_' + str(i) for i in range(2 * fourier_order)])
        df_cols = list()
        for order_idx in range(2 * fourier_order):
            skey = component + '_' + str(order_idx)
            self.seasonality_info[skey] = (period, fourier_order)
            self.data[skey] = s_df[skey]
            df_cols.append(skey)

        # merge components of a seasonality and have one prior per seasonality
        if self.merge_seasonalities:
            self.seasonality_info[component] = (period, fourier_order)
            self.data[component] = ut.ScaledData(component, s_df[df_cols].sum(axis=1), self.scaled_dict).scaled_data   # add all fourier components
            self.data.drop(df_cols, axis=1, inplace=True)
            with self.my_model:
                # >>>>>>>>>>>>>>> keep shape parameter: runs much faster!!!?????? <<<<<<<<<<<<<<<<<<<
                setattr(self, component, pm.Laplace(component, 0, self.seasonality_prior_scale, shape=1))                  # create self.<seasonality_name>
        else:      # one prior per seasonality component: >>>>>>>>> no need to scale <<<<<<<<<<
            with self.my_model:
                setattr(self, component, pm.Laplace(component, 0, self.seasonality_prior_scale, shape=2 * fourier_order))  # create self.<seasonality_name_index>
            _ = [self.seasonality_info.pop(k) for k in list(self.seasonality_info.keys()) if component in k]               # drop components of the form component_<idx>
        self.seasonality_info[component] = (period, fourier_order)
        self.seasonality_components.append(component)
        return fourier_order

    def _seasonality_error_check(self, name, period):
        # check seasonality input errors
        self.check_reserved(name)
        component = name
        if component in self.seasonality_info.keys():
            raise Exception('duplicated seasonality component: ' + component)
        if period <= 0.0:
            raise Exception('period must be positive: ' + str(period))
        periods = [x[0] for x in self.seasonality_info.values()]
        if period in periods:
            raise Exception('duplicated period period: ' + str(period) + ' for ' + str(name))
        return component

    # ###########################################################
    # ###########################################################
    # holidays and events
    # ###########################################################
    # ###########################################################
    def add_holidays(self):
        """
        add holiday components to the model
        may merge holidays
        sets prior(s)
        :return:
        """
        if self.holidays is not None:
            for h in self.holidays['holiday'].unique():
                self.add_holiday(h, self.holidays[self.holidays['holiday'] == h].copy())
        else:
            print('no holidays/events provided. Skipping')
            return

        # merge all holidays and have 1 prior for all holidays
        h_key = 'holidays'
        self.check_reserved(h_key)

        # set the new holidays column in self.data and drop the old ones
        self.data[h_key] = self.data[self.holiday_components].apply(lambda x: 1 if x.sum() > 0 else 0, axis=1)
        self.data.drop(self.holiday_components, axis=1, inplace=True)

        # reset holidays_data
        hf = pd.concat(list(self.holidays_data.values()), axis=0)       # all holiday dates (w/ dups)
        hf.drop_duplicates(subset=['ds'], inplace=True)
        self.holidays_data[h_key] = pd.DataFrame(hf['ds'])         # all unique holiday dates {'holidays': DF with unique holiday dates}
        _ = [self.holidays_data.pop(k) for k in self.holiday_components]

        self.holiday_components = [h_key]
        with self.my_model:
            setattr(self, h_key, pm.Laplace(h_key, 0, self.holidays_prior_scale, shape=1))  # create self.holidays_all

    def add_holiday(self, name: str, holiday: pd.DataFrame):
        """
        add holiday to the model
        :param name: holiday ame
        :param holiday: holiday dates
        :return:
        """
        component, h_ds = self._holiday_error_check(name, holiday)
        self.data[component] = self.data['ds'].apply(lambda x: 1 if x in h_ds else 0)
        self.holidays_data[component] = holiday   # {name: DF with all the holiday active dates. No need to scale}

    def _holiday_error_check(self, name, holiday):
        if 'ds' not in holiday.columns:
            raise Exception('holiday DF must have a ds column')

        self.check_reserved(name)
        key = name
        if key in self.holiday_components:
            raise Exception('duplicate holiday name: ' + name)
        else:
            self.holiday_components.append(key)
        h_ds = list(holiday[holiday['holiday'] == name]['ds'].values)  # list of this holiday's dates
        return key, h_ds

    # ###########################################################
    # ###########################################################
    # regressors
    # ###########################################################
    # ###########################################################
    def add_regressors(self):
        """
        adds regressors to the model
        regressors are may not be merged as holidays and seasonalities
        :return:
        """
        if self.regressors is not None:
            _ = [self.add_regressor(r, self.regressors[['ds', r]].copy()) for r in [c for c in self.regressors.columns if c != 'ds']]
        else:
            print('no regressors provided. Skipping')
            return

    def add_regressor(self, name: str, regressor_: pd.DataFrame):
        """
        adds a single regressor to the model
        sets prior
        regressors are may not be merged as holidays and seasonalities
        :param name: regressor name
        :param regressor_: DF with ds and regressor values in a column named <name>
        :return:
        """
        regressor = self._regressor_error_checks(name, regressor_)
        component = name
        ut.ScaledData(component, regressor[name], self.scaled_dict)
        scaled_regressor = pd.DataFrame({'ds': regressor['ds'].values, component: self.scaled_dict[component].scaled_data})
        self.regressor_components.append(component)
        self.data[component] = self.data.merge(scaled_regressor, on='ds', how='left')[component].copy()
        self.regressors_data[component] = scaled_regressor.copy()   # {component: all the regressor date range scaled}

        self.reserved_names.append(component)
        with self.my_model:
            if self.positive_regressors_coefficients:
                setattr(self, component, pm.Exponential(component, self.regressors_prior_scale, shape=1))   # create self.<regressor::regressor_name>
            else:
                setattr(self, component, pm.Laplace(component, 0, self.regressors_prior_scale, shape=1))    # create self.<regressor::regressor_name>

    def _regressor_error_checks(self, name, regressor):
        self.check_reserved(name)
        if name not in regressor.columns:
            raise Exception('regressor name: ' + str(name) + ' should be a column in the regressor DF')
        if name in self.regressor_names:
            raise Exception('duplicate regressor name: ' + str(name))
        else:
            self.regressor_names.append(name)
        if regressor[name].isnull().sum() > 0:
            raise Exception('null values in regressor ' + str(name))

        if 'ds' not in regressor.columns:
            raise Exception('Time variable should be called `ds` in the `regressor` dataframe')
        regressor['ds'] = pd.to_datetime(regressor['ds'].values)
        if regressor['ds'].isnull().sum() > 0:
            raise Exception('null time values in regressor ' + str(name))
        if regressor['ds'].max() <= self.max_ds or regressor['ds'].min() > self.min_ds:
            raise Exception('invalid date range in regressor ' + str(name))
        return regressor.copy()

    # ###########################################################
    # ###########################################################
    # fit
    # ###########################################################
    # ###########################################################

    @staticmethod
    def _set_growth(ts, cpt):
        A = (ts[:, None] > cpt) * 1
        return A, ts

    def _fit_growth(self):
        """
        only called when grwoth is not None
        piecewise linear trend with changepoints
        k ~ N(0, gr)           base growth
        delta ~ Laplace(0, cp) incremental growth changes
        m ~ N(0, s2)           initial intercept (at time 0)
        g | m, k, delta = m + (k + A * delta) t + A * (-s * delta)
        :return g
        """
        print('fit::adding growth model')

        if self.growth is None:
            raise Exception('_fit_growth(): only when growth not None')
            # eps = 1.0e-8
            # self.growth_prior_scale = eps        # scaled_y between 0 and 1
            # self.changepoints_prior_scale = eps  # scaled_y between 0 and 1

        with self.my_model:
            ts = self.data['t'].values
            cpt = np.linspace(start=0, stop=self.changepoint_range * np.max(ts), num=self.n_changepoints + 1)[1:]
            A, ts = self._set_growth(ts, cpt)

            # create self.k = pm.Normal('k', 0, self.growth_prior_scale)
            self.check_reserved('k')
            setattr(self, 'k', pm.Normal('k', 0, self.growth_prior_scale, shape=1))
            self.growth_components.append('k')

            # create self.delta = pm.Laplace('delta', 0, self.changepoints_prior_scale, shape=self.n_changepoints)
            self.check_reserved('delta')
            setattr(self, 'delta', pm.Laplace('delta', 0, self.changepoints_prior_scale, shape=self.n_changepoints))
            self.growth_components.append('delta')

            # create self.m
            self.check_reserved('m')
            setattr(self, 'm', pm.Normal('m', 0, self.offset_prior_scale, shape=1))     # self.m = pm.Normal('m', 0, self.offset_prior_scale)
            self.growth_components.append('m')

            if self.growth is None:
                trend = pm.Deterministic('trend', (self.k + tt.dot(A, self.delta)) * ts + self.m)
            else:
                gamma = -cpt * self.delta
                trend = pm.Deterministic('trend', (self.k + tt.dot(A, self.delta)) * ts + (self.m + tt.dot(A, gamma)))
            return trend

    def _fit_func(self, name, func_components):
        if len(func_components) > 0:
            print('fit::adding ' + name + ' components: ' + str(func_components))
        else:
            raise Exception('fit:: ' + name + ' nothing to fit')

        y = None
        for component in func_components:
            cols = self.get_cols(component)
            arr = self.data[cols].values
            prior = getattr(self, component)  # priors for betas already defined!
            tprod = tt.dot(arr, prior)
            self._add_component(component, self.set_ctype(name))

            if y is None:
                y = tprod
            else:
                y += tprod

        if y is None:
            return None

        y_det = pm.Deterministic(name, y)
        if name in self.additive:
            self.additive.append(y_det)

        if name in self.multiplicative:
            self.multiplicative.append(y_det)
        return y_det

    def _add_component(self, cname, ctype):
        if ctype == 'add':
            self.additive_components.append(cname)
        elif ctype == 'mult':
            self.multiplicative_components.append(cname)
        else:
            self.additive_components.append(cname)
            self.multiplicative_components.append(cname)

    def set_ctype(self, name):
        v_add = True if name in self.additive else False
        v_mult = True if name in self.multiplicative else False
        if v_add is True and v_mult is True:
            return 'all'
        elif v_add is True and v_mult is False:
            return 'add'
        elif v_add is False and v_mult is True:
            return 'mult'
        else:
            return None

    def _check_add_mult(self):
        # model components add/mult check
        # each must be at least in either additive or multiplicative
        all_components = ['regressor', 'seasonality', 'holiday']
        if len(self.multiplicative) == 0 and len(self.additive) == 0:
            print('defaulting to additive model: addtive: ' + str(all_components))
            self.additive = all_components
            self.multiplicative = list()
        else:      # check all components are assigned
            mdl_components = self.additive + self.multiplicative
            if set(mdl_components) != set(all_components):
                raise Exception('incorrect model specification::' + str(mdl_components) + ' are set but the model components are ' + str(all_components))

    def _txt_rm(self):
        # remove txt names from additive and multiplicative
        all_components = ['regressor', 'seasonality', 'holiday']
        for c in all_components:
            try:
                self.additive.remove(c)
            except ValueError:
                pass
            try:
                self.multiplicative.remove(c)
            except ValueError:
                pass

    def model_formula(self):
        # check components
        self._check_add_mult()

        # fitting order matters!!!
        _ = self._fit_func('regressor', self.regressor_components)
        _ = self._fit_func('holiday', self.holiday_components)
        scaled_trend = None if self.growth is None else self._fit_growth()
        _ = self._fit_func('seasonality', self.seasonality_components)
        self._txt_rm()  # rm the txt labels now that we have the deterministic values

        print('additive components: ' + str(self.additive_components))
        print('multiplicative components: ' + str(self.multiplicative_components))

        scaled_yadd = np.sum([c for c in self.additive])
        scaled_ymult = np.prod([c for c in self.multiplicative])
        if self.growth is not None:
            scaled_yadd += scaled_trend
            scaled_ymult *= scaled_trend

        if len(self.additive) > 0 and len(self.multiplicative) > 0:
            self.check_reserved('eta_sum')
            self.check_reserved('eta_prod')
            setattr(self, 'eta_sum', pm.Uniform('eta_sum', lower=0, upper=1, shape=1))
            setattr(self, 'eta_prod', pm.Deterministic('eta_prod', 1 - self.eta_sum))
            self.mdl_components.extend(['eta_sum', 'eta_prod'])
            return tt.dot(scaled_yadd[:, None], self.eta_sum) + tt.dot(scaled_ymult[:, None], self.eta_prod)
        elif len(self.additive) > 0 and len(self.multiplicative) == 0:
            return scaled_yadd
        elif len(self.additive) == 0 and len(self.multiplicative) > 0:
            return scaled_ymult
        else:
            return None

    def _prepare_fit(self):
        """
        set the last priors and overall model
        :return:
        """
        with self.my_model:
            y_mdl = self.model_formula()  # build the model formula (additve/multiplicative)

            # construct the final model
            self.check_reserved('sigma')
            self.mdl_components.append('sigma')
            setattr(self, 'sigma', pm.HalfCauchy('sigma', beta=self.sigma_tau, testval=1))           # self.sigma = pm.HalfCauchy('sigma', beta=self.sigma_tau, testval=1)
            pm.Normal('y_%s' % self.name, mu=y_mdl, sd=self.sigma, observed=self.data['scaled_y'])   # do not append y_mdl to mdl_components

    def fit(self, draws: int = 500, tune: int = 500):
        """
        Fit the model
        :param draws: number of mcmc samples
        :param tune: number of mcmc tuning samples
        :return:
        """
        if self.fitted is True:
            raise Exception('can only fit the model once')
        else:
            self.fitted = True

        self._prepare_fit()
        with self.my_model:
            if self.mcmc is False:
                self.map = pm.find_MAP()
            else:
                step_method = pm.NUTS(target_accept=0.90, max_treedepth=15)
                self.trace = pm.sample(draws, chains=None, step=step_method, tune=tune)
                # self.trace = pm.sample(draws)

    # ###########################################################
    # ###########################################################
    # diagnostics
    # ###########################################################
    # ###########################################################

    def fit_diagnostics(self, p=0.025, component_plots=True):
        """

        :param p:
        :param component_plots:
        :return:
        """
        if self.mcmc is False:
            print('Running MAP. no mcmc diagnostics needed')
            return
        print('diagnostics')
        with self.my_model:
            print('\ttrace summary ... ')
            smry_df = pm.summary(self.trace)

            print('\ttrace plots ... ')
            var_names = self.seasonality_components + self.regressor_components + self.holiday_components + self.growth_components + self.mdl_components
            pm.traceplot(self.trace, var_names=var_names)

        if component_plots:
            print('\tcomponent plots ... ')
            self._plot_components(p, self.data)

        return smry_df

    def _scaled_trend_eval(self, ts, cpt, deltas, k, m):
        if self.growth is None:
            raise Exception('_scaled_trend_eval(): only when growth not None')
        intercept = m
        if self.growth is None:
            return np.repeat(intercept, len(ts), axis=0)
        else:
            A, ts = self._set_growth(ts, cpt)
            gamma = (-cpt * deltas).T
            slope = (k + np.dot(A, deltas.T))
            intercept = m + np.dot(A, gamma)
            t = ts[:, None] if slope.ndim == 2 else ts
            return slope * t + intercept

    def _plot_components(self, p, a_df):
        # determine the posterior by evaluating all the values in the trace. Need to de-scale
        model_key = 'y_mdl'

        # trend
        if self.growth is not None:
            ts = a_df['t'].values
            cpt = np.linspace(start=0, stop=self.changepoint_range * np.max(ts), num=self.n_changepoints + 1)[1:]
            self.fit_data['trend'] = self._scaled_trend_eval(ts, cpt, self.trace['delta'], self.trace['k'].T, self.trace['m'].T)   # scaled

        # regressors, hols and seasonalities
        for component in self.regressor_components + self.holiday_components + self.seasonality_components + self.mdl_components:
            cols = self.get_cols(component)
            trace = self.trace[component]
            self.fit_data[component] = np.dot(self.data[cols].values, trace.T) if len(cols) > 0 else trace.T

        # fit_data components: trend, holidays (holiday_components), regressor_components, seasonality_components
        scaled_model = self._scaled_yfit()
        self.fit_data[model_key] = self.scaled_dict['y'].reset_vals(scaled_data=scaled_model)

        # plot in sample model
        df = self.data[['ds', 'y', 't']].copy()
        for component, vals in self.fit_data.items():
            if vals.ndim == 1:
                vals = vals[None, :]
            avg = vals.mean(axis=1)
            df[component] = avg if len(avg) == len(df) else avg[0]
            # if len(avg) == len(df):  # exclude single values
            quantiles = np.nanquantile(vals, [p, 1 - p], axis=1)
            df[component + '_upr'] = quantiles[1, :] if len(avg) == len(df) else quantiles[1][0]
            df[component + '_lwr'] = quantiles[0, :] if len(avg) == len(df) else quantiles[0][0]
        nfigs = len(self.fit_data.keys())  # figsize=(width, height)
        p_plt.plot(self, ax=None, plot_cap=True, xlabel='ds', ylabel='y', figsize=(10, 6), df=df)  # plots linearized 'y' in the logistic case
        p_plt.plot_components(self, plot_cap=True, weekly_start=0, yearly_start=0, figsize=(10, 2 * nfigs), df=df, components=self.fit_data.keys())

    def get_cols(self, a_component):
        # gets the col names associated with a_component when a_component itself is not a column, eg weekly_5 when a_component = weekly
        cols = [c for c in self.data.columns if a_component in c] if a_component not in self.data.columns else [a_component]
        return cols

    def _scaled_yfit(self):
        if len(self.additive) > 0 and len(self.multiplicative) == 0:
            eta_sum = 1.0
        elif len(self.additive) == 0 and len(self.multiplicative) > 0:
            eta_sum = 0.0
        elif len(self.additive) > 0 and len(self.multiplicative) > 0:
            eta_sum = self.trace['eta_sum'].mean()
        else:
            raise Exception('model error')
        eta_prod = 1.0 - eta_sum

        _sum = np.sum(np.array([v for k, v in self.fit_data.items() if k in self.additive_components]), axis=0)
        _prod = np.prod(np.array([v for k, v in self.fit_data.items() if k in self.multiplicative_components]), axis=0)

        return eta_sum * _sum + eta_prod * _prod

    # ###########################################################
    # ###########################################################
    # predict
    # ###########################################################
    # ###########################################################

    def predict(self, mid, forecasting_periods: int = None, alpha: float = 0.20, n_samples: int = 1000):
        """
        issue a prediction
        :param mid: mean or median
        :param forecasting_periods: number of future points in self.freq scale
        :param alpha: width of credible interval
        :param n_samples: number of sample to draw from the ppc
        :return:
        """
        print('forecasting ...')
        try:
            if int(forecasting_periods) >= 0:
                self.forecasting_periods = int(forecasting_periods)
                print('forecasting periods: ' + str(self.forecasting_periods))
        except TypeError:
            raise Exception('invalid forecasting periods: ' + str(self.forecasting_periods))

        if mid not in ['mean', 'median']:
            raise Exception(str(mid) + ' is invalid')

        future_df = pd.DataFrame()
        future_ds = pd.date_range(start=self.cutoff_date + pd.to_timedelta(1, unit=self.freq), periods=self.forecasting_periods, freq=self.freq)
        future_df['ds'] = np.concatenate([self.data['ds'], future_ds])
        future_df['t'] = self.set_ts(pd.Series(future_df['ds']), min_ds=self.min_ds, max_ds=self.max_ds)   # future_df['t'] > 1 on forecast range

        self.fcast_df = pd.DataFrame()
        self.fcast_df['ds'] = future_df['ds'].copy()
        self.fcast_df['t'] = future_df['t'].copy()
        self.fcast_df['scaled_y'] = list(self.scaled_dict['y'].scaled_data) + [np.nan] * self.forecasting_periods
        descaled_y = list(self.scaled_dict['y'].data_in)
        if self.growth != 'logistic':
            self.fcast_df['y'] = descaled_y + [np.nan] * self.forecasting_periods
        else:
            # yl = self.linearizer.inverse_transform(descaled_y, y_var=np.var(descaled_y))
            self.fcast_df['y'] = list(self.linearizer.data_in) + [np.nan] * self.forecasting_periods

        if self.mcmc is True:
            with self.model:
                var_names = self.seasonality_components + self.regressor_components + self.growth_components + self.holiday_components + self.mdl_components
                self.spp = pm.sample_posterior_predictive(self.trace, samples=n_samples, var_names=var_names, progressbar=False)
        else:
            n_samples = 1
            self.spp = dict()
            for k, v in self.map.items():
                if v.ndim == 0:
                    self.spp[k] = float(str(v))
                else:
                    self.spp[k] = v

        d_smpls = self.model_spp(n_samples, future_df)
        _ = [self.get_bands(v_arr, k, alpha=alpha, mid=mid) for k, v_arr in d_smpls.items() if k not in ['ds', 't']]
        if self.growth == 'logistic':
            self.fcast_df['floor'] = self.floor
            self.fcast_df['ceiling'] = self.ceiling
        return self.fcast_df

    def model_spp(self, n_samples, future_df):
        d_list = [self.sample_mdl(i, future_df) for i in range(n_samples)]
        d_join = defaultdict(list)
        for d in d_list:
            for k, v in d.items():
                d_join[k].append(v)
        return d_join

    def sample_mdl(self, i_, future_df):
        holidays = self._holidays(i_, future_df)                    # never scaled
        scaled_seasonality = self._scaled_seasonality(i_, future_df)
        scaled_regressors = self._scaled_regressors(i_, future_df)
        sigma = self.spp['sigma'][i_] if self.mcmc else self.spp['sigma']  # float(str(self.spp['sigma']))
        scaled_noise = np.random.normal(0, sigma, len(future_df))

        d_out = {'t': future_df['t'].values, 'ds': future_df['ds'].values,
                 'holidays': holidays,                       # never scaled
                 'scaled_noise': scaled_noise,               # scaled
                 }

        if self.growth is not None:
            scaled_trend = self._scaled_trend(i_, future_df)
            trend = scaled_trend * self.scaled_dict['y'].scale
            d_out['scaled_trend'] = scaled_trend  # scaled
            d_out['trend'] = trend                # de-scaled
        else:
            scaled_trend = None

        for k, v in scaled_seasonality.items():
            d_out[k] = v  # scaled
        for k, v in scaled_regressors.items():
            d_out[k] = v  # scaled

        # model formula
        scaled_yhat = self._scaled_yhat(i_, scaled_trend, scaled_seasonality, holidays, scaled_regressors, scaled_noise)
        d_out['scaled_yhat'] = scaled_yhat
        yhat = self.scaled_dict['y'].reset_vals(scaled_data=scaled_yhat)
        y_var = self.spp['sigma'][i_] ** 2 if self.mcmc is True else self.spp['sigma'] ** 2
        d_out['yhat'] = yhat if self.growth != 'logistic' else self.linearizer.inverse_transform(yhat, y_var=y_var)
        return d_out

    def _scaled_trend(self, i_, future_df):
        # only called when growth not None
        if self.growth is None:
            raise Exception('_scaled_trend(): only when growth not None')
        t = future_df['t'].values
        k = self.spp['k'][i_]
        m = self.spp['m'][i_]
        delta_arr = self.spp['delta'][i_, :] if self.mcmc else self.spp['delta']
        new_changepoints_t, new_deltas = self.new_cpd(np.max(t), delta_arr)
        changepoint_t = np.concatenate((self.changepoints_t, new_changepoints_t))
        deltas = np.concatenate((delta_arr, new_deltas))
        return self._scaled_trend_eval(t, changepoint_t, deltas, k, m)

    def new_cpd(self, t_max, delta_arr):    # New changepoints from a Poisson process with rate S on [1, T]
        if t_max > 1:
            S = self.changepoint_range * len(self.changepoints_t)
            n_changes = 0 if self.growth is None else np.random.poisson(S * (t_max - 1))
        else:
            n_changes = 0

        if n_changes > 0:
            new_changepoints_t = 1 + np.random.rand(n_changes) * (t_max - 1)
            new_changepoints_t.sort()
            lambda_ = np.mean(np.abs(delta_arr)) + 1e-8  # Get the empirical scale of the deltas, plus epsilon to avoid NaNs.
            new_deltas = np.random.laplace(0, lambda_, n_changes)  # Sample deltas
            self.new_cpt_cnt.append(n_changes)
            self.delta_arr.append(np.mean(new_deltas))
        else:
            new_changepoints_t, new_deltas = list(), list()
        return new_changepoints_t, new_deltas
    
    def new_cpt_smry(self, t_max):
        # used for plots
        if t_max <= 1.0:
            return list(), list()
        else:
            n_changes = int(np.round(np.mean(self.new_cpt_cnt), 0))
            lambda_ = np.mean(np.mean(np.abs(self.spp['delta']), axis=0))
            deltas = np.random.laplace(0, lambda_, n_changes)
            new_changepoints_t = 1 + np.random.rand(n_changes) * (t_max - 1)
            new_changepoints_t.sort()
            cs = pd.DataFrame({'t': new_changepoints_t})
            ff = self.fcast_df[['t', 'ds']].copy()
            ff.sort_values(by='t', inplace=True)
            m = pd.merge_asof(cs, ff, on='t', direction='nearest')
            return list(m['ds'].values), list(deltas)   # time of change, size of change

    def _fourier_seasonality(self, a_df):
        fcast_ts_scale = self.set_ts_scale(a_df['ds'].max(), a_df['ds'].min())
        for k, po in self.seasonality_info.items():  # k = component: seasonality_<name>, po = value: (period, order)
            n = '_'.join(k.split('_')[:2])           # seasonality_name
            period, order = po
            fs = self.fourier_series(a_df['t'].values, period / fcast_ts_scale, order)
            s_df = pd.DataFrame(fs, columns=[n + '_' + str(i) for i in range(2 * order)])
            df_cols = list()
            for idx in range(2 * order):
                skey = n + '_' + str(idx)
                a_df[skey] = s_df[skey].values
                df_cols.append(skey)

            if self.merge_seasonalities:
                a_df[n] = a_df[df_cols].sum(axis=1)
                a_df.drop(df_cols, axis=1, inplace=True)
        self.seasons_df = a_df.copy()

    def _scaled_seasonality(self, i_, a_df):
        if self.seasons_df is None:
            self._fourier_seasonality(a_df)
        scaled_s = dict()
        for k in self.seasonality_components:
            cols = self.get_cols(k)
            z = np.reshape(self.seasons_df[cols].values, (len(a_df), len(cols)))
            beta = self.spp[k][i_, :] if self.mcmc else self.spp[k]
            s_scaled = np.dot(z, beta.T)
            scaled_s[k] = s_scaled
        return scaled_s

    def _holidays(self, i_, a_df):
        hols_ = list()
        # there is only one holiday component called 'holidays'
        for k in self.holiday_components:  # assume no dates in holidays_data, no hols in that date
            dates = [x for x in self.holidays_data['holidays']['ds']]
            a_df[k] = a_df['ds'].apply(lambda x: 1 if x in dates else 0)
            z = np.reshape(a_df[k].values, (len(a_df), 1))
            beta = self.spp[k][i_, :] if self.mcmc else self.spp[k]
            hols_.append(np.dot(z, beta.T))
        hols_ = np.array(hols_)
        return np.sum(hols_, axis=0)

    def _scaled_regressors(self, i_, a_df):
        scaled_reg_ = dict()
        for k, rdf in self.regressors_data.items():
            a_df = a_df.merge(rdf, on='ds', how='left')
            if a_df[k].isnull().sum() == 0:
                z = np.reshape(a_df[k].values, (len(a_df), 1))
                beta = self.spp[k][i_, :] if self.mcmc else self.spp[k]
                reg_scaled = np.dot(z, beta.T)
                scaled_reg_[k] = reg_scaled
            else:
                raise Exception(k + ' regressor has an invalid date range')
        return scaled_reg_

    def _scaled_yhat(self, i_, scaled_trend, scaled_seasonality, holidays, scaled_regressors, scaled_noise):
        if len(self.additive) > 0 and len(self.multiplicative) == 0:
            eta_sum = 1.0
        elif len(self.additive) == 0 and len(self.multiplicative) > 0:
            eta_sum = 0.0
        elif len(self.additive) > 0 and len(self.multiplicative) > 0:
            eta_sum = self.spp['eta_sum'][i_]
        else:
            raise Exception('model error')
        eta_prod = 1.0 - eta_sum

        seasonality_sum = np.sum(np.array([v for k, v in scaled_seasonality.items() if k in self.additive_components]), axis=0)
        seasonality_prod = np.prod(np.array([v for k, v in scaled_seasonality.items() if k in self.multiplicative_components]), axis=0)

        regressor_sum = np.sum(np.array([v for k, v in scaled_regressors.items() if k in self.additive_components]), axis=0)
        regressor_prod = np.prod(np.array([v for k, v in scaled_regressors.items() if k in self.multiplicative_components]), axis=0)

        holiday_sum = holidays if 'holidays' in self.additive_components else 0
        holiday_prod = holidays if 'holidays' in self.multiplicative_components else 1

        if self.growth is None:
            yhat_sum = eta_sum * (seasonality_sum + holiday_sum + regressor_sum)
            yhat_prod = eta_prod * (seasonality_prod * holiday_prod * regressor_prod)
        else:
            yhat_sum = eta_sum * (scaled_trend + seasonality_sum + holiday_sum + regressor_sum)
            yhat_prod = eta_prod * (scaled_trend * seasonality_prod * holiday_prod * regressor_prod)

        return yhat_sum + yhat_prod + scaled_noise

    def get_bands(self, mcy, col, alpha=0.2, mid=0.5):   # alpha < mid
        if np.shape(mcy) == ():
            return
        else:
            self.fcast_df[col] = np.nanmean(mcy, axis=0) if mid == 'mean' else np.nanmedian(mcy, axis=0)
            self.fcast_df[col + '_upr'] = np.nanquantile(mcy, (1.0 - alpha / 2.0), axis=0)
            self.fcast_df[col + '_lwr'] = np.nanquantile(mcy, alpha / 2.0, axis=0)

    def plot_model(self, df=None):
        if self.fcast_df is None:
            return
        p_plt.plot(self, ax=None, plot_cap=True, xlabel='ds', ylabel='y', figsize=(10, 6), df=df)

        components = list()
        if self.growth is not None:
            components.append('trend')
        if self.holiday_components is not None:
            components.append('holidays')
        if self.seasonality_components is not None:
            components.extend(self.seasonality_components)
        if self.regressor_components is not None:
            components.extend(self.regressor_components)

        p_plt.plot_components(self, plot_cap=True, weekly_start=0, yearly_start=0, figsize=None, df=df, components=components)



