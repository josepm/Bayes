"""
PyMC3 forecasts
"""

if __name__ == '__main__':

    import os
    import numpy as np
    import pandas as pd
    from Bayes.Prophet.model import PyMCProphet
    from Bayes.Prophet import utilities as ut
    import pickle

    # ########################################################################
    # ########################################################################
    # ########################################################################
    # ########################################################################
    # read data
    freq = 'D'                 # time series frequency
    df = pd.read_csv('~/data.csv')
    df['ds'] = pd.to_datetime(df['ds'].values)

    # holidays: must run up to or past fcast date
    # prepare holidays DF
    # a DF with cols holiday (holiday name) and ds (holiday dates)
    # windows around holiday must be in the holiday date for each holiday,
    # eg if we want to mark the day before Christmas as a holiday,
    # include 12/24 in the list of dates for the Christmas holiday DF
    playoffs = pd.DataFrame({
        'holiday': 'playoff',
        'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                              '2010-01-24', '2010-02-07', '2011-01-08',
                              '2013-01-12', '2014-01-12', '2014-01-19',
                              '2014-02-02', '2015-01-11', '2016-01-17',
                              '2016-01-24', '2016-02-07'])
    })
    superbowls = pd.DataFrame({
        'holiday': 'superbowl',
        'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07'])
    })
    holidays_df = pd.concat((playoffs, superbowls))
    holidays = holidays_df.copy()

    # regressor: must run up to or past fcast date
    # DF with ds, regressor1, regressor2, ... which contain the values of each regressor at ds
    regressors_df = pd.DataFrame()
    dr = pd.date_range(df['ds'].min(), df['ds'].max(), freq=freq)
    regressors_df['ds'] = dr
    regressors_df['nfl_sun'] = regressors_df['ds'].apply(lambda x: 1 if x.weekday() == 6 and (x.month > 8 or x.month < 2) else 0)
    regressors_df['nfl_mon'] = regressors_df['ds'].apply(lambda x: 1 if x.weekday() == 0 and (x.month > 8 or x.month < 2) else 0)
    regressors = regressors_df.copy()

    # seasonalities
    # list of tuples (name, periods, order)
    # period in the time series frequency (e.g. days, ...)
    seasonalities = [('monthly', 365.25/12, 5), ('weekly', 7, 2)]

    # ########################################################################
    # ########################################################################
    # ########################################################################
    # ########################################################################

    # ########################################################################
    # final set up
    forecast_periods = 365
    ds_max = df['ds'].max()
    cutoff_date = ds_max - pd.to_timedelta(forecast_periods, unit=freq)
    df_in = df[df['ds'] >= '2014-01-01'].copy()  # fewer data points
    growth = 'linear'
    floor, ceiling = None, None
    if growth == 'logistic':
        df_in['y'] = (df_in['y'] - df_in['y'].min()) / (df_in['y'].max() - df_in['y'].min())
        ceiling, floor = df_in['y'].max(), df_in['y'].min()

    df_future = df_in[df_in['ds'] > cutoff_date]    # Data to be forecasted
    data = df_in[df_in['ds'] <= cutoff_date].copy()    # training data

    # fcast object
    fcast_m = PyMCProphet(data,
                          name='test',
                          freq=freq,
                          cutoff_date=str(cutoff_date.date()),
                          growth=growth,                                   # linear, logistic, None
                          floor=floor,
                          ceiling=ceiling,
                          seasonalities=seasonalities,
                          regressors=regressors,
                          holidays=holidays,                                 # Note: all holidays are merged (single Beta for all holidays)
                          n_changepoints=25,
                          merge_seasonalities=False,
                          multiplicative=['regressor'],                      # only at whole component level. Cannot do weekly additive and monthly multiplicative for example
                          additive=['regressor', 'seasonality', 'holiday'],  # only at whole component level. Cannot do weekly additive and monthly multiplicative for example
                          mcmc=True,
                          )

    # Fit the model (using NUTS)
    fcast_m.fit()

    # diagnostics
    diag_smry = fcast_m.fit_diagnostics(p=0.025, component_plots=True)

    # predict
    ddf = fcast_m.predict('mean', forecast_periods, alpha=0.2)

    # final plots
    # include actuals (but first must fix any data gaps to align ds) to be able to compute fcast errors
    dt = pd.DataFrame({'ds': pd.date_range(start=df_in['ds'].min(), end=df_in['ds'].max(), freq=freq)})
    rf = dt.merge(df_in, on='ds', how='left')
    ddf['y'] = ut.data_gaps(rf['y'].copy(), fill_data=True)
    fcast_m.plot_model(df=ddf)

    # forecast error: we added the future values to fcast_df!
    # when y values close to 0's mape may blow out. This may happen with logistic with 0 floor
    # mase is better for logistic
    # h_mase computes the avg mase over the forecast horizon
    ferr = ddf[ddf['ds'] > cutoff_date].copy()
    mape_err = ut.mape(ferr[['y', 'yhat']])
    periods = [int(np.round(s[1], 0)) for s in seasonalities]
    mase_err = ut.h_mase(ddf, forecast_periods, cutoff_date, freq=freq, periods=periods, t_col='ds', y_col='y', yhat_col='yhat')
    print('PyMC forecast performance::MAPE: ' + str(np.round(mape_err, 2)) + '% MASE: ' + str(np.round(mase_err, 2)))

    # save fcast object
    # fout = os.path.expanduser('~/my_data/fcast_obj.pkl')
    # with open(fout, 'wb') as fp:
    #     pickle.dump(fcast_m, fp, pickle.HIGHEST_PROTOCOL)

    # read fcast obj
    # with open(fout, 'rb') as fp:
    #     fcast_obj = pickle.load(fp)

    # FB Prophet
    from fbprophet import Prophet
    dt = pd.DataFrame({'ds': pd.date_range(start=df_in['ds'].min(), end=df_in['ds'].max(), freq=freq)})
    rf = dt.merge(df_in, on='ds', how='left')
    data = rf[rf['ds'] <= cutoff_date].copy()  # training data
    fbp_data = data.merge(regressors, on='ds', how='left')
    if growth == 'logistic':
        fbp_data['cap'] = ceiling
        fbp_data['floor'] = floor
    m = Prophet(holidays=holidays, weekly_seasonality=False, yearly_seasonality=False, growth=growth, mcmc_samples=1000)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name='weekly', period=7, fourier_order=2)
    m.add_regressor('nfl_sun')
    m.add_regressor('nfl_mon')
    m.fit(fbp_data)
    future = m.make_future_dataframe(periods=forecast_periods)
    future['nfl_sun'] = regressors['nfl_sun']
    future['nfl_mon'] = regressors['nfl_mon']
    if growth == 'logistic':
        future['cap'] = ceiling
        future['floor'] = floor
    forecast = m.predict(future)

    # forecast errors
    ff = df_future[['ds', 'y']].merge(forecast[forecast['ds'] > cutoff_date][['ds', 'yhat']], on='ds', how='inner')
    mape_err = ut.mape(ff[['y', 'yhat']])

    forecast['y'] = rf[['ds', 'y']].merge(forecast[['ds']], on='ds', how='inner')['y']
    periods = [30, 7]
    mase_err = ut.h_mase(forecast[['ds', 'y', 'yhat']], forecast_periods, cutoff_date, freq=freq, periods=periods, t_col='ds', y_col='y', yhat_col='yhat')
    print('FB Prophet forecast performance::MAPE: ' + str(np.round(mape_err, 2)) + '% MASE: ' + str(np.round(mase_err, 2)))




