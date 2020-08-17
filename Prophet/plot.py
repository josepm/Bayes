import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('fbprophet.plot')
try:
    from matplotlib import pyplot as plt
    from matplotlib.dates import (
        MonthLocator,
        num2date,
        AutoDateLocator,
        AutoDateFormatter,
    )
    from matplotlib.ticker import FuncFormatter

    from pandas.plotting import deregister_matplotlib_converters
    deregister_matplotlib_converters()
except ImportError:
    logger.error('Importing matplotlib failed. Plotting will not work.')

try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ImportError:
    logger.error('Importing plotly failed. Interactive plots will not work.')


def plot(m, ax=None, plot_cap=True, xlabel='ds', ylabel='y', figsize=(10, 6), df=None):
    """Plot the Prophet forecast.
    Parameters
    ----------
    m: Prophet model.
    df: DF
    ax: Optional matplotlib axes on which to plot.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    xlabel: Optional label name on X-axis
    ylabel: Optional label name on Y-axis
    figsize: Optional tuple width, height in inches.
    Returns
    -------
    A matplotlib figure.
    """
    print('plotting overall model')
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if df is None:
        fcst = m.fcast_df.copy()
        fcst_t = fcst['ds']
        history = m.data[['t', 'ds', 'y']].copy()
    else:
        fcst = df.copy()
        fcst_t = fcst['ds']
        history = df[['t', 'ds', 'y']].copy()

    if 'yhat' in fcst.columns:
        ax.plot(fcst_t, fcst['yhat'], ls='-', marker='*', c='#0072B2')
        if 'ceiling' in fcst and plot_cap:
            ax.plot(fcst_t, fcst['ceiling'], ls='--', c='k')
        if 'floor' in fcst and plot_cap:
            ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
        if 'yhat_lwr' in fcst.columns and 'yhat_upr' in fcst.columns:
            ax.fill_between(fcst_t, fcst['yhat_lwr'], fcst['yhat_upr'], color='#0072B2', alpha=0.2)

    # actuals
    ax.plot(history['ds'], history['y'], 'k.')

    # change points
    cp_list = m.actual_cpt(t_max=fcst['t'].max(), thres=0.25)
    for idx, change_point in enumerate(cp_list):
        plt.axvline(change_point, color='C2', lw=1, ls='dashed')
    plt.axvline(m.max_ds, color='C3', lw=1, ls='dotted')

    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title( ylabel + ' plot')
    fig.tight_layout()
    return fig


def plot_components(m, plot_cap=True, weekly_start=0, yearly_start=0, figsize=None, df=None, components=None):
    """Plot the Prophet forecast components.
    Will plot whichever are available of: trend, holidays, weekly
    seasonality, yearly seasonality, and additive and multiplicative extra
    regressors.
    Parameters
    ----------
    m: Prophet model.
    df: DF
    components: list of components to plot
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    weekly_start: Optional int specifying the start day of the weekly
        seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
        by 1 day to Monday, and so on.
    yearly_start: Optional int specifying the start day of the yearly
        seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts
        by 1 day to Jan 2, and so on.
    figsize: Optional tuple width, height in inches.
    Returns
    -------
    A matplotlib figure.
    """

    if components is None:
        components = ['trend']
        if m.holiday_components is not None:
            components.append('holidays')
        if m.seasonality_components is not None:
            components.extend(m.seasonality_components)
            # components.append('scaled_seasonality')
        if m.regressor_components is not None:
            # components.append('scaled_regressors')
            components.extend(m.regressor_components)
    npanel = len(components)
    print('plotting components: ' + str(components))

    # Identify components to be plotted
    figsize = figsize if figsize else (8, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor='w', figsize=figsize)

    if npanel == 1:
        axes = [axes]

    # multiplicative_axes = []

    for ax, pname in zip(axes, components):
        plot_name = pname[7:] if 'scaled_' in pname else pname
        if plot_name == 'trend':
            plot_forecast_component(m=m, name='trend', ax=ax, plot_cap=plot_cap, df=df, figsize=figsize)
        elif plot_name in m.seasonality_components:
            if plot_name == 'weekly' or m.seasonality_info[plot_name][0] == 7:
                plot_weekly(m=m, name=plot_name, ax=ax, weekly_start=weekly_start, df=df, figsize=figsize)
            elif plot_name == 'yearly' or m.seasonality_info[plot_name][0] == 365.25:
                plot_yearly(m=m, name=plot_name, ax=ax, yearly_start=yearly_start, df=df, figsize=figsize)
            else:
                plot_seasonality(m=m, name=plot_name, ax=ax, df=df, figsize=figsize)
        else:
            plot_forecast_component(m=m, name=plot_name, ax=ax, plot_cap=False, df=df, figsize=figsize)
        # if plot_name in m.component_modes['multiplicative']:
        #     multiplicative_axes.append(ax)

    fig.tight_layout()
    # Reset multiplicative axes labels after tight_layout adjustment
    # for ax in multiplicative_axes:
    #     ax = set_y_as_percent(ax)
    return fig


def plot_forecast_component(m, name, ax=None, plot_cap=False, figsize=(10, 6), df=None):
    """Plot a particular component of the forecast.
    Parameters
    ----------
    m: Prophet model.
    df: DF
    name: Name of the component to plot.
    ax: Optional matplotlib Axes to plot on.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    figsize: Optional tuple width, height in inches.
    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)

    if df is None:
        fcst = m.fcast_df.copy()
        fcst_t = fcst['ds']
    else:
        fcst = df.copy()
        fcst_t = fcst['ds']

    artists += ax.plot(fcst_t, fcst[name], ls='-', marker='*', c='#0072B2')
    # if 'ceiling' in fcst and plot_cap:
    #     artists += ax.plot(fcst_t, fcst['ceiling'], ls='--', c='k')
    # if 'floor' in fcst and plot_cap:
    #     ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    if name + '_lwr' in fcst.columns and name + '_upr' in fcst.columns:
        artists += [ax.fill_between(fcst_t, fcst[name + '_lwr'], fcst[name + '_upr'], color='#0072B2', alpha=0.2)]

    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('ds')
    ax.set_ylabel(name)
    # if name in m.component_modes['multiplicative']:
    #     ax = set_y_as_percent(ax)
    return artists


def plot_weekly(m, ax=None, weekly_start=0, figsize=(10, 6), name='weekly', df=None):
    """Plot the weekly component of the forecast.
    Parameters
    ----------
    m: Prophet model.
    df: DF
    ax: Optional matplotlib Axes to plot on. One will be created if this
        is not provided.
    weekly_start: Optional int specifying the start day of the weekly
        seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
        by 1 day to Monday, and so on.
    figsize: Optional tuple width, height in inches.
    name: Name of seasonality component if changed from default 'weekly'.
    Returns
    -------
    a list of matplotlib artists
    """
    if df is None:
        fcst = m.fcast_df.copy()
    else:
        fcst = df.copy()

    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)

    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days = (pd.date_range(start=m.min_ds, periods=7) + pd.Timedelta(days=weekly_start))

    if name + '_lwr' in fcst.columns and name + '_upr' in fcst.columns:
        h_df = fcst[fcst['ds'].isin(days)][[name, name + '_upr', name + '_lwr']].copy()
    else:
        h_df = fcst[fcst['ds'].isin(days)][[name]].copy()
    days = days.day_name()
    artists += ax.plot(range(len(days)), h_df[name], ls='-', marker='*', c='#0072B2')
    if name + '_lwr' in fcst.columns and name + '_upr' in fcst.columns:
        artists += [ax.fill_between(range(len(days)), h_df[name + '_lwr'], h_df[name + '_upr'], color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)
    ax.set_xlabel('Day of week')
    ax.set_ylabel(name)
    # if m.seasonalities[name]['mode'] == 'multiplicative':
    #     ax = set_y_as_percent(ax)
    return artists


def plot_yearly(m, ax=None, yearly_start=0, figsize=(10, 6), name='yearly', df=None):
    """Plot the yearly component of the forecast.
    Parameters
    ----------
    m: Prophet model.
    df: DF
    ax: Optional matplotlib Axes to plot on. One will be created if
        this is not provided.
    yearly_start: Optional int specifying the start day of the yearly
        seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts
        by 1 day to Jan 2, and so on.
    figsize: Optional tuple width, height in inches.
    name: Name of seasonality component if previously changed from default 'yearly'.
    Returns
    -------
    a list of matplotlib artists
    """
    if df is None:
        fcst = m.fcast_df.copy()
    else:
        fcst = df.copy()

    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)

    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = (pd.date_range(start=m.min_ds, periods=365) + pd.Timedelta(days=yearly_start))
    if name + '_lwr' in fcst.columns and name + '_upr' in fcst.columns:
        y_df = fcst[fcst['ds'].isin(days)][[name, name + '_upr', name + '_lwr']].copy()
    else:
        y_df = fcst[fcst['ds'].isin(days)][[name]].copy()
    artists += ax.plot(y_df['ds'].dt.to_pydatetime(), y_df[name], ls='-', marker='*', c='#0072B2')
    if name + '_lwr' in fcst.columns and name + '_upr' in fcst.columns:
        artists += [ax.fill_between(y_df['ds'].dt.to_pydatetime(), y_df[name + '_lwr'], y_df[name + '_upr'], color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos=None: '{dt:%B} {dt.day}'.format(dt=num2date(x))))
    ax.xaxis.set_major_locator(months)
    ax.set_xlabel('Day of year')
    ax.set_ylabel(name)
    # if m.seasonalities[name]['mode'] == 'multiplicative':
    #     ax = set_y_as_percent(ax)
    return artists


def plot_seasonality(m, name, ax=None, figsize=(10, 6), df=None):
    """Plot a custom seasonal component.
    Parameters
    ----------
    m: Prophet model.
    df: DF
    name: Seasonality name, like 'daily', 'weekly'.
    ax: Optional matplotlib Axes to plot on. One will be created if
        this is not provided.
    figsize: Optional tuple width, height in inches.
    Returns
    -------
    a list of matplotlib artists
    """
    if df is None:
        fcst = m.fcast_df.copy()
    else:
        fcst = df.copy()

    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)

    # Compute seasonality from Jan 1 through a single period.
    period = max([q[0] for q in m.seasonality_info.values()])  # assumed in m.freq units
    start = m.min_ds
    end = m.min_ds + pd.Timedelta(period, unit=m.freq)
    dr = pd.date_range(start=start, end=end, freq=m.freq)
    if name + '_lwr' in fcst.columns and name + '_upr' in fcst.columns:
        y_df = fcst[fcst['ds'].isin(dr)][['ds', name, name + '_upr', name + '_lwr']].copy()
    else:
        y_df = fcst[fcst['ds'].isin(dr)][['ds', name]].copy()

    artists += ax.plot(y_df['ds'].dt.to_pydatetime(), y_df[name], ls='-', marker='*', c='#0072B2')
    if name + '_lwr' in fcst.columns and name + '_upr' in fcst.columns:
        artists += [ax.fill_between(y_df['ds'].dt.to_pydatetime(), y_df[name + '_lwr'], y_df[name + '_upr'], color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    xticks = pd.to_datetime(np.linspace(start.value, end.value, 7)).to_pydatetime()
    ax.set_xticks(xticks)
    if period <= 2:
        fmt_str = '{dt:%T}'
    elif period < 14:
        fmt_str = '{dt:%m}/{dt:%d} {dt:%R}'
    else:
        fmt_str = '{dt:%m}/{dt:%d}'
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos=None: fmt_str.format(dt=num2date(x))))
    ax.set_xlabel('ds')
    ax.set_ylabel(name)
    # if m.seasonalities[name]['mode'] == 'multiplicative':
    #     ax = set_y_as_percent(ax)
    return artists


def set_y_as_percent(ax):
    yticks = 100 * ax.get_yticks()
    yticklabels = ['{0:.4g}%'.format(y) for y in yticks]
    ax.set_yticklabels(yticklabels)
    return ax

