import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def backtest_chart1(backtest_timeseries, start_date, end_date, figsize=(15, 9), save=False, show=True, yscale='linear'):
    """
    Plots the performance for all efficient frontier portfolios.

    :param backtest_timeseries: (pd.DataFrame) Ex-post performance of efficient frontier portfolios.
    :param start_date: (str) Start date string or datetime. Date format 'MM-DD-YYYY'.
    :param end_date: (str) End date string or datetime. Date format 'MM-DD-YYYY'.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
                                   Defaults to (11, 5).
    :param save: (bool) Optional, width, height in inches. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :param yscale: (str) Optional, axis scale type to apply. Choose from 'linear', 'log', 'symlog', or 'logit'. Defaults
                         to linear.
    :return: (fig) Plot of performance for all efficient frontier portfolios.
    """
    backtest_timeseries = backtest_timeseries.loc[start_date: end_date]
    backtest_timeseries = backtest_timeseries / backtest_timeseries.iloc[0]
    x_values = backtest_timeseries.index
    y_values = backtest_timeseries.iloc[:, :-1]
    fig = plt.gcf()
    fig.set_size_inches(figsize)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.plot(x_values, y_values)
    plt.plot(backtest_timeseries.iloc[:, -1:], 'k')
    plt.legend(backtest_timeseries.columns, loc='upper left')
    plt.title("\nBacktest of Portfolios {} to {}".format(
        x_values[0].strftime('%m-%d-%Y'),
        x_values[-1].strftime('%m-%d-%Y')))
    plt.yscale(yscale)
    plt.xlabel('Date')
    plt.ylabel('Value of $1 Investment')
    plt.gcf().autofmt_xdate()
    if save == True: plt.savefig(
        '../charts/backtest_chart1_{}.png'.format(datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if show == False: plt.close()


def backtest_chart2(
        backtest_timeseries, start_date, end_date, portfolios=[2, 5, 9, 13, 17],
        api_source='yfinance', figsize=(15, 9), save=False, show=True, yscale='linear'):
    """
    Plots the performance for 5 selected efficient frontier portfolios.

    :param backtest_timeseries: (pd.DataFrame) Ex-post performance of efficient frontier portfolios.
    :param start_date: (str) Start date string or datetime. Date format 'MM-DD-YYYY'.
    :param end_date: (str) End date string or datetime. Date format 'MM-DD-YYYY'.
    :param portfolios: (list) Optional, list of 5 integers anywhere from 1-20 corresponding to portfolios along the efficient
                              frontier in ascending order of risk. Portfolio 1 has the lowest risk while Portfolio 20
                              has the highest risk.
    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
                                   Defaults to (11, 5).
    :param save: (bool) Optional, width, height in inches. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :param yscale: (str) Optional, axis scale type to apply. Choose from 'linear', 'log', 'symlog', or 'logit'. Defaults
                         to linear.
    :return: (fig) Plot of performance for all efficient frontier portfolios.
    """

    backtest_timeseries = backtest_timeseries.loc[start_date: end_date]
    backtest_timeseries = backtest_timeseries / backtest_timeseries.iloc[0]
    plt.figure(figsize=figsize)
    backtest_timeseries[portfolios[0]].plot(color='#00B050', label='Portfolio {}'.format(portfolios[0]))
    backtest_timeseries[portfolios[1]].plot(color='#92D050', label='Portfolio {}'.format(portfolios[1]))
    backtest_timeseries[portfolios[2]].plot(color='#00B0F0', label='Portfolio {}'.format(portfolios[2]))
    backtest_timeseries[portfolios[3]].plot(color='#FFC000', label='Portfolio {}'.format(portfolios[3]))
    backtest_timeseries[portfolios[4]].plot(color='#FF0000', label='Portfolio {}'.format(portfolios[4]))
    backtest_timeseries[backtest_timeseries.columns[-1]].plot(
        color='#000000',
        label='{} ({})'.format(name(api_source, backtest_timeseries.columns[-1]), backtest_timeseries.columns[-1]))
    plt.yscale(yscale)
    plt.legend(loc='upper left')
    plt.title("\nBacktest of Portfolios {} to {}".format(
        backtest_timeseries.index[0].strftime('%m-%d-%Y'),
        backtest_timeseries.index[-1].strftime('%m-%d-%Y')))
    plt.xlabel('Date')
    plt.ylabel('Value of $1 Investment')
    if save == True: plt.savefig(
        '../charts/backtest_chart2_{}.png'.format(datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if show == False: plt.close()


def backtest_chart3(Results, title='Portfolio Backtests', figsize=(15, 9), save=False, show=True, colormap='jet'):
    """
    Plots the performance for all efficient frontier portfolios.

    :param Results: (object) Results object from bt.backtest.Result(*backtests). Refer to the following documentation
                             https://pmorissette.github.io/bt/bt.html?highlight=display#bt.backtest.Result
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
                                   Defaults to (15, 9).
    :param save: (bool) Optional, width, height in inches. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :param colormap: (str or matplotlib colormap object) Colormap to select colors from. If string, load colormap with
                                                         that name from matplotlib. Defaults to 'jet'.
    :return: (fig) Plot of performance for all efficient frontier portfolios.
    """
    plot = Results.plot(title=title, figsize=figsize, colormap=colormap)
    fig = plot.get_figure()
    plt.legend(loc="upper left")
    if save == True: plt.savefig(
        '../charts/linechart_{}.png'.format(datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if show == False: plt.close()


def backtest_linechart(
        backtest_timeseries, start_date, end_date, portfolios=[2, 5, 9, 13, 17],
        api_source='yfinance', chart=3, figsize=(15, 9), save=False, show=True, yscale='linear',
        Results=None, colormap='jet'
):
    """
    Plots the performance for 5 selected efficient frontier portfolios.

    :param backtest_timeseries: (pd.DataFrame) Ex-post performance of efficient frontier portfolios.
    :param start_date: (str) Start date string or datetime. Date format 'MM-DD-YYYY'.
    :param end_date: (str) End date string or datetime. Date format 'MM-DD-YYYY'.
    :param portfolios: (list) Optional, list of 5 integers anywhere from 1-20 corresponding to portfolios along the efficient
                              frontier in ascending order of risk. Portfolio 1 has the lowest risk while Portfolio 20
                              has the highest risk.
    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
                                   Defaults to (11, 5).
    :param save: (bool) Optional, width, height in inches. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :param yscale: (str) Optional, axis scale type to apply. Choose from 'linear', 'log', 'symlog', or 'logit'. Defaults
                         to linear.
    :return: (fig) Plot of performance for all efficient frontier portfolios.
    """
    if chart == 1:
        backtest_chart1(backtest_timeseries, start_date, end_date, figsize, save, show, yscale)
    elif chart == 2:
        backtest_chart2(
            backtest_timeseries, start_date, end_date, portfolios,
            api_source, figsize, save, show, yscale)
    elif chart == 3:
        backtest_chart3(Results, figsize, save, show, colormap)
    else:
        raise ValueError('chart parameter must be 1 or 2')
