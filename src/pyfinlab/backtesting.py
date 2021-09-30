import bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pyfinlab import data_api as api
np.seterr(divide='ignore')

"""
These functions backtest the efficient frontier portfolios.  
"""
class OrderedWeights(bt.Algo):
    def __init__(self, weights):
        self.target_weights = weights

    def __call__(self, target):
        target.temp['weights'] = dict(zip(target.temp['selected'], self.target_weights))
        return True


def backtest_parameters(portfolio, weightings, prices):
    """
    Creates Backtest object combining Strategy object with price data.

    :param portfolio: (int) Choose any portfolio from 1-20.
    :param weightings: (pd.DataFrame) Weightings for efficient frontier portfolios.
    :param prices: (pd.DataFrame) Dataframe where each column is a series of prices for an asset.
    :return: (obj) Backtest object combining Strategy object with price data.
    """
    target_weights = weightings[portfolio]
    target_weights = target_weights[target_weights!=0].to_frame()
    tickers = list(target_weights.index)
    weights_dict = target_weights.to_dict().get(portfolio)
    prices_df = prices[tickers]
    strategy = bt.Strategy('{}'.format(portfolio), [bt.algos.RunQuarterly(),
                           bt.algos.SelectAll(tickers),
                           OrderedWeights(list(weights_dict.values())),
                           bt.algos.Rebalance()])
    return bt.Backtest(strategy, prices_df)


def compile_backtests(weightings, prices):
    """
    Compiles multiple backtest objects.

    :param weightings: (pd.DataFrame) Weightings for efficient frontier portfolios.
    :param prices: (pd.DataFrame) Dataframe where each column is a series of prices for an asset.
    :return: (list) List of Backtest objects, one for each efficient frontier portfolio.
    """
    backtests = []
    for backtest in list(weightings.columns):
        backtests.append(backtest_parameters(backtest, weightings, prices))
    return backtests


def benchmark_strategy(benchmark_ticker='SPY'):
    """
    Creates a Strategy object for the benchmark ticker.

    :param benchmark_ticker: (str) Optional, benchmark ticker. Defaults to 'SPY'.
    :return: (obj) Strategy object assigned to the benchmark.
    """
    return bt.Strategy(
        benchmark_ticker,
        algos = [bt.algos.RunQuarterly(),
        bt.algos.SelectAll(),
        bt.algos.SelectThese([benchmark_ticker]),
        bt.algos.WeighEqually(),
        bt.algos.Rebalance()],
    )


def benchmark_backtest(benchmark_ticker, start_date, end_date, api_source):
    """
    Creates Backtest object combining Strategy object with price data from the benchmark.

    :param benchmark_ticker: (str) Optional, benchmark ticker. Defaults to 'SPY'.
    :param start_date: (str) Start date of requested time series. Must be in 'YYYY-MM-DD' (i.e. '2021-06-21') if
                             api_source is yfinance. Must be in 'MM/DD/YYYY' (i.e. '2021-06-21') format if api_source is
                             bloomberg.
    :param end_date: (str) End date of requested time series. Must be in 'YYYY-MM-DD' (i.e. '2021-06-21') if
                           api_source is yfinance. Must be in 'MM/DD/YYYY' (i.e. '2021-06-21') format if api_source is
                           bloomberg.
    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :return: (obj) Backtest object combining Strategy object with price data.
    """
    benchmark_prices = api.price_history([benchmark_ticker], start_date, end_date, api_source)
    benchmark_prices.columns = [benchmark_ticker]
    benchmark_name = api.name(api_source, benchmark_ticker)
    return bt.Backtest(benchmark_strategy(benchmark_ticker), benchmark_prices)


def run_backtest(backtests, benchmark):
    """
    Runs the backtest.

    :param backtests: (list) List of Backtest objects, one for each efficient frontier portfolio.
    :param benchmark: (list) Backtest object for the benchmark_strategy.
    :return: (obj) Result object containing backtest results.
    """
    return bt.run(
        backtests[0], backtests[1], backtests[2], backtests[3], backtests[4],
        backtests[5], backtests[6], backtests[7], backtests[8], backtests[9],
        backtests[10], backtests[11], backtests[12], backtests[13], backtests[14],
        backtests[15], backtests[16], backtests[17], backtests[18], backtests[19],
        benchmark
    )


def linechart(Results, title='backtests', figsize=(15, 9), save=False, show=True, colormap='jet'):
    """
    Plots the performance for all efficient frontier portfolios.

    :param Results: (object) Results object from bt.backtest.Result(*backtests). Refer to the following documentation
                             https://pmorissette.github.io/bt/bt.html?highlight=display#bt.backtest.Result
    :param title: (str) Optional, used to name image file if saved. Defaults to 'backtests'.
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


def backtest_timeseries(Results, freq='d'):
    """
    Plots the performance for all efficient frontier portfolios.

    :param Results: (object) Results object from bt.backtest.Result(*backtests). Refer to the following documentation
                             https://pmorissette.github.io/bt/bt.html?highlight=display#bt.backtest.Result
    :param freq: (str) Data frequency used for display purposes. Refer to pandas docs for valid freq strings.
    :return: (pd.DataFrame) Time series of each portfolio's value over time according to the backtest Results object.
    """
    return Results._get_series(freq).drop_duplicates().iloc[1:]
