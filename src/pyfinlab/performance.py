import ffn
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from patsy import dmatrices
from pyfinlab import return_models as rets
from pyfinlab import risk_models as risk

"""
These functions measure the historical performance of efficient frontier portfolios. 
"""
periods = [5, 21, 63, 126, 252, 756, 1260, 2520]


def days(start_date, end_date):
    """
    Counts the number of days between start and end dates. Helpful for splicing backtest timeseries pd.DataFrame.

    :param start_date: (str) Start date string or datetime. Date format 'MM-DD-YYYY'.
    :param end_date: (str) End date string or datetime. Date format 'MM-DD-YYYY'.
    :return: (int) Number of days between start_date and end_date.
    """
    date_format = "%m-%d-%Y"
    a = datetime.strptime(start_date, date_format)
    b = datetime.strptime(end_date, date_format)
    delta = b - a
    return delta.days


def period_count(backtest_timeseries):
    """
    Plots the performance for all efficient frontier portfolios.

    :param backtest_timeseries: (pd.DataFrame) Timeseries performance of efficient frontier portfolios.
    :return: (int) Number of periods from backtest timeseries.
    """
    count = 0
    for period, i in zip(periods, range(0, len(periods))):
        try:
            backtest_timeseries.index[-period].strftime('%m-%d-%Y')
            count += 1
        except IndexError:
            break
    return count


def backtested_periods(backtest_statistics):
    """
    Helper function for organizing column labels and their corresponding index in backtest_statistics.

    :param backtest_statistics: (pd.DataFrame) Ex-post performance of efficient frontier portfolios.
    :return: (dict) Dictionary of column labels as keys and their corresponding index in backtest_statistics.
    """
    periods = list(backtest_statistics.keys())
    columns = ['1WK', '1MO', '3MO', '6MO', '1YR', '3YR', '5YR', '10YR']
    dictionary = dict(zip(columns, periods))
    return dictionary


def performance_stats(backtest_timeseries, risk_model='sample_cov', benchmark_ticker='SPY', risk_free_rate=0.02):
    """
    Computes the cumulative performance statistics based on data from the backtest_timeseries.

    :param backtest_timeseries: (pd.DataFrame) Timeseries performance of efficient frontier portfolios.
    :param risk_model: (str) Optional, risk model used to compute cov_matrix from either PyPortfolioOpt (free open
                             source) or Hudson & Thames' PortfolioLab (subscription required). Defaults to 'sample_cov'.
    :param benchmark_ticker: (str) Optional, benchmark ticker. Takes only one ticker. Defaults to 'SPY'.
    :param risk_free_rate: (float) Optional, annualized risk-free rate, defaults to 0.02.
    :return: (pd.DataFrame) DataFrame of cumulative performance statistics for all efficient frontier portfolios.
    """
    perf = ffn.core.GroupStats(backtest_timeseries)
    perf.set_riskfree_rate(float(risk_free_rate))
    portfolios = list(backtest_timeseries.columns)
    start_date = backtest_timeseries.index[0].strftime('%m-%d-%Y')
    end_date = backtest_timeseries.index[-1].strftime('%m-%d-%Y')

    cagrs = {}
    vols = {}
    capms = {}
    betas = {}
    jensen_alphas = {}
    appraisal_ratios = {}
    appraisal_ratios = {}
    sharpes = {}
    treynors = {}
    information_ratios = {}
    sortinos = {}
    capture_ratios = {}
    drawdowns = {}
    ulcers = {}
    m2s = {}
    m2_alphas = {}

    for portfolio in portfolios[:-1]:
        p = backtest_timeseries.copy()[[portfolio, benchmark_ticker]]
        r = p.pct_change().dropna()
        risk_free_rate = risk_free_rate / (252 / r.shape[0]) if r.shape[0] < 252 else risk_free_rate / 252
        p.name, r.name = portfolio, benchmark_ticker
        # return
        cagr = (1 + r).prod() ** (252 / (252 if r.shape[0] < 252 else r.shape[0])) - 1
        # risk
        vol = r.std() * (252 if r.shape[0] > 252 else r.shape[0]) ** 0.5
        # client regression model
        y, x = r[portfolio], r[benchmark_ticker]
        yx = pd.concat([y, x], axis=1)
        y, X = dmatrices(
            'y ~ x',
            data=yx,
            return_type='dataframe'
        )
        mod = sm.OLS(y, X)
        res = mod.fit()
        # benchmark regression model
        y_b, x_b = r[benchmark_ticker], r[benchmark_ticker]
        yx_b = pd.concat([y_b, x_b], axis=1)
        y_b, X_b = dmatrices(
            'y_b ~ x_b',
            data=yx_b,
            return_type='dataframe'
        )
        mod_b = sm.OLS(y_b, X_b)
        res_b = mod_b.fit()
        # capm
        capm = risk_free_rate + res.params.values[1] * (cagr[benchmark_ticker] - risk_free_rate)
        beta = res.params.values[1]
        capm_b = risk_free_rate + res_b.params.values[1] * (cagr[benchmark_ticker] - risk_free_rate)
        beta_b = res_b.params.values[1]
        # jensen's alpha
        non_systematic_risk = (
                vol[portfolio] ** 2
                - res.params.values[1] ** 2
                * vol[benchmark_ticker] ** 2
        )
        non_systematic_risk_b = (
                vol[benchmark_ticker] ** 2
                - res_b.params.values[1] ** 2
                * vol[benchmark_ticker] ** 2
        )
        jensen_alpha = float(cagr[portfolio] - capm)
        jensen_alpha_b = float(cagr[benchmark_ticker] - capm_b)
        # appraisal ratio
        appraisal_ratio = jensen_alpha / (non_systematic_risk ** 0.5)
        appraisal_ratio_b = jensen_alpha_b / (non_systematic_risk_b ** 0.5)
        # sharpe ratio
        sharpe = (cagr[portfolio] - risk_free_rate) / vol[portfolio]
        sharpe_b = (cagr[benchmark_ticker] - risk_free_rate) / vol[benchmark_ticker]
        # treynor ratio
        treynor = cagr[portfolio] / beta
        treynor_b = cagr[benchmark_ticker] / 1.
        # information ratio
        yx1 = yx.copy()
        yx1['Active_Return'] = yx1[portfolio] - yx1[benchmark_ticker]
        information_ratio = yx1['Active_Return'].mean() / yx1['Active_Return'].std()
        # sortino ratio
        downside_returns = (yx1[yx1[portfolio] < 0])[portfolio].values
        downside_deviation = downside_returns.std() * (252 if r.shape[0] > 252 else r.shape[0]) ** 0.5
        sortino = cagr[portfolio] / downside_deviation
        downside_returns_b = (yx1[yx1[benchmark_ticker] < 0])[[benchmark_ticker]].values
        downside_deviation_b = downside_returns_b.std() * (252 if r.shape[0] > 252 else r.shape[0]) ** 0.5
        sortino_b = cagr[benchmark_ticker] / downside_deviation_b
        # capture ratio
        up_returns = yx[yx[portfolio] >= 0].round(4)
        try:
            up_geo_avg = (1 + up_returns[portfolio]).prod() ** (1 / len(up_returns.index)) - 1
            up_geo_avg_b = (1 + up_returns[benchmark_ticker]).prod() ** (1 / len(up_returns.index)) - 1
            down_returns = yx[yx[portfolio] < 0].round(4)
            down_geo_avg = (1 + down_returns[portfolio]).prod() ** (1 / len(down_returns.index)) - 1
            down_geo_avg_b = (1 + down_returns[benchmark_ticker]).prod() ** (1 / len(down_returns.index)) - 1
            up_capture = up_geo_avg / up_geo_avg_b
            down_capture = down_geo_avg / down_geo_avg_b
            capture_ratio = up_capture / down_capture
            capture_ratio_b = 1.
        except ZeroDivisionError:
            capture_ratio = np.nan
            capture_ratio_b = 1.
        # drawdown
        lowest_return = yx[portfolio].min()
        drawdown = p.copy()[[portfolio]]
        drawdown = drawdown.fillna(method='ffill')
        drawdown[np.isnan(drawdown)] = -np.Inf
        roll_max = np.maximum.accumulate(drawdown)
        drawdown = drawdown / roll_max - 1.
        drawdown = drawdown.round(4)
        drawdown = drawdown.iloc[-1:, :].squeeze()
        lowest_return_b = yx[benchmark_ticker].min()
        drawdown_b = p.copy()[[benchmark_ticker]]
        drawdown_b = drawdown_b.fillna(method='ffill')
        drawdown_b[np.isnan(drawdown_b)] = -np.Inf
        roll_max_b = np.maximum.accumulate(drawdown_b)
        drawdown_b = drawdown_b / roll_max_b - 1.
        drawdown_b = drawdown_b.round(4)
        drawdown_b = drawdown_b.iloc[-1:, :].squeeze()
        # ulcer performance index
        ulcer = \
        ffn.core.to_ulcer_performance_index(
            p[[portfolio]], risk_free_rate, nperiods=252).to_frame('ulcer_index').values[0].squeeze()
        ulcer_b = ffn.core.to_ulcer_performance_index(
            p[[benchmark_ticker]], risk_free_rate, nperiods=252).to_frame('ulcer_index').values[0].squeeze()
        # M^2 alpha
        m2 = float(sharpe * vol[benchmark_ticker] + risk_free_rate)
        m2_b = float(sharpe_b * vol[benchmark_ticker] + risk_free_rate)
        m2_alpha = m2 - cagr[benchmark_ticker]
        m2_alpha_b = m2_b - cagr[benchmark_ticker]
        # record results
        cagrs[portfolio] = cagr[portfolio]
        vols[portfolio] = vol[portfolio]
        capms[portfolio] = capm
        betas[portfolio] = beta
        jensen_alphas[portfolio] = jensen_alpha
        appraisal_ratios[portfolio] = appraisal_ratio
        sharpes[portfolio] = sharpe
        treynors[portfolio] = treynor
        information_ratios[portfolio] = information_ratio
        sortinos[portfolio] = sortino
        capture_ratios[portfolio] = capture_ratio
        drawdowns[portfolio] = drawdown
        ulcers[portfolio] = ulcer.round(4)
        m2s[portfolio] = m2
        m2_alphas[portfolio] = m2_alpha
    cagrs[benchmark_ticker] = cagr[benchmark_ticker]
    vols[benchmark_ticker] = vol[benchmark_ticker]
    capms[benchmark_ticker] = capm_b
    betas[benchmark_ticker] = beta_b
    jensen_alphas[benchmark_ticker] = jensen_alpha_b
    appraisal_ratios[benchmark_ticker] = 0
    sharpes[benchmark_ticker] = sharpe_b
    treynors[benchmark_ticker] = treynor_b
    information_ratios[benchmark_ticker] = 0
    sortinos[benchmark_ticker] = sortino_b
    capture_ratios[benchmark_ticker] = capture_ratio_b
    drawdowns[benchmark_ticker] = drawdown_b
    ulcers[benchmark_ticker] = ulcer_b.round(4)
    m2s[benchmark_ticker] = m2_b
    m2_alphas[benchmark_ticker] = m2_alpha_b

    cols = [
        'vol',
        'beta',
        'cagr',
        'drawdown',
        'capm',
        'jensen_alpha',
        'm2',
        'm2_alpha',
        'sharpe',
        'treynor',
        'sortino',
        'info_ratio',
        'capture_ratio',
        'appraisal_ratio',
        'ulcer',
    ]

    dicts = [
        vols,
        betas,
        cagrs,
        drawdowns,
        capms,
        jensen_alphas,
        m2s,
        m2_alphas,
        sharpes,
        treynors,
        sortinos,
        information_ratios,
        capture_ratios,
        appraisal_ratios,
        ulcers,
    ]

    performance_data = pd.DataFrame(index=list(cagrs.keys()), columns=cols).reset_index()
    for col, d in zip(cols, dicts):
        performance_data[col] = performance_data['index'].map(d)
    performance_data = performance_data.set_index('index')
    performance_data.index.name = start_date + ' - ' + end_date
    return performance_data.round(4)


def compile_performance_stats(backtest_timeseries, risk_model, benchmark_ticker, risk_free_rate=0.02):
    """
    Compiles the performance statistics of multiple backtest periods as DataFrames within a dictionary organized by time
    period.

    :param backtest_timeseries: (pd.DataFrame) Timeseries performance of efficient frontier portfolios.
    :param risk_model: (str) Optional, risk model used to compute cov_matrix from either PyPortfolioOpt (free open
                             source) or Hudson & Thames' PortfolioLab (subscription required). Defaults to 'sample_cov'.
    :param benchmark_ticker: (str) Optional, benchmark ticker. Takes only one ticker. Defaults to 'SPY'.
    :param risk_free_rate: (float) Optional, annualized risk-free rate, defaults to 0.02.
    :return: (dict) Dictionary of DataFrames of performance statistics organized by time period.
    """
    count = period_count(backtest_timeseries)
    compiled_stats = {}
    np.seterr(divide='ignore', invalid='ignore')
    for period in periods[:count]:
        compiled_stats[period] = performance_stats(
            backtest_timeseries.iloc[-period:, :], risk_model, benchmark_ticker, risk_free_rate)
    np.seterr(divide='warn', invalid='warn')
    return compiled_stats


def periodic_data(backtest_timeseries, backtest_statistics, stat='cagr'):
    """
    Compiles the performance statistics of multiple backtest periods as DataFrames within a dictionary organized by time
    period.

    :param backtest_timeseries: (pd.DataFrame) Timeseries performance of efficient frontier portfolios.
    :param backtest_statistics: (pd.DataFrame) Ex-post performance of efficient frontier portfolios.
    :param stat: (str) Optional, statistic you want to display. Defaults to 'cagr'. Use the available_keys() function
                       to see the available statistics.
    :return: (pd.DataFrame) DataFrame of the specified statistic for all efficient frontier portfolios over all
                            available time periods.
    """
    periods = list(backtested_periods(backtest_statistics).values())
    count = period_count(backtest_timeseries)
    cols = list(backtested_periods(backtest_statistics).keys())
    df = pd.DataFrame(index=backtest_timeseries.copy().columns)
    df.index.name = stat
    for period in periods:
        try:
            if period < 252:
                df['{}D'.format(period)] = backtest_statistics[period][stat]
            else:
                df['{}YR'.format(int(period / 252))] = backtest_statistics[period][stat]
        except KeyError:
            pass
    df = df.iloc[:, :count]
    df.columns = cols[:count]
    return df


def available_keys(backtest_statistics, backtested_periods, time_period='1WK'):
    """
    Creates a list of periods that were backtested.
    Possible time periods: '1WK', '1MO', '3MO', '6MO', '1YR', '3YR', '5YR', '10YR'

    :param backtest_timeseries: (pd.DataFrame) Ex-post performance of efficient frontier portfolios.
    :param backtested_periods: (dict) Dictionary computed from backtest_periods() function.
    :param time_period: (str) Optional, time period for which you want to see backtest_statistics.
    :return: (list) Returns list of what statistics are being computed.
    """
    return list(backtest_statistics[backtested_periods.get(time_period)].columns)


def compile_periodic_stats(backtest_timeseries, backtest_statistics, backtested_periods):
    """
    Creates a list of periods that were backtested.
    Possible time periods: '1WK', '1MO', '3MO', '6MO', '1YR', '3YR', '5YR', '10YR'

    :param backtest_timeseries: (pd.DataFrame) Timeseries performance of efficient frontier portfolios.
    :param backtest_statistics: (pd.DataFrame) Ex-post performance of efficient frontier portfolios.
    :param backtested_periods: (dict) Dictionary computed from backtest_periods() function.
    :return: (dict) Dictionary of DataFrames of performance statistics organized by statistic.
    """
    keys = available_keys(backtest_statistics, backtested_periods)
    periodic_stats = {}
    for key in keys:
        periodic_stats[key] = periodic_data(backtest_timeseries, backtest_statistics, key)
    return periodic_stats
