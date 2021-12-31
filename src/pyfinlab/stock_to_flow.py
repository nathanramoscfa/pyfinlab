import re
import requests
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates as mdates
from matplotlib import style, cm
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import stats


def coinmetrics_urls():
    """
    Scapes URLs where data is located.

    :return: (pd.DataFrame, list) Filename URLs of the csv files where data is located along with a list of tickers for
                                  which a csv file exists with data.
    """
    query = re.findall('[^>]+\.csv', requests.get('https://github.com/coinmetrics/data/tree/master/csv/').text)
    filenames, tickers = [], []
    for filename in query:
        try:
            filenames.append(filename.split('href="')[1])
            tickers.append(filename.split('href="')[1].split('/')[-1].split('.')[0])
        except IndexError:
            continue
    filenames = pd.DataFrame(filenames, columns=['filenames'])
    return filenames, tickers


def download():
    """
    Downloads data updated daily by Coin Metrics from https://github.com/coinmetrics/data/tree/master/csv. CSV files
    containing the data is saved to '../data/coinmetrics/' path.
    """
    filenames, tickers = coinmetrics_urls()
    for ticker in tqdm(tickers):
        filename = filenames[filenames['filenames']=='/coinmetrics/data/blob/master/csv/{}.csv'.format(ticker)].squeeze()
        csv = pd.read_csv('https://github.com' + filename + '?raw=true')
        csv.to_csv('../data/coinmetrics/{}.csv'.format(ticker))


def check_tickers(search_term, filenames, tickers, show=False):
    """
    Helper function to filter out tickers which have no available data based on the search term parameter.

    :param search_term: (str) Search term to look for in all ticker datasets. Type only one search term.
    :param filenames: (pd.DataFrame) DataFrame of filenames where csv files are located.
    :param tickers: (list of str) List of tickers to filter through.
    :param show: (bool) Optional, prints tickers which have data available.
    :return: (list) List of tickers which have data available.
    """
    filtered_tickers = []
    for ticker in tqdm(tickers):
        filename = filenames[filenames['filenames'] == '/coinmetrics/data/blob/master/csv/{}.csv'.format(ticker)].squeeze()
        df = pd.read_csv('../data/coinmetrics/{}.csv'.format(ticker))
        columns = pd.DataFrame(df.columns, columns=['columns'])
        if columns[(columns['columns'].str.contains(search_term))].empty is not True:
            filtered_tickers.append(ticker)
        else:
            continue
    if show==True:
        print('Tickers with Available Data: {}'.format(filtered_tickers))
    return filtered_tickers


def timeseries(ticker, fields):
    """
    Helper function to filter out tickers which have no available data based on the search term parameter.

    :param ticker: (str) Ticker of cryptocurrency you want to look up.
    :param fields: (str or list of str) Data fields for which to include in timeseries DataFrame.
    :return: (pd.DateFrame or pd.Series) DataFrame or Series (if only one field) containing requested data.
    """
    df = pd.read_csv('../data/coinmetrics/{}.csv'.format(ticker)).iloc[:, 1:].set_index('time', drop=True)
    df.index = pd.to_datetime(df.index)
    return df[fields].dropna()


def stock_to_flow_data(ticker):
    """
    Compiles required data to compute stock-to-flow model.

    :param ticker: (str) Ticker of cryptocurrency you want to look up.
    :return: (pd.DateFrame) DataFrame containing stock-to-flow data.
    """
    df = timeseries(ticker, ['BlkCnt', 'PriceUSD', 'SplyCur'])
    df.insert(1, 'TotalBlks', df.BlkCnt.cumsum().values)
    df['StocktoFlow'] = df['SplyCur'] / ((df['SplyCur'] - df['SplyCur'].shift(365)))
    return df.dropna().round(2)


def objective(x, a, b):
    """
    Power Law Function
    """
    return np.exp(a) * (x ** b)


def stock_to_flow_model(ticker, p0=None, show=False):
    """
    Computes a fitted stock-to-flow model to observed data, computes Spearman Correlation Coefficient, and tests the
    null hypothesis that there is no correlation between the computed stock-to-flow model and observations. Rejecting the null
    hypothesis means accepting the alternative hypothesis that there is correlation between the stock-to-flow model and
    observed values.

    :param ticker: (str) Ticker of cryptocurrency.
    :param p0: (list of floats) Optional, initial guesses for coefficients a and b in the objective function. Defaults
                                to None.
    :param show: (bool) Optional, prints the results from fitting a power law function to the stock-to-flow data.
                        Defaults to False.
    :return: (pd.DataFrame, np.array) DataFrame containing data necessary to compute stock-to-flow model
                                                  along with a np.array containing the fitted values for coefficients
                                                  a and b in the objective function.
    """
    df = stock_to_flow_data(ticker)
    xdata = df.StocktoFlow.values
    ydata = df.PriceUSD.values
    params, cov = curve_fit(objective, xdata, ydata, p0)
    drawdown = df.PriceUSD.copy().fillna(method='ffill')
    drawdown[np.isnan(drawdown)] = -np.Inf
    roll_max = np.maximum.accumulate(drawdown)
    drawdown = drawdown / roll_max - 1.
    df['ModelPriceUSD'] = (np.exp(params[0]) * (df['StocktoFlow'] ** params[1])).round(4)
    df['PotentialReturn%'] = df['ModelPriceUSD'] / df['PriceUSD'] - 1
    df['MaxDrawdown%'] = drawdown.round(4)
    df.insert(2, 'BlkCntMonthly', df['TotalBlks'] - df['TotalBlks'].shift(30))
    sf = df.StocktoFlow.values[-1].round(2)
    p0 = df.PriceUSD[-1].round(2)
    p1 = df.ModelPriceUSD[-1].round(2)
    r, p = (stats.spearmanr(df['PriceUSD'], df['ModelPriceUSD']))
    r2 = r**2
    n = len(xdata)
    k = len(params)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    if show==True:
        print('Current Stock-to-Flow: {}'.format(sf))
        print('Current Price: ${:,.2f}'.format(p0))
        print('Model Prediction Price: ${:,.2f}'.format(p1))
        print('Potential Return%: {:,.2f}%'.format((p1 / p0 - 1) * 100))
        print('')
        print('Fitted Power Law Model: e^{:.3f} * SF^{:.3f}'.format(*params))
        print('Spearman R-Squared: {}'.format(r2.round(4)))
        print('Adj. Spearman R-Squared: {}'.format(adj_r2.round(4)))
        print('P-value of Correlation Coefficient: {}'.format(p.round(4)))
        print(' ')
        print('Conclusion: ')
        if p < 0.05:
            print('[1]: Spearman R-Squared appears to be statistically significant and different from 0.')
            print('[2]: There appears to be correlation between price and stock-to-flow.')
        else:
            print('[1]: Spearman R-Squared does not appear to be statistically significant or different from 0.')
            print('[2]: There does not appear to be correlation between price and stock-to-flow.')
    return df, params


def clean_tickers(p0=None):
    """
    Display tickers for which data is available for a stock-to-flow model and for which no Error or RuntimeWarning are
    thrown. Refer to https://docs.coinmetrics.io/asset-metrics/network-usage/blkcnt and
    https://docs.coinmetrics.io/asset-metrics/supply/splycur for more information on the required inputs. The method
    below searches all ticker data specifically for 'blkcnt' and 'splycur' and removes tickers which have no data.
    Finally, the method also tries to fit a power law model to the data and filters out tickers whose data causes the
    parameter optimization algorithm to throw a RuntimeWarning. Tickers which have data but throws a RuntimeWarning may
    contain data which cannot be properly fitted to a power law curve for whatever reasons, one of which is not having
    a large enough sample size.

    :param p0: (list of [float,float]) Optional, initial guesses for coefficients a and b in the objective function.
                                       Defaults to None.
    :return: (list) List of tickers which have data available and do not produce a RuntimeWarning.
    """
    warnings.filterwarnings('error')
    filenames, tickers = coinmetrics_urls()
    tickers = check_tickers('BlkCnt', filenames, tickers) # Search Term 1
    tickers = check_tickers('SplyCur', filenames, tickers) # Search Term 2
    filtered_tickers = []
    for ticker in tqdm(tickers):
        try:
            df, params = stock_to_flow_model(ticker, p0)
            filtered_tickers.append(ticker)
        except (AttributeError, ValueError, RuntimeWarning):
            continue
    print('Tickers which ran without raising an Error or RuntimeWarning: {}'.format(filtered_tickers))
    print('Number of Tickers: {}'.format(len(filtered_tickers)))
    return filtered_tickers


def regression_analysis(df, show=False, cov_type='HAC'):
    """
    Tests the null hypothesis that the computed stock-to-flow model does not correlate with actual observed values. Rejecting
    the null hypothesis means accepting the alternative hypothetis that the stock-to-flow model does correlate with
    observed data.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model
    :param show: (bool) Optional, if True, prints the results of the regression analysis and hypothesis test. Defaults
                        to False.
    :param cov_type: (str) Optional, the type of robust sandwich estimator to use. See https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html#statsmodels.regression.linear_model.OLSResults.get_robustcov_results
                           for more information. Defaults to 'HAC'. See https://www.statsmodels.org/devel/generated/statsmodels.stats.sandwich_covariance.cov_hac.html#statsmodels.stats.sandwich_covariance.cov_hac
                           for more information.
    :return: (obj) Results instance with the requested robust covariance as the default
                   covariance of the parameters. Inferential statistics like p-values and hypothesis tests will be based
                   on this covariance matrix.
    """
    x = df['ModelPriceUSD']
    y = df['PriceUSD']
    X = sm.add_constant(x)
    results = sm.OLS(y, X).fit().get_robustcov_results(cov_type, maxlags=1) # 'HAC' uses Newey-West method
    if show==True:
        print(results.summary())
        print('\nConclusion: ')
        if results.f_pvalue < 0.05:
            print('[1] Reject H\N{SUBSCRIPT ZERO} because \N{greek small letter beta}\N{SUBSCRIPT ONE} is statistically different from 0.')
            print('[2] Power law model may have explanatory value.')
        else:
            print('[1] Fail to reject H\N{SUBSCRIPT ZERO} because \N{greek small letter beta}\N{SUBSCRIPT ONE} is not statistically different from 0.')
            print('[2] Power law model does not appear to have explanatory value.')
    return results


def model_significance(ticker, results):
    """
    Generates DataFrame containing statistical significance and correlation data for quick reference.

    :param ticker: (str) Ticker of cryptocurrency.
    :param results: (obj) Results instance with the requested robust covariance as the default covariance of the
                          parameters. Inferential statistics like p-values and hypothesis tests will be based on this
                          covariance matrix.
    :return: (pd.DataFrame) DataFrame containing statistical significance and correlation data.
    """
    return pd.DataFrame(
        index=['f_pvalue', 'const_pvalue', 'beta_pvalue', 'rsquared', 'rsquared_adj'],
        columns=[ticker],
        data=[results.f_pvalue, results.pvalues[0], results.pvalues[1], results.rsquared, results.rsquared_adj]
    ).round(3)


def confidence_interval(df, ticker, results, show=False):
    """
    Generates confidence interval data based on regression analysis.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model
    :param ticker: (str) Ticker of cryptocurrency.
    :param results: (obj) Results instance with the requested robust covariance as the default covariance of the
                          parameters. Inferential statistics like p-values and hypothesis tests will be
                          based on this covariance matrix.
    :param show: (bool) Optional, if True, prints the results of the regression analysis and hypothesis test. Defaults
                        to false.
    :return: (pd.Series, pd.Series) Contains tuple of two pd.Series containing the lower confidence level and upper
                                    confidence level.
    """
    get_prediction = results.get_prediction().summary_frame()
    obs_ci_lower, obs_ci_upper = get_prediction.obs_ci_lower, get_prediction.obs_ci_upper
    if show==True:
        print('Ticker: {}'.format(ticker))
        print('Confidence Level: 95%')
        print('Current Value: ${:,.2f}'.format(df['PriceUSD'][-1]))
        print('Lower 95%: ${:,.2f} or {:,.2f}%'.format(obs_ci_lower[-1], (obs_ci_lower[-1] / df['PriceUSD'][-1] - 1) * 100))
        print('Mean Estimate: ${:,.2f} or {:,.2f}%'.format(results.predict()[-1], (results.predict()[-1] / df['PriceUSD'][-1] - 1) * 100))
        print('Upper 95%: ${:,.2f} or {:,.2f}%'.format(obs_ci_upper[-1], (obs_ci_upper[-1] / df['PriceUSD'][-1] - 1) * 100))
    return obs_ci_lower, obs_ci_upper


def conf_int_chart(df, ticker, results, figsize=(12,6), save=False, show=True):
    """
    Generates a plot of regression model data and confidence interval.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model
    :param ticker: (str) Ticker of cryptocurrency.
    :param results: (obj) Results instance with the requested robust covariance as the default covariance of the
                          parameters. Inferential statistics like p-values and hypothesis tests will be
                          based on this covariance matrix.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the
                                   ticker level. Defaults to (12,6).
    :param save: (bool) Optional, saves the chart as a png file to charts folder. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :return: (plt) Generates log scale pyplot of PriceUSD over ModelPriceUSD with 95% confidence interval.
    """
    params = results.params
    ytrue = df['PriceUSD'].to_numpy()
    ypred = df['ModelPriceUSD'].to_numpy()
    obs_ci_lower, obs_ci_upper = confidence_interval(df, ticker, results)
    plt.style.use('default')
    fig = plt.gcf()
    fig.set_size_inches(figsize)
    plt.plot(ypred, ytrue, 'bo')
    plt.plot(ypred, results.predict(), 'r-')
    plt.plot(ypred, sorted(obs_ci_lower), 'r--')
    plt.plot(ypred, sorted(obs_ci_upper), 'r--')
    plt.title("PriceUSD vs. ModelPriceUSD ({})\n {} to {}".format(
        ticker,
        df.index[0].strftime('%m-%d-%Y'),
        df.index[-1].strftime('%m-%d-%Y')))
    plt.legend([
        'PriceUSD / ModelPriceUSD ({})'.format(ticker),
        'Linear Model: {:.4f}x + {:.4f}'.format(params[1], params[0]),
        '95% Confidence Interval'
    ])
    plt.xlabel('ModelPriceUSD ({})'.format(ticker))
    plt.ylabel('PriceUSD ({})'.format(ticker))
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda ytrue, _: '{:.16g}'.format(ytrue)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda ypred, _: '{:.16g}'.format(ypred)))
    if save == True: plt.savefig(
        '../charts/conf_int_chart_{}.png'.format(datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if show == False: plt.close()


def charts(df, ticker, params, chart=1, figsize=(12,6), save=False, show=True):
    """
    Helper function of preformatted charts to show the results of the stock-to-flow model curve fitting.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model
    :param ticker: (str) Ticker of cryptocurrency.
    :param params: (list of [float,float]) Ticker of cryptocurrency.
    :param chart: (int) Select one of 3 pre-formatted charts labeled 1, 2 and 3. Defaults to 1.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
    :param save: (bool) Optional, saves the chart as a png file to charts folder. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :return: (pyplot) Generates log scale pyplot.
    """
    dates = np.array(df.index)
    sf = df['StocktoFlow'].to_numpy()
    d = (df['MaxDrawdown%'] * 100).to_numpy()
    ytrue = df['PriceUSD'].to_numpy()
    ypred = df['ModelPriceUSD'].to_numpy()
    if chart==1:
        plt.style.use('grayscale')
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.scatter(dates, ytrue, c=d, cmap=cm.jet, lw=1, alpha=1, zorder=5, label=ticker)
        plt.yscale('log', subsy=[1])
        ax.plot(dates, ypred, c='black', label='ModelPriceUSD: e^{:.3f} * SF^{:.3f}'.format(*params))
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda ytrue, _: '{:.16g}'.format(ytrue)))
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Maximum Drawdown%')
        plt.xlabel('Year')
        plt.ylabel('PriceUSD ({})'.format(ticker))
        plt.title("PriceUSD and ModelPriceUSD ({})\n {} to {}".format(
            ticker,
            df.index[0].strftime('%m-%d-%Y'),
            df.index[-1].strftime('%m-%d-%Y')))
        plt.legend()
        plt.show()
    elif chart==2:
        plt.style.use('default')
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        plt.yscale('log')
        plt.plot(dates, ytrue, '-b')
        plt.plot(dates, ypred, 'r')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda ytrue, _: '{:.16g}'.format(ytrue)))
        plt.legend(['PriceUSD ({})'.format(ticker), 'ModelPriceUSD: e^{:.3f} * SF^{:.3f}'.format(*params)])
        plt.title("PriceUSD and ModelPriceUSD ({})\n {} to {}".format(
            ticker,
            df.index[0].strftime('%m-%d-%Y'),
            df.index[-1].strftime('%m-%d-%Y')))
        plt.xlabel('Year')
        plt.ylabel('PriceUSD ({})'.format(ticker))
    elif chart==3:
        plt.style.use('default')
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        plt.plot(sf, ytrue, 'bo', label='data')
        plt.plot(sf, objective(sf, *params), 'r-', label='curve_fit')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda ytrue, _: '{:.16g}'.format(ytrue)))
        plt.legend(['PriceUSD ({})'.format(ticker), 'Fitted Power Law Model: e^{:.3f} * SF^{:.3f}'.format(*params)])
        plt.title("PriceUSD vs. Stock-to-Flow ({})\n {} to {}".format(
            ticker,
            df.index[0].strftime('%m-%d-%Y'),
            df.index[-1].strftime('%m-%d-%Y')))
        plt.xlabel('Stock-to-Flow ({})'.format(ticker))
        plt.ylabel('PriceUSD ({})'.format(ticker))
        plt.yscale('log')
    else:
        raise ValueError('Invalid chart number. Type a valid number to the chart parameter.')
    if save == True: plt.savefig(
        '../charts/chart{}_{}.png'.format(chart, datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if show == False: plt.close()
