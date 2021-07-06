import yfinance as yf
import tia.bbg.datamgr as dm
from datetime import datetime

"""
These functions help easily pull in financial data using either yfinance (free) or tia (requires Bloomberg terminal 
subscription). 
 
"""


def price_history(tickers, start_date, end_date, api_source='yfinance'):
    """
    Downloads price history data into a pd.DataFrame.

    :param tickers: (list) List of tickers. Example: ['SPY', 'AGG', 'GLD']
    :param start_date: (str) The start date of requested time series. Must be in 'YYYY-MM-DD' (i.e. '2021-06-21') if
                                using yfinance. Must be in 'MM/DD/YYYY' (i.e. '2021-06-21') format if using tia.
    :param end_date: (str) The end date of requested time series. Must be in 'YYYY-MM-DD' (i.e. '2021-06-21') if
                                using yfinance. Must be in 'MM/DD/YYYY' (i.e. '2021-06-21') format if using tia.
    :param api_source: (str) The api source to pull data from. Default is yfinance.
    :return: (pd.DataFrame) Dataframe of daily asset prices as a time series.
    """

    if api_source == 'yfinance':
        try:
            if start_date != datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d'):
                raise ValueError
            elif end_date != datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y-%m-%d'):
                raise ValueError
            else:
                prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        except ValueError:
            raise ValueError(
                'Make sure date parameters are strings formatted like "YYYY-MM-DD" or "2020-06-30".')
    elif api_source == 'bloomberg':
        try:
            for ticker in tickers:
                if ' US Equity' not in ticker:
                    raise ValueError
        except ValueError:
            raise ValueError('Bloomberg tickers need proper format. For example, AAPL should be "AAPL US Equity".')
        try:
            if start_date != datetime.strptime(start_date, '%m/%d/%Y').strftime('%m/%d/%Y'):
                raise ValueError
            elif end_date != datetime.strptime(end_date, '%m/%d/%Y').strftime('%m/%d/%Y'):
                raise ValueError
            else:
                mgr = dm.BbgDataManager()
                prices = mgr[tickers].get_historical('PX_LAST', start_date, end_date, 'DAILY').fillna(method='ffill')
                prices = prices.dropna(axis=1)
        except ValueError:
            raise ValueError(
                'Make sure date parameters are strings formatted like "MM/DD/YYYY" or "06/30/2020".')
    else:
        pass
    if prices.isnull().values.any() is True:
        raise ValueError('NaN values present in dataframe. Check your dates.')
    return prices
