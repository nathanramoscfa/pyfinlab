import pandas as pd
import yfinance as yf
import tia.bbg.datamgr as dm
from tqdm import tqdm
from tia.bbg import LocalTerminal
from datetime import datetime, timedelta
from dateutil.parser import parse

"""
These functions help easily pull financial data using either yfinance (free) or tia (requires Bloomberg terminal 
subscription). 

"""

def price_history(
        tickers, start_date, end_date, api_source='yfinance', country_code='US', asset_class_code='Equity',
        drop_method='dropna', drop_by='index'
):
    """
    Downloads price history data into a pd.DataFrame.

    :param tickers: (list) List of tickers. Example: ['SPY', 'AGG', 'GLD']
    :param start_date: (str) Start date of requested time series. Must be in 'YYYY-MM-DD' (i.e. '2021-06-21') if
                             api_source is yfinance. Must be in 'MM/DD/YYYY' (i.e. '2021-06-21') format if api_source is
                             bloomberg.
    :param end_date: (str) End date of requested time series. Must be in 'YYYY-MM-DD' (i.e. '2021-06-21') if
                           api_source is yfinance. Must be in 'MM/DD/YYYY' (i.e. '2021-06-21') format if api_source is
                           bloomberg.
    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :param country_code: (str) Country code for tickers if using bloomberg as api_source. For example, SPY on the
                               Bloomberg terminal would be "SPY US Equity" with "US" being the country code.
    :param asset_class_code: (str) Asset class code for tickers if using bloomberg as api_source. For example, SPY
                                   on the Bloomberg terminal would be "SPY US Equity" with "Equity" being the country code.
    :param drop_method: (str): Optional, useful for creating DataFrame of prices all starting on the date for which all
                                tickers have a price. 'dropna' removes rows with missing prices. 'bfill' backfills
                                NaN values using the first available price.
    :param drop_by: (str): Optional, 'index' will drop rows with NaN. 'columns' will drop columns with NaN.
    :return: (pd.DataFrame) Dataframe of daily asset prices as a time series.

    """
    if api_source == 'yfinance':
        prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
    elif api_source == 'bloomberg':
        mgr = dm.BbgDataManager()
        prices = mgr[tickers].get_historical('PX_LAST', start_date, end_date, 'DAILY').fillna(method='ffill')
        prices = prices.reindex(sorted(prices.columns), axis=1)
        prices.columns = [ticker.replace(' {} {}'.format(country_code, asset_class_code), '') for ticker in prices.columns]
    else: raise ValueError('api_source must be set to either yfinance or bloomberg')
    if drop_method=='dropna':
        if drop_by=='index':
            prices = prices.dropna()
        elif drop_by=='columns':
            prices = prices.dropna(axis=1)
    elif drop_method=='bfill':
        prices = prices.bfill()
    else:
        pass
    return prices


def risk_free_rate(start_date, end_date, api_source='yfinance'):
    """
    Downloads the price data for a risk-free rate index and computes the average risk-free rate between start_date and
    end date.

    :param start_date: (str) Start date of requested time series. Must be in 'YYYY-MM-DD' (i.e. '2021-06-21') if
                             api_source is yfinance. Must be in 'MM/DD/YYYY' (i.e. '2021-06-21') format if api_source is
                             bloomberg.
    :param end_date: (str) End date of requested time series. Must be in 'YYYY-MM-DD' (i.e. '2021-06-21') if
                           api_source is yfinance. Must be in 'MM/DD/YYYY' (i.e. '2021-06-21') format if api_source is
                           bloomberg.
    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :return: (float) The average risk-free rate between start_date and end_date in decimal units.
    """
    if api_source == 'yfinance':
        return (price_history(['^TNX'], start_date, end_date, 'yfinance').mean() / 100).round(4).squeeze()
    elif api_source == 'bloomberg':
        return (price_history(['USGG10YR Index'], start_date, end_date, 'bloomberg', None, 'Index').mean() / 100).round(4).squeeze()
    else:
        raise ValueError('api_source must be set to either yfinance or bloomberg')


def current_equity_data(
        tickers, info=None, api_source='yfinance', get_list=False,
        start_date=None, end_date=None, market_data_override=None, calc_interval=None, best_fperiod_override=None,
        scaling_format=None
):
    """
    Downloads point-in-time data. For example, you can download the current price or fundamental data like the PE
    ratio. You can only download available data for one ticker at a time if api_source is yfinance. You can download
    current point-in-time data for as many tickers as you want if your api_source is bloomberg.

    :param tickers: (list or str) List of tickers or string of single ticker. You can only look up one ticker at a time
                                  if api_source is yfinance. You can look up as many tickers as you want if api_source
                                  is bloomberg. Example: ['SPY', 'AGG', 'GLD']
    :param info: (list) List of information to lookup. Also called mnemonics on the Bloomberg terminal.
    :param get_list: (boolean) If True, the function will only display a dictionary of available information from yfinance.
                               The keys this dictionary are what the info parameter of this function uses to look up
                               information. This parameter only works if api_source is yfinance, otherwise, it is ignored.
    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :param start_date: (str) Start date string or datetime. Date format must be 'YYYY-MM-DD'.
    :param end_date: (str) End date string or datetime. Date format must be 'YYYY-MM-DD'.
    :param market_data_override: (str) Type of market data used in the calculation. Any historical field can be used for
                                       this override. Use FLDS function on the Bloomberg Terminal to lookup fields.
    :param calc_interval: (str) Number and type of period that contains the data for the override's calculation. This
                                field accepts overrides in a "number/period" format. Numbers may consist of any integer.
                                Period consists of d(day), w(week), m(month), q(quarter), s(semi-annual), and y(year).
                                Fifteen month would be entered as "15m" to override correctly. In addition, users can
                                set the override to WTD (week to date), MTD (month to date), YTD (year to date), FWTD
                                (first day of week to date), FMTD (first day of month to date), FYTD ( first day of
                                year to date).
    :param calc_interval: (str) Specifies the fiscal period override for Bloomberg Estimates at the company level. There
                                are two valid formats, relative and fixed. This override can also be used on certain
                                applicable Equity Index fields. Search BEST_FPERIOD_OVERRIDE on the Bloomberg terminal
                                for more information.
    :param best_fperiod_override: (str) Specifies the fiscal period override for Bloomberg Estimates at the company level.
                                        There are two valid formats, relative and fixed. This override can also be used
                                        on certain applicable Equity Index fields.
    :param scaling_format: (str) Specifies the display format of calculated field returns. Specifies the display unit for
                                 equity fundamental data. The following values can be used as overrides to change the
                                 scaling for fundamental fields:  'UNT' - Units; 'K' - Thousands; 'MLN' - Millions;
                                 'BLN' - Billions; 'TRN' - Trillions.
    :return: (str) Returns the current point-in-time data as specified in the info parameter for the requested tickers.
    """
    if api_source == 'yfinance':
        if isinstance(tickers, str):
            if len([tickers])!=1:
                raise ValueError(
                    'yfinance api only allows one ticker at a time. Check your ticker list to ensure it contains only one ticker.')
        else:
            if len(tickers) != 1:
                raise ValueError(
                    'yfinance api only allows one ticker at a time. Check your ticker list to ensure it contains only one ticker.')
        if get_list:
            return yf.Ticker(tickers).info
        else:
            df = yf.Ticker(tickers).info
            df = pd.DataFrame.from_dict(
                dict((k, df[k]) for k in info),
                orient='index', columns=[tickers]).T
            df.index.name = 'TICKER'
            return df
    elif api_source == 'bloomberg':
        df = LocalTerminal.get_reference_data(
            tickers, info,
            CUST_TRR_START_DT=start_date,
            CUST_TRR_END_DT=end_date,
            MARKET_DATA_OVERRIDE=market_data_override,
            CALC_INTERVAL=calc_interval,
            BEST_FPERIOD_OVERRIDE=best_fperiod_override,
            SCALING_FORMAT=scaling_format,
            ignore_field_error=1, ignore_security_error=1).as_frame()
        df.index.name = 'TICKER'
        return df
    else:
        raise ValueError('api_source must be either yfinance or bloomberg. ')


def start_end_dates(num_years=10, api_source='yfinance'):
    """
    Helper function to convert datetime dates to appropriate date format for the api_source. yfinance requires dates to
    be in the "%Y-%m-%d" format and bloomberg requires the '%m-%d-%Y' format.

    :param num_years: (float or int) Indicate the number of years into the past you want to pull data for.
    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :return: (str) Returns the start_date and end_date properly formatted for the specified api_source.
    """
    start_date = (datetime.today() - timedelta(days=365 * num_years))
    end_date = (datetime.today() - timedelta(days=0))
    if api_source == 'yfinance':
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    elif api_source == 'bloomberg':
        start_date = start_date.strftime('%m-%d-%Y')
        end_date = end_date.strftime('%m-%d-%Y')
    else:
        raise ValueError('api_source must be either yfinance or bloomberg. ')
    return start_date, end_date


def name(ticker=['SPY'], api_source='yfinance'):
    """
    Downloads the name of each ticker in ticker list. Only 1 ticker can be downloaded at a time if using yfinance as
    api_source. There is not such limit if api_source is bloomberg.

    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :param ticker: (list) List of single ticker if using yfinance, or, multiple tickers if using bloomberg.
    :return: (str) Name of the ticker, or, tickers.
    """
    if api_source=='yfinance':
        name = current_equity_data(ticker, ['longName'], api_source).squeeze()
    elif api_source=='bloomberg':
        name = current_equity_data(ticker + ' US Equity', ['LONG_COMP_NAME'], api_source).squeeze()
    else:
        raise ValueError('api_source must be set to either yfinance or bloomberg')
    return name


def coinmetrics_urls():
    """
    Scapes URLs where data is located.

    :returns: (pd.DataFrame, list) Filename URLs of the csv files where data is located along with a list of tickers for
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


def coinmetrics_download():
    """
    Downloads data updated daily by Coin Metrics from https://github.com/coinmetrics/data/tree/master/csv. CSV files
    containing the data is saved to '../data/coinmetrics/' path.
    """
    filenames, tickers = coinmetrics_urls()
    for ticker in tqdm(tickers):
        filename = filenames[filenames['filenames']=='/coinmetrics/data/blob/master/csv/{}.csv'.format(ticker)].squeeze()
        csv = pd.read_csv('https://github.com' + filename + '?raw=true')
        csv.to_csv('../data/coinmetrics/{}.csv'.format(ticker))


def coinmetrics_tickers(search_term, filenames, tickers, show=False):
    """
    Helper function to filter out tickers which have no available data based on the search term parameter.

    :param search_term: (str) Search term to look for in all ticker datasets. Type only one search term.
    :param filenames: (pd.DataFrame) DataFrame of filenames where csv files are located.
    :param tickers: (list of str) List of tickers to filter through.
    :param show: (bool) Optional, prints tickers which have data available.
    :returns: (list) List of tickers which have data available.
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


def coinmetrics_timeseries(ticker, fields):
    """
    Helper function to filter out tickers which have no available data based on the search term parameter.

    :param ticker: (str) Ticker of cryptocurrency you want to look up.
    :param fields: (str or list of str) Data fields for which to include in timeseries DataFrame.
    :returns: (pd.DateFrame or pd.Series) DataFrame or Series (if only one field) containing requested data.
    """
    df = pd.read_csv('../data/coinmetrics/{}.csv'.format(ticker)).iloc[:, 1:].set_index('time', drop=True)
    df.index = pd.to_datetime(df.index)
    return df[fields].dropna()
