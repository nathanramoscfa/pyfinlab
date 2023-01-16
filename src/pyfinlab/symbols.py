import pandas as pd

"""
Helper function for preprocessing tickers. 
"""

def symbols(tickers, api_source, country_code='US', asset_class_code='Equity', restricted=False, banned=False, allow_cat2=False):
    """
    Converts tickers to the proper format depending on whether api_source is 'yfinance' or 'bloomberg'.

    :param tickers: (list or str) List of tickers or string of single ticker. Example: ['SPY', 'AGG', 'GLD']
    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :param country_code: (str) Country code for tickers if using bloomberg as api_source. For example, SPY on the
                               Bloomberg terminal would be "SPY US Equity" with "US" being the country code.
    :param asset_class_code: (str) Asset class code for tickers if using bloomberg as api_source. For example, SPY
                                   on the Bloomberg terminal would be "SPY US Equity" with "Equity" being the country code.
    :param restricted: (bool) Optional, filters out tickers on the "restricted" tab in ('../data/portopt_inputs.xlsx'). Default is False.
    :param banned: (bool) Optional, filters out tickers on the "banned" tab in ('../data/portopt_inputs.xlsx'). Default is False.
    :return: (list) List of formatted tickers.
    """
    if api_source=='yfinance':
        pass
    elif api_source=='bloomberg':
        try:
            tickers = pd.DataFrame(tickers, columns=['TICKER'])
        except ValueError:
            tickers = pd.DataFrame([tickers], columns=['TICKER'])
        if asset_class_code == 'Equity':
            tickers['TICKER'] = tickers['TICKER'].astype(str) + ' ' + country_code + ' ' + asset_class_code
        else:
            tickers['TICKER'] = tickers['TICKER'].astype(str) + ' ' + asset_class_code
        tickers = tickers.squeeze()
    else:
        raise ValueError('api_source must be set to either yfinance or bloomberg')
    if restricted==True:
        restricted_list = pd.read_excel('../data/portopt_inputs.xlsx', engine='openpyxl', sheet_name='restricted')[
            ['Symbol', 'Prohibition Reason']].dropna().drop_duplicates(subset=['Symbol'], keep='first')
        if allow_cat2==True:
            restricted_list = restricted_list[
                (restricted_list['Prohibition Reason']!='Category 2A ETF') &
                (restricted_list['Prohibition Reason']!='Category 2B ETF') &
                (restricted_list['Prohibition Reason']!='Category 2A ETF - Hold') &
                (restricted_list['Prohibition Reason']!='Category 2B ETF - Hold')
                ]
        if api_source=='bloomberg':
            restricted_list['Symbol'] = restricted_list['Symbol'] + ' US Equity'
        restricted_tickers = list(restricted_list.Symbol)
        tickers = [x for x in tickers if x not in restricted_tickers]
    if banned==True:
        banned_tickers = pd.read_excel('../data/portopt_inputs.xlsx', engine='openpyxl', sheet_name='banned')[
            ['Symbol', 'Prohibition Reason']].dropna().drop_duplicates(subset=['Symbol'], keep='first')
        if api_source=='bloomberg':
            banned_tickers['Symbol'] = banned_tickers['Symbol'] + ' US Equity'
        banned_tickers = list(banned_tickers.Symbol)
        tickers = [x for x in tickers if x not in banned_tickers]
    tickers = tickers.squeeze() if isinstance(tickers, pd.DataFrame) else tickers
    return tickers


# def tickers_(tickers, api_source, country_code='US', asset_class_code='Equity'):
#     """
#     Converts tickers to the proper format depending on whether api_source is 'yfinance' or 'bloomberg'.
#
#     :param tickers: (list or str) List of tickers or string of single ticker. Example: ['SPY', 'AGG', 'GLD']
#     :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
#     :param country_code: (str) Country code for tickers if using bloomberg as api_source. For example, SPY on the
#                                Bloomberg terminal would be "SPY US Equity" with "US" being the country code.
#     :param asset_class_code: (str) Asset class code for tickers if using bloomberg as api_source. For example, SPY
#                                    on the Bloomberg terminal would be "SPY US Equity" with "Equity" being the country code.
#     :return: (list) List of formatted tickers.
#     """
#     if api_source == 'yfinance':
#         pass
#     elif api_source == 'bloomberg':
#         try:
#             tickers = pd.DataFrame(tickers, columns=['TICKER'])
#         except ValueError:
#             tickers = pd.DataFrame([tickers], columns=['TICKER'])
#         if asset_class_code == 'Equity':
#             tickers['TICKER'] = tickers['TICKER'].astype(str) + ' ' + country_code + ' ' + asset_class_code
#         else:
#             tickers['TICKER'] = tickers['TICKER'].astype(str) + ' ' + asset_class_code
#         tickers = tickers.squeeze()
#     else:
#         raise ValueError('api_source must be set to either yfinance or bloomberg')
#     tickers = tickers.squeeze() if isinstance(tickers, pd.DataFrame) else tickers
#     return tickers
