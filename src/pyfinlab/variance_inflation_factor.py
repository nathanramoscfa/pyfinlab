import pandas as pd
from datetime import datetime, timedelta
from pyfinlab import data_api as api
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(tickers, num_years=10, api_source='yfinance', threshold=5, restricted=False):
    """
    Computes the variance inflation factor of each asset and removes assets which have high variance inflation factor
    in order to reduce multicollinearity.

    :param tickers: (list) List of tickers. Example: ['SPY', 'AGG', 'GLD']
    :param threshold: (float or int) Limit by which the variance inflation factor cannot exceed. Assets with a
                                     variance inflation factor above this number will be omitted from the assets.
    :return: (pd.DataFrame) Returns DataFrame of assets whose variance inflation factor is below the threshold.
    """
    start_date, end_date = api.start_end_dates(num_years, api_source)
    prices = api.price_history(tickers, start_date, end_date, api_source, restricted=restricted)
    returns = prices.pct_change().dropna()
    for i in tqdm(range(len(returns))):
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(returns.values, i) for i in range(returns.shape[1])]
        vif.index = returns.columns
        if (vif.max()[0] > threshold):
            omit = vif.idxmax()
            returns = returns.drop(omit, axis=1)
    vif.index.name = 'TICKER'
    return vif.sort_values(by='VIF Factor')
