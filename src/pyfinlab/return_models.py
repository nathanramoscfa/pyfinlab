import pandas as pd
import numpy as np
from pypfopt import expected_returns

"""
Available covariance risk models in PyPortfolioOpt library. 
https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#

These functions bring together all covariance matrix risk models from PyPortfolioOpt into one
function for ease of use.  
"""

def return_model(prices, model, risk_free_rate=0.02, frequency=252, span=500):
    """
    Calculates the covariance matrix for a dataframe of asset prices.

    :param prices: (pd.DataFrame) Dataframe where each column is a series of prices for an asset.
    :param model: (str) Risk model to use. Should be one of:

        PyPortfolioOpt
            - 'avg_historical_return',
            - 'ema_historical_return',
            - 'capm_return',

    :param risk_free_rate: (float) Optional, annualized risk-free rate, defaults to 0.02.
    :param frequency: (int) Optional, number of time periods in a year, defaults to 252 (the number of trading days
                            in a year).
    :param span: (int) Optional, time-span for the EMA, defaults to 500-day EMA.
    :return: (pd.DataFrame) Estimated covariance matrix.
    """
    if model == 'avg_historical_return':
        return expected_returns.mean_historical_return(prices, frequency=frequency)
    elif model == 'ema_historical_return':
        return expected_returns.ema_historical_return(prices, frequency=frequency, span=span)
    elif model == 'capm_return':
        return expected_returns.capm_return(prices, risk_free_rate=risk_free_rate, frequency=frequency)
    else:
        raise ValueError('Double-check model parameter entered is correctly.')
