import pandas as pd
import numpy as np
from pypfopt import risk_models as risk_models_

"""
Available covariance risk models in PyPortfolioOpt library. 
https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#

These functions bring together all covariance matrix risk models from PyPortfolioOpt into one
function for ease of use.  
"""

risk_models = [
    # PyPortfolioOpt
    'sample_cov',
    'semicovariance',
    'exp_cov',
    'ledoit_wolf_constant_variance',
    'ledoit_wolf_single_factor',
    'ledoit_wolf_constant_correlation',
    'oracle_approximating',
]

def risk_model(prices, model):
    """
    Calculates the covariance matrix for a dataframe of asset prices.

    :param prices: (pd.DataFrame) Dataframe where each column is a series of prices for an asset.
    :param model: (str) Risk model to use. Should be one of:

        PyPortfolioOpt
            - 'sample_cov',
            - 'semicovariance',
            - 'exp_cov',
            - 'ledoit_wolf_constant_variance',
            - 'ledoit_wolf_single_factor'
            - 'ledoit_wolf_constant_correlation',
            - 'oracle_approximating'

    :return: (pd.DataFrame) Estimated covariance matrix.
    """
    tn_relation = prices.shape[0] / prices.shape[1]
    sample_cov = prices.pct_change().dropna().cov()
    if model == 'sample_covariance':
        return prices.pct_change().dropna().cov()
    elif model == 'sample_cov':
        covariance_matrix = risk_models_.fix_nonpositive_semidefinite(
            risk_models_.sample_cov(prices)) / 252
    elif model == 'semicovariance':
        covariance_matrix = risk_models_.fix_nonpositive_semidefinite(
            risk_models_.semicovariance(prices)) / 252
    elif model == 'exp_cov':
        covariance_matrix = risk_models_.fix_nonpositive_semidefinite(
            risk_models_.exp_cov(prices, span=180)) / 252
    elif model == 'ledoit_wolf_constant_variance':
        covariance_matrix = risk_models_.fix_nonpositive_semidefinite(
            risk_models_.risk_matrix(prices, model)) / 252
    elif model == 'ledoit_wolf_single_factor':
        covariance_matrix = risk_models_.fix_nonpositive_semidefinite(
            risk_models_.risk_matrix(prices, model)) / 252
    elif model == 'ledoit_wolf_constant_correlation':
        covariance_matrix = risk_models_.fix_nonpositive_semidefinite(
            risk_models_.risk_matrix(prices, model)) / 252
    elif model == 'oracle_approximating':
        covariance_matrix = risk_models_.fix_nonpositive_semidefinite(
            risk_models_.risk_matrix(prices, model)) / 252
    else:
        raise NameError('You must input a risk model. Check spelling. Case-Sensitive.')
    if not isinstance(covariance_matrix, pd.DataFrame):
        covariance_matrix = pd.DataFrame(covariance_matrix, index=sample_cov.index, columns=sample_cov.columns).round(6)
    return covariance_matrix * 252
