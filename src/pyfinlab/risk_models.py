import pandas as pd
import numpy as np
from portfoliolab.utils import RiskMetrics
from portfoliolab.estimators import RiskEstimators
from pypfopt import risk_models as risk_models_

"""
Available covariance risk models in PortfolioLab library. 
https://hudson-and-thames-portfoliolab-pro.readthedocs-hosted.com/en/latest/estimators/risk_estimators.html

Available covariance risk models in PyPortfolioOpt library. 
https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#

These functions bring together all covariance matrix risk models from PortfolioLab and PyPortfolioOpt into one
function for ease of use.  
"""
risk_met = RiskMetrics()
risk_estimators = RiskEstimators()

risk_models = [
    # PyPortfolioOpt
    'sample_cov',
    'semicovariance',
    'exp_cov',
    'ledoit_wolf_constant_variance',
    'ledoit_wolf_single_factor',
    'ledoit_wolf_constant_correlation',
    'oracle_approximating',

    # PortfolioLab
    'sample_covariance',
    'minimum_covariance_determinant',
    'empirical_covariance',
    'shrinked_covariance_basic',
    'shrinked_covariance_lw',
    'shrinked_covariance_oas',
    'semi_covariance',
    'exponential_covariance',
    'constant_residual_eigenvalues_denoised',
    'constant_residual_spectral_denoised',
    'targeted_shrinkage_denoised',
    'targeted_shrinkage_detoned',
    'constant_residual_detoned',
    'hierarchical_filtered_complete',
    'hierarchical_filtered_single',
    'hierarchical_filtered_avg'
]


def risk_model(prices, model, kde_bwidth=0.01, basic_shrinkage=0.1):
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

        PortfolioLab
            - 'sample_covariance',
            - 'minimum_covariance_determinant',
            - 'empirical_covariance',
            - 'shrinked_covariance_basic',
            - 'shrinked_covariance_lw',
            - 'shrinked_covariance_oas',
            - 'semi_covariance',
            - 'exponential_covariance',
            - 'constant_residual_eigenvalues_denoised',
            - 'constant_residual_spectral_denoised',
            - 'targeted_shrinkage_denoised',
            - 'targeted_shrinkage_detoned',
            - 'constant_residual_detoned',
            - 'hierarchical_filtered_complete',
            - 'hierarchical_filtered_single',
            - 'hierarchical_filtered_avg'

    :param kde_bwidth: (float) Optional, bandwidth of the kernel to fit KDE. (0.01 by default)
    :param basic_shrinkage: (float) Optional, between 0 and 1. Coefficient in the convex combination for basic shrinkage.
                                    (0.1 by default)
    :return: (pd.DataFrame) Estimated covariance matrix.
    """
    tn_relation = prices.shape[0] / prices.shape[1]
    sample_cov = prices.pct_change().dropna().cov()
    empirical_cov = pd.DataFrame(risk_estimators.empirical_covariance(prices, price_data=True),
                                 index=sample_cov.index, columns=sample_cov.columns)
    empirical_corr = pd.DataFrame(risk_estimators.cov_to_corr(empirical_cov ** 2),
                                  index=sample_cov.index, columns=sample_cov.columns)
    std = np.diag(empirical_cov) ** (1 / 2)
    if model == 'sample_covariance':
        return prices.pct_change().dropna().cov()
    elif model == 'minimum_covariance_determinant':
        covariance_matrix = risk_estimators.minimum_covariance_determinant(prices, price_data=True)
    elif model == 'empirical_covariance':
        covariance_matrix = risk_estimators.empirical_covariance(prices, price_data=True)
    elif model == 'shrinked_covariance_basic':
        covariance_matrix = risk_estimators.shrinked_covariance(
            prices, price_data=True, shrinkage_type='basic', basic_shrinkage=basic_shrinkage)
    elif model == 'shrinked_covariance_lw':
        covariance_matrix = risk_estimators.shrinked_covariance(
            prices, price_data=True, shrinkage_type='lw', basic_shrinkage=basic_shrinkage)
    elif model == 'shrinked_covariance_oas':
        covariance_matrix = risk_estimators.shrinked_covariance(
            prices, price_data=True, shrinkage_type='oas', basic_shrinkage=basic_shrinkage)
    elif model == 'semi_covariance':
        covariance_matrix = risk_estimators.semi_covariance(prices, price_data=True, threshold_return=0)
    elif model == 'exponential_covariance':
        covariance_matrix = risk_estimators.exponential_covariance(prices, price_data=True, window_span=60)
    elif model == 'constant_residual_eigenvalues_denoised':
        covariance_matrix = risk_estimators.denoise_covariance(
            empirical_cov, tn_relation, denoise_method='const_resid_eigen', detone=False, kde_bwidth=kde_bwidth)
    elif model == 'constant_residual_spectral_denoised':
        covariance_matrix = risk_estimators.denoise_covariance(empirical_cov, tn_relation, denoise_method='spectral')
    elif model == 'targeted_shrinkage_denoised':
        covariance_matrix = risk_estimators.denoise_covariance(
            empirical_cov, tn_relation, denoise_method='target_shrink', detone=False, kde_bwidth=kde_bwidth)
    elif model == 'targeted_shrinkage_detoned':
        covariance_matrix = risk_estimators.denoise_covariance(
            empirical_cov, tn_relation, denoise_method='target_shrink', detone=True, kde_bwidth=kde_bwidth)
    elif model == 'constant_residual_detoned':
        covariance_matrix = risk_estimators.denoise_covariance(
            empirical_cov, tn_relation, denoise_method='const_resid_eigen', detone=True, market_component=1,
            kde_bwidth=kde_bwidth)
    elif model == 'hierarchical_filtered_complete':
        covariance_matrix = risk_estimators.corr_to_cov(risk_estimators.filter_corr_hierarchical(
            empirical_corr.to_numpy(), method='complete', draw_plot=False), std)
    elif model == 'hierarchical_filtered_single':
        covariance_matrix = risk_estimators.corr_to_cov(risk_estimators.filter_corr_hierarchical(
            empirical_corr.to_numpy(), method='single', draw_plot=False), std)
    elif model == 'hierarchical_filtered_avg':
        covariance_matrix = risk_estimators.corr_to_cov(risk_estimators.filter_corr_hierarchical(
            empirical_corr.to_numpy(), method='average', draw_plot=False), std)
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
