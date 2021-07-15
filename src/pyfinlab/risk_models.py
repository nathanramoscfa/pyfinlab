import pandas as pd
import numpy as np
from portfoliolab.utils import RiskMetrics
from portfoliolab.estimators import RiskEstimators
from tqdm import tqdm

risk_met = RiskMetrics()
risk_estimators = RiskEstimators()

"""
Available covariance risk models in portfoliolab library. 
https://hudson-and-thames-portfoliolab-pro.readthedocs-hosted.com/en/latest/estimators/risk_estimators.html

These functions are designed to test covariance risk models from the portfoliolab library on out-of-sample data. 
To demonstrate, price time series data is split into sub-samples and each risk model's covariance matrix of each 
sub-sample is compared to the empirical covariance matrix of the next sequential sub-sample. Forecast error is measured 
by the Sum of Squared Errors (SSE) and averaged together for each risk model. The risk model with the lowest average
SSE is hypothetically the best risk model because it apparently had the lowest forecast error. 
"""

risk_models = [
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
    :param model: (str) The risk model to use. Should be one of:

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

    :param kde_bwidth: (float) The bandwidth of the kernel to fit KDE. (0.01 by default)
    :param basic_shrinkage: (float) Between 0 and 1. Coefficient in the convex combination for basic shrinkage.
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
    else:
        raise NameError('You must input a risk model. Check spelling. Case-Sensitive')
    if not isinstance(covariance_matrix, pd.DataFrame):
        covariance_matrix = pd.DataFrame(covariance_matrix, index=sample_cov.index, columns=sample_cov.columns).round(6)
    return covariance_matrix


def covariance_loop(prices, kde_bwidth, basic_shrinkage):
    """
    Calculates a covariance matrix using all available risk models.

    :param prices: (pd.DataFrame) Dataframe where each column is a series of prices for an asset.
    :param kde_bwidth: (float) The bandwidth of the kernel to fit KDE. (0.01 by default)
    :param basic_shrinkage: (float) Between 0 and 1. Coefficient in the convex combination for basic shrinkage.
                                    (0.1 by default)
    :return: (dict) Dictionary of each risk model's covariance matrix calculation.
    """

    covariances = {}
    for model in risk_models:
        covariances[model] = risk_model(prices, model, kde_bwidth, basic_shrinkage)
    return covariances


def volatility_loop(covariances, weights, freq=252):
    """
    Calculates a portfolio's volatility using all available risk models.

    :param covariances: (dict) Dictionary of each risk model's covariance matrix calculation. This can be obtained
                               using the covariance_loop(prices, kde_bwidth, basic_shrinkage) function.
    :param weights: (np.array) Vector of portfolio weights to each asset.
    :param freq: (int) Number of periods in a year. Default is 252 for daily price data.
    :return: (dict) Dictionary of each risk models covariance matrix calculation.
    """

    stdevs = {}
    for model in risk_models:
        stdevs[model] = np.sqrt(risk_met.calculate_variance(covariances[model], weights) * freq)
    stdevs = pd.Series(stdevs, name='Annualized Volatility').sort_values(ascending=False)
    print('The average portfolio volatility using all price data as one sample: {}'.format(stdevs.mean().round(6)))
    return stdevs


def split_sample(prices, num_of_samples, model):
    """
    Splits the data sample (prices) into the specified number of sub-samples and returns a dictionary of
    covariance matrices of each sub-sample. Example: Assuming 3 years of daily prices data and 36 is specified
    for number of samples, then the data will be split into 36 sub-samples of about 20 daily prices (about 1 month).
    The function then computes a covariance matrix on each sub-sample and returns two dictionaries of covariance
    matrices and prices.

    :param prices: (pd.DataFrame) Dataframe where each column is a series of prices for an asset.
    :param num_of_samples: (int) The number of times to split the prices DataFrame into sub-samples.
    :param model: (str) Risk model to use to calculate each sub-sample's covariance matrix.
    :return: (dict) Dictionary of dictionaries of daily prices and empirical covariance matrices for each sub-sample.
    """

    sample_size = int(len(prices) / num_of_samples)
    p = {}  # synthetic prices
    c = {}  # covariance matrices
    counter = 0
    for i in tqdm(range(0, num_of_samples)):
        p[i] = prices.iloc[counter: (counter + sample_size), :]
        c[i] = risk_model(p[i], model)
        counter += sample_size
    split_samples = {0: p, 1: c}
    print('Sub-sample size: {} days'.format(sample_size))
    print('Number of samples: {}'.format(num_of_samples))
    return split_samples


def compute_sse(split_samples):
    """
    Computes the Sum of Squared Errors (SSE) for each risk model on each sub-sample versus the next forward period
    sub-sample. This is intended to measure the accuracy of each risk model compared to the next period's
    empirical covariance matrix.

    :param split_samples: (dict) Dictionary of dictionaries of daily prices and empirical covariance matrices for each
                                 sub-sample.
    :return: (dict) Dictionary of the SSE of each risk model computed on all sub-samples.
    """

    prices = split_samples[0]
    covariances = split_samples[1]
    sum_squared_errors = {}
    for i in tqdm(range(0, len(covariances) - 1)):
        future_variance = np.diag(covariances[i + 1])
        sse_list = []
        sse_df = pd.DataFrame()
        for model in risk_models:
            variance = np.diag(risk_model(prices[i], model))
            sse_list.append(np.sum((variance - future_variance) ** 2))
            sse_df['risk_model'] = model
        variance_sse = pd.DataFrame(risk_models, columns=['risk_model'])
        variance_sse['sum_squared_error'] = sse_list
        sum_squared_errors[i] = variance_sse.sort_values(by='sum_squared_error')
    return sum_squared_errors


def sse_average(sum_squared_errors):
    """
    Averages all sub-sample Sum of Squared Errors (SSE) of each risk model's covariance matrix versus next
    period's sub-sample empirical covariance matrix computed across all sub-samples. This is intended to measure how
    close each risk model's forecast of next period's empirical covariance matrix.

    :param sum_squared_errors: (dict) Dictionary of dictionaries of daily prices and empirical covariance matrices for
                                      each sub-sample.
    :return: (dict) Dictionary of the SSE of each risk model computed on all sub-samples.
    """

    sse_avgs = {}
    for model in risk_models:
        sse_values = []
        for i in range(0, len(sum_squared_errors)):
            sample = sum_squared_errors.get(int(i))
            sse_values.append(sample.loc[sample['risk_model'] == model, 'sum_squared_error'].squeeze())
        sse_avgs[model] = np.mean(sse_values)
    sse_avgs = pd.DataFrame.from_dict(sse_avgs, orient='index')
    sse_avgs.columns = ['mean_sum_squared_error']
    sse_avgs = sse_avgs.sort_values(by='mean_sum_squared_error') * 10000000
    print('Risk model with the lowest mean sum of squared error: {}'.format(sse_avgs.index[0]))
    return sse_avgs


def win_count(sum_squared_errors):
    """
    Counts the number of a times a particular risk model had the lowest mean squared error (MSE) within an
    individual sub-sample.

    :param sum_squared_errors: (dict) Dictionary of dictionaries of daily prices and empirical covariance matrices for
                                      each sub-sample.
    :return: (dict) Dictionary of the SSE of each risk model computed on all sub-samples.
    """

    m = []
    count = []
    [m.append(sum_squared_errors[i].iloc[0, 0]) for i in range(0, len(sum_squared_errors) - 1)]
    [count.append(m.count(model)) for model in risk_models]
    count = pd.DataFrame(data=count, columns=['Win Count'], index=risk_models).sort_values(
        by='Win Count', ascending=False)
    print('The risk model with the most months with the lowest mean SSE: {}'.format(count.index[0]))
    print(count)
    return count
