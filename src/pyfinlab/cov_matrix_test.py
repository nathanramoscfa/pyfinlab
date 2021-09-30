import pandas as pd
import numpy as np
from portfoliolab.utils import RiskMetrics
from pyfinlab import risk_models as rm
from tqdm import tqdm

"""
These functions are designed to test covariance risk models from the portfoliolab library on out-of-sample data. 
To demonstrate, price time series data is split into sub-samples and each risk model's covariance matrix of each 
sub-sample is compared to the empirical covariance matrix of the next sequential sub-sample. Forecast error is measured 
by the Sum of Squared Errors (SSE) and averaged together for each risk model. The risk model with the lowest average
SSE is hypothetically the best risk model because it apparently had the lowest forecast error. 
"""


# Bring in all risk models
risk_models = rm.risk_models
risk_met = RiskMetrics()


def covariance_loop(prices, kde_bwidth=0.01, basic_shrinkage=0.1):
    """
    Calculates a covariance matrix using all available risk models.

    :param prices: (pd.DataFrame) Dataframe where each column is a series of prices for an asset.
    :param kde_bwidth: (float) Bandwidth of the kernel to fit KDE. (0.01 by default)
    :param basic_shrinkage: (float) Between 0 and 1. Coefficient in the convex combination for basic shrinkage.
                                    (0.1 by default)
    :return: (dict) Dictionary of each risk model's covariance matrix calculation.
    """

    covariances = {}
    for model in risk_models:
        covariances[model] = rm.risk_model(prices, model, kde_bwidth, basic_shrinkage)
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
    :param num_of_samples: (int) Number of times to split the prices DataFrame into sub-samples.
    :param model: (str) Risk model to use to calculate each sub-sample's covariance matrix.
    :return: (dict) Dictionary of dictionaries of daily prices and empirical covariance matrices for each sub-sample.
    """

    sample_size = int(len(prices) / num_of_samples)
    p = {}  # synthetic prices
    c = {}  # covariance matrices
    counter = 0
    for i in tqdm(range(0, num_of_samples)):
        p[i] = prices.iloc[counter: (counter + sample_size), :]
        c[i] = rm.risk_model(p[i], model)
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
            variance = np.diag(rm.risk_model(prices[i], model))
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
    sse_avgs = sse_avgs.squeeze()
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
