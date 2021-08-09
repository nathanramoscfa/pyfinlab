import math
import ffn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from tqdm import tqdm
from patsy import dmatrices
from datetime import datetime, timedelta
from pyfinlab import data_api as api
from pyfinlab import risk_models as risk
from pypfopt import efficient_frontier, plotting

"""
These functions optimize portfolios and generate output dataframes, displays, and plots for further analysis and usage. 
Available risk and return models used to compute cov_matrix come from either PyPortfolioOpt (free open source) or Hudson 
& Thames' PortfolioLab (subscription required). Available objective functions come only from PyPortfolioOpt.  

Available risk models:

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
        
Available return models:

    PyPortfolioOpt
        - 'avg_historical_return',
        - 'ema_historical_return',
        - 'capm_return'
        
    PortfolioLab
        - 'simple_return',
        - 'mean_historical_return',
        - 'exponential_historical_return',
        
Available objective functions:

        PyPortfolioOpt
            - 'min_volatility',
            - 'max_sharpe',
            - 'max_quadratic_utility'
            - 'efficient_risk'
            - 'efficient_return'
"""

# Global variables
inputs = pd.read_excel('../data/portopt_inputs.xlsx', engine='openpyxl', sheet_name=None)
mapping = inputs.get('mapping').fillna('').sort_values(by='TICKER')
tickers = list(mapping.TICKER)
classification = mapping.iloc[:, :11].set_index('TICKER')
groups = list(classification.columns[1:])


def constraints(ticker_adj=1.0, upper_adj=1.0):
    """
    Returns dictionaries and tuples of ticker and sector-level constraints which are then input into the
    optimize_portfolio() function. The ticker-level maximum weighting constraints can be adjusted by a multiple
    of the ticker_adj parameter. The sector-level maximum weighting constraints can be adjusted by a multiple of the
    upper_adj parameter.

    :param ticker_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker
                               level. Defaults to 1.0.
    :param upper_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the sector
                              level. Defaults to 1.0.
    :return: (tuple) Returns tuple of ticker-level constraints as a list and sector-level constraints as tuples.
    """
    bounds = mapping[['TICKER', 'MIN', 'MAX']]
    bounds = [tuple(x) for x in bounds[['MIN', 'MAX']].to_numpy()]
    bounds = list(zip(
        list((min(a[0] * ticker_adj, 1) for a in bounds)),
        list((min(a[1] * ticker_adj, 1) for a in bounds))
    ))

    region_constraint = inputs.get('region')
    size_constraint = inputs.get('size')
    style_constraint = inputs.get('style')
    credit_constraint = inputs.get('credit')
    duration_constraint = inputs.get('duration')
    asset_constraint = inputs.get('asset_class')
    type_constraint = inputs.get('sec_type')
    holding_constraint = inputs.get('holding')
    sector_constraint = inputs.get('sector')

    region_lower = dict(zip(list(region_constraint['REGION']), list(region_constraint['MIN'])))
    region_upper = dict(zip(list(region_constraint['REGION']), list(region_constraint['MAX'])))
    region_mapper = dict(zip(list(classification.index), list(classification['REGION'])))
    size_lower = dict(zip(list(size_constraint['SIZE']), list(size_constraint['MIN'])))
    size_upper = dict(zip(list(size_constraint['SIZE']), list(size_constraint['MAX'])))
    size_mapper = dict(zip(list(classification.index), list(classification['SIZE'])))
    style_lower = dict(zip(list(style_constraint['STYLE']), list(style_constraint['MIN'])))
    style_upper = dict(zip(list(style_constraint['STYLE']), list(style_constraint['MAX'])))
    style_mapper = dict(zip(list(classification.index), list(classification['STYLE'])))
    credit_lower = dict(zip(list(credit_constraint['CREDIT']), list(credit_constraint['MIN'])))
    credit_upper = dict(zip(list(credit_constraint['CREDIT']), list(credit_constraint['MAX'])))
    credit_mapper = dict(zip(list(classification.index), list(classification['CREDIT'])))
    duration_lower = dict(zip(list(duration_constraint['DURATION']), list(duration_constraint['MIN'])))
    duration_upper = dict(zip(list(duration_constraint['DURATION']), list(duration_constraint['MAX'])))
    duration_mapper = dict(zip(list(classification.index), list(classification['DURATION'])))
    asset_lower = dict(zip(list(asset_constraint['ASSET_CLASS']), list(asset_constraint['MIN'])))
    asset_upper = dict(zip(list(asset_constraint['ASSET_CLASS']), list(asset_constraint['MAX'])))
    asset_mapper = dict(zip(list(classification.index), list(classification['ASSET_CLASS'])))
    type_lower = dict(zip(list(type_constraint['SECURITY_TYPE']), list(type_constraint['MIN'])))
    type_upper = dict(zip(list(type_constraint['SECURITY_TYPE']), list(type_constraint['MAX'])))
    type_mapper = dict(zip(list(classification.index), list(classification['SECURITY_TYPE'])))
    holding_lower = dict(zip(list(holding_constraint['HOLDING']), list(holding_constraint['MIN'])))
    holding_upper = dict(zip(list(holding_constraint['HOLDING']), list(holding_constraint['MAX'])))
    holding_mapper = dict(zip(list(classification.index), list(classification['HOLDING'])))
    sector_lower = dict(zip(list(sector_constraint['SECTOR']), list(sector_constraint['MIN'])))
    sector_upper = dict(zip(list(sector_constraint['SECTOR']), list(sector_constraint['MAX'])))
    sector_mapper = dict(zip(list(classification.index), list(classification['SECTOR'])))

    upper_list = [
        region_upper,
        size_upper,
        style_upper,
        credit_upper,
        duration_upper,
        asset_upper,
        type_upper,
        holding_upper,
        sector_upper
    ]

    for ele in upper_list:
        newList = list(np.clip([ele.get(key) * upper_adj for key in ele.keys()], 0, 1))
        for key, i in zip(ele.keys(), range(0, len(newList))):
            ele[key] = newList[i]

    return (
        bounds,
        region_lower, region_upper, region_mapper, size_lower, size_upper, size_mapper,
        style_lower, style_upper, style_mapper, credit_lower, credit_upper, credit_mapper,
        duration_lower, duration_upper, duration_mapper, asset_lower, asset_upper, asset_mapper,
        type_lower, type_upper, type_mapper, holding_lower, holding_upper, holding_mapper,
        sector_lower, sector_upper, sector_mapper
    )


def optimize_portfolio(
        exp_returns, cov_matrix,
        risk_model='sample_cov',
        return_model='avg_historical_return',
        obj_function='max_sharpe',
        target_volatility=0.01, target_return=0.2,
        risk_free_rate=0.02, risk_aversion=1, market_neutral=False,
        ticker_adj = 1.0, upper_adj = 1.0
):
    """
    Compute the optimal portfolio.

    :param exp_returns: (pd.Series) Expected returns for each asset.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param risk_model: (str) Optional, risk model used to compute cov_matrix from either PyPortfolioOpt (free open
                             source) or Hudson & Thames' PortfolioLab (subscription required). Defaults to 'sample_cov'.
    :param return_model: (str) Optional, return model used to compute the exp_returns from either PyPortfolioOpt (free
                               open source) or Hudson & Thames' PortfolioLab (subscription required). Defaults to
                               'avg_historical_return'.
    :param obj_function: (str) Objective function used in the portfolio optimization. Defaults to 'max_sharpe'.
    :param target_volatility: (float) Optional, the desired maximum volatility of the resulting portfolio. Required if
                                      objective function is 'efficient_risk', otherwise, parameter is ignored. Defaults
                                      to 0.01.
    :param target_return: (float) Optional, the desired return of the resulting portfolio. Required if objective
                                  function is 'efficient return', otherwise, parameter is ignored. Defaults to 0.2.
    :param risk_free_rate: (float) Optional, annualized risk-free rate, defaults to 0.02. Required if objective function
                                   is 'max_sharpe', otherwise, parameter is ignored.
    :param risk_aversion: (positive float) Optional, risk aversion parameter (must be greater than 0). Required if
                                           objective function is 'max_quadratic_utility'. Defaults to 1.
    :param market_neutral: (bool) Optional, if weights are allowed to be negative (i.e. short). Defaults to False.
    :param ticker_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker
                               level. Defaults to 1.0.
    :param upper_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the sector
                              level. Defaults to 1.0.
    :return: (tuple) Tuple of weightings (pd.DataFrame) and results (pd.DataFrame) showing risk, return, sharpe ratio
                     metrics.
    """
    # Generate constraints
    (
        bounds,
        region_lower, region_upper, region_mapper,
        size_lower, size_upper, size_mapper,
        style_lower, style_upper, style_mapper,
        credit_lower, credit_upper, credit_mapper,
        duration_lower, duration_upper, duration_mapper,
        asset_lower, asset_upper, asset_mapper,
        type_lower, type_upper, type_mapper,
        holding_lower, holding_upper, holding_mapper,
        sector_lower, sector_upper, sector_mapper
    ) = constraints(ticker_adj, upper_adj)

    # Empty DataFrames
    weightings = pd.DataFrame()
    results = pd.DataFrame()

    # Instantiate efficient_frontier class
    ef = efficient_frontier.EfficientFrontier(exp_returns, cov_matrix, bounds)

    # Add Sector Constraints
    ef.add_sector_constraints(region_mapper, region_lower, region_upper)
    ef.add_sector_constraints(size_mapper, size_lower, size_upper)
    ef.add_sector_constraints(style_mapper, style_lower, style_upper)
    ef.add_sector_constraints(credit_mapper, credit_lower, credit_upper)
    ef.add_sector_constraints(duration_mapper, duration_lower, duration_upper)
    ef.add_sector_constraints(asset_mapper, asset_lower, asset_upper)
    ef.add_sector_constraints(type_mapper, type_lower, type_upper)
    ef.add_sector_constraints(holding_mapper, holding_lower, holding_upper)
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

    # Objective Function
    if obj_function=='min_volatility':
        ef.min_volatility()
    elif obj_function=='max_sharpe':
        ef.max_sharpe(risk_free_rate)
    elif obj_function=='max_quadratic_utility':
        ef.max_quadratic_utility(risk_aversion, market_neutral)
    elif obj_function=='efficient_risk':
        ef.efficient_risk(target_volatility, market_neutral)
    elif obj_function=='efficient_return':
        ef.efficient_return(target_return, market_neutral)
    else:
        raise NotImplementedError('Check objective parameter. Double-check spelling.')

    # Compile the optimized weightings and performance results
    weights = ef.clean_weights(0.005)
    weights = pd.DataFrame.from_dict(weights, orient='index', columns=[int(1)]).round(4)
    performance = pd.DataFrame(ef.portfolio_performance(risk_free_rate=risk_free_rate)).round(4)
    weightings = pd.concat([weightings, weights], axis=1)
    results = pd.concat([results, performance], axis=1)
    results.columns = ['PORTFOLIO']
    results = results.rename(index={0: return_model, 1: risk_model, 2: 'sharpe_ratio'})
    weightings.index.name = 'TICKER'
    weightings = pd.merge(classification, weightings, on='TICKER')
    return weightings, results


def display_portfolio(portfolio, results):
    """
    Display the optimal portfolio.

    :param portfolio: (pd.DataFrame) Weightings DataFrame from optimize_portfolio() function.
    :param results: (pd.DataFrame) Results DataFrame from optimize_portfolio() function.
    :return: (pd.DataFrame) Displays the optimized portfolio.
    """
    portfolio.rename(columns={int(1): 'WEIGHTING'}, inplace=True)
    portfolio = portfolio[portfolio['WEIGHTING'] != 0].sort_values(by='WEIGHTING', ascending=False)
    print(results)
    return portfolio


def min_risk(
        exp_returns, cov_matrix,
        risk_model='sample_cov',
        return_model='avg_historical_return',
        obj_function='min_volatility',
        ticker_adj = 1.0, upper_adj = 1.0
):
    """
    Computes the minimum volatility of the minimum volatility portfolio.

    :param exp_returns: (pd.Series) Expected returns for each asset.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param risk_model: (str) Optional, risk model used to compute cov_matrix from either PyPortfolioOpt (free open
                             source) or Hudson & Thames' PortfolioLab (subscription required). Defaults to 'sample_cov'.
    :param return_model: (str) Optional, return model used to compute the exp_returns from either PyPortfolioOpt (free
                               open source) or Hudson & Thames' PortfolioLab (subscription required). Defaults to
                               'avg_historical_return'.
    :param obj_function: (str) Objective function used in the portfolio optimization. Defaults to 'min_volatility'.
    :param ticker_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker
                               level. Defaults to 1.0.
    :param upper_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the sector
                              level. Defaults to 1.0.
    :return: (float) Volatility of the minimum risk portfolio.
    """
    min_volatility = optimize_portfolio(
        exp_returns, cov_matrix,
        risk_model, return_model,
        obj_function,
        ticker_adj, upper_adj
    )
    return min_volatility[1].loc[risk_model].squeeze()


def max_risk(
        exp_returns, cov_matrix,
        risk_model='sample_cov',
        return_model='avg_historical_return',
        obj_function='efficient_risk',
        target_volatility=0.4,
        ticker_adj = 1.0, upper_adj = 1.0
):
    """
    Computes the maximum volatility of the maximum volatility portfolio. The default target volatility used to optimize
    the maximum volatility portfolio is 0.40.

    :param exp_returns: (pd.Series) Eexpected returns for each asset.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param risk_model: (str) Optional, risk model used to compute cov_matrix from either PyPortfolioOpt (free open
                             source) or Hudson & Thames' PortfolioLab (subscription required). Defaults to 'sample_cov'.
    :param return_model: (str) Optional, return model used to compute the exp_returns from either PyPortfolioOpt (free
                               open source) or Hudson & Thames' PortfolioLab (subscription required). Defaults to
                               'avg_historical_return'.
    :param obj_function: (str) Objective function used in the portfolio optimization. Defaults to 'efficient_risk'.
    :param target_volatility: (float) Optional, the desired maximum volatility of the resulting portfolio. Required if
                                      objective function is 'efficient_risk', otherwise, parameter is ignored. Defaults
                                      to 0.01.
    :param ticker_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker
                               level. Defaults to 1.0.
    :param upper_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the sector
                              level. Defaults to 1.0.
    :return: (float) Portfolio volatility of the maximum risk portfolio.
    """
    max_volatility = optimize_portfolio(
        exp_returns, cov_matrix,
        risk_model, return_model,
        obj_function,
        target_volatility,
        ticker_adj, upper_adj
    )
    return max_volatility[1].loc[risk_model].squeeze()


def compute_efficient_frontier(exp_returns, cov_matrix, risk_model, return_model, ticker_adj = 1.0, upper_adj = 1.0):
    """
    Computes 20 efficient frontier portfolios.

    :param exp_returns: (pd.Series) Expected returns for each asset.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param risk_model: (str) Optional, risk model used to compute cov_matrix from either PyPortfolioOpt (free open
                             source) or Hudson & Thames' PortfolioLab (subscription required). Defaults to 'sample_cov'.
    :param return_model: (str) Optional, return model used to compute the exp_returns from either PyPortfolioOpt (free
                               open source) or Hudson & Thames' PortfolioLab (subscription required). Defaults to
                               'avg_historical_return'.
    :param ticker_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker
                               level. Defaults to 1.0.
    :param upper_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the sector
                              level. Defaults to 1.0.
    :return: (tuple) Tuple of pd.DataFrames representing the optimized portfolio weightings and ex-ante risk, return,
                     and Sharpe ratio.
    """
    min_risk_ = min_risk(exp_returns, cov_matrix, risk_model, return_model, ticker_adj=ticker_adj, upper_adj=upper_adj)
    max_risk_ = max_risk(exp_returns, cov_matrix, risk_model, return_model, ticker_adj=ticker_adj, upper_adj=upper_adj)
    cash_weightings = pd.DataFrame()
    results = pd.DataFrame()
    counter = 1
    for i in tqdm(np.linspace(min_risk_ + .001, max_risk_, 20).round(4)):
        optimized_portfolio, optimized_performance = optimize_portfolio(
            exp_returns, cov_matrix,
            risk_model, return_model,
            obj_function='efficient_risk',
            target_volatility=i,
            ticker_adj=ticker_adj, upper_adj=upper_adj
        )
        cash_weighting = optimized_portfolio[int(1)]
        cash_weighting.name = counter
        cash_weightings = pd.concat([cash_weightings, cash_weighting], axis=1)
        result = optimized_performance
        result.columns = [counter]
        results = pd.concat([results, result], axis=1)
        counter += 1
    results.index.name = 'efficient_frontier'
    cash_weightings.fillna(0, inplace=True)
    optimized_portfolios = pd.concat([classification, cash_weightings], axis=1).fillna(0)
    return optimized_portfolios, results


def focus(optimized_portfolios, group):
    """
    Computes group weightings of efficient frontier portfolios. Portfolio holdings can be grouped by:
        - 'SIZE': Market cap size of common stocks and underlying holdings in equity ETFs.
            * example: large cap, mid cap, small cap
        - 'SECTOR': Equity sector of common stocks and underlying holdings in equity ETFs.
            * example: technology, consumer staples, healthcare
        - 'STYLE': Investment style of common stocks and underlying holdings in equity ETFs.
            * example: value, blend, growth
        - 'REGION': Geographic region of all portfolio holdings.
            * example: U.S., Developed Markets, Emerging Markets
        - 'HOLDING': Core versus satellite specification for common stocks and underlying holdings in equity ETFs.
            * example: core, satellite...helpful when managing a core/satellite strategy
        - 'ASSET_CLASS': Asset class of all portfolio holdings.
            * example: equity, bond
        - 'SECURITY_TYPE': Security type of all portfolio holdings.
            * example: common stocks, ETFs
        - 'DURATION': Duration for bond ETF underlying holdings.
            * example: long, intermediate, short
        - 'CREDIT': Credit quality of bond ETF underlying holdings.
            * example: high, mid, low

    :param optimized_portfolios: (pd.DataFrame) Weightings for efficient frontier portfolios.
    :param group: (str) Group to groupby portfolio holdings. Can be one of: 'SIZE', 'SECTOR', 'STYLE', 'REGION', 'HOLDING',
                        'ASSET_CLASS', 'SECURITY_TYPE', 'DURATION', 'CREDIT'
    :return: (pd.DataFrame) Portfolio weightings weighted by the group.
    """
    optimized_portfolios = optimized_portfolios[(optimized_portfolios.T != 0).any()]
    focus = optimized_portfolios.groupby([group][0]).sum()
    strings = focus.index
    strings = strings.to_list()

    def unique_list(l):
        ulist = []
        [ulist.append(x) for x in l if x not in ulist]
        return ulist

    list1 = []
    for string in strings:
        a = string
        a = ' '.join(unique_list(a.split()))
        list1.append(a)
    focus.index = list1
    return focus


def cash_focus(optimized_portfolios):
    """
    Computes the cash weighting of all portfolios for all available groups.

    :param optimized_portfolios: (pd.DataFrame) Weightings for efficient frontier portfolios.
    :return: (pd.DataFrame) Cash-weightings weighted by group for efficient frontier portfolios.
    """
    cash_focus = {}
    for group_name in groups:
        cash_focus[group_name] = focus(optimized_portfolios, group_name)
    return cash_focus


def risk_weightings(optimized_portfolios, cov_matrix):
    """
    Computes the risk weightings of all portfolios.

    :param optimized_portfolios: (pd.DataFrame) Weightings for efficient frontier portfolios.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset.
    :return: (pd.DataFrame) Risk-weightings for efficient frontier portfolios.
    """
    cash_weightings = optimized_portfolios.iloc[:, 10:]
    risk_weightings = pd.DataFrame(index=tickers)
    for i in range(1, cash_weightings.shape[1] + 1):
        try:
            w = cash_weightings[i]
        except KeyError:
            w = cash_weightings['WEIGHTING']
        pvar = np.dot(w.T, np.dot(cov_matrix, w))
        pvol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        pvolw = ((np.dot(w, cov_matrix)) / pvar) * w
        risk_weightings = pd.concat([risk_weightings, pvolw], axis=1)
    risk_weightings = pd.concat([classification, risk_weightings.round(4)], axis=1)
    return risk_weightings


def risk_focus(optimized_portfolios, cov_matrix):
    """
    Computes the risk weightings of all portfolios.

    :param optimized_portfolios: (pd.DataFrame) Weightings for efficient frontier portfolios.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset.
    :return: (pd.DataFrame) Risk-weightings weighted by group for efficient frontier portfolios.
    """
    weightings = risk_weightings(optimized_portfolios, cov_matrix)
    risk_focus = {}
    for group_name in groups:
        risk_focus[group_name] = focus(weightings, group_name)
    return risk_focus


def compile_focus_stats(optimized_portfolios, cov_matrix, focus='cash'):
    """
    Computes the risk weightings of all portfolios.

    :param optimized_portfolios: (pd.DataFrame) Weightings for efficient frontier portfolios.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset.
    :param focus: (str) Optional, select whether to show cash or risk focus. Can only be 'cash' or 'risk'. Defaults to 'cash'.
    :return: (pd.DataFrame) Cash- or risk-weightings weighted by group for efficient frontier portfolios.
    """
    cash_focus_df = pd.DataFrame(columns=optimized_portfolios.columns[10:])
    risk_focus_df = pd.DataFrame(columns=optimized_portfolios.columns[10:])
    for category in groups:
        cash_focus_df = cash_focus_df.append(cash_focus(optimized_portfolios).get(category))
        risk_focus_df = risk_focus_df.append(risk_focus(optimized_portfolios, cov_matrix).get(category))
    cash_focus_df.drop_duplicates(inplace=True)
    risk_focus_df.drop_duplicates(inplace=True)
    cash_focus_df.index.name = 'Cash Weighting'
    risk_focus_df.index.name = 'Risk Weighting'
    return cash_focus_df if focus == 'cash' else risk_focus_df


def compute_ticker_vols(tickers, cov_matrix):
    """
    Computes the volatility of each ticker.

    :param tickers: (list) List of tickers.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset.
    :return: (pd.Series) Volatility for each ticker.
    """
    count = 0
    ticker_stds = []
    weight_vector = [1] + [0] * (len(tickers) - 1)
    while count < len(tickers):
        ticker_stds.append(np.dot(weight_vector, np.dot(cov_matrix, weight_vector)).round(4))
        try:
            weight_vector[count], weight_vector[count + 1] = 0, 1
        except IndexError:
            break
        count += 1
    ticker_stds = pd.Series(ticker_stds, tickers)
    return ticker_stds


def eff_frontier_plot(tickers, cov_matrix, exp_returns, results, figsize=(12, 6), save=False, show=True):
    """
    Plots the efficient frontier and individual assets.

    :param tickers: (list) List of tickers.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param exp_returns: (pd.Series) Expected returns for each asset.
    :param results: (pd.DataFrame) Risk, return, and sharpe ratio for all efficient frontier portfolios. Input the
                                   results DataFrame computed using the optimize_portfolio() function.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the
                                   ticker level. Defaults to (12, 6).
    :param save: (bool) Optional, width, height in inches. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :return: (fig) Plot of efficient frontier and individual assets.
    """
    ticker_vols = compute_ticker_vols(tickers, cov_matrix)
    portfolio_volatilities = list(results.iloc[1:2, :].squeeze())
    returns = list(results.iloc[:1, :].squeeze())
    sharpe_ratios = list(results.iloc[2:3, :].squeeze())
    ticker_volatilities = list(ticker_vols.values)
    max_sharpe_ratio_index = sharpe_ratios.index(max(sharpe_ratios))
    min_volatility_index = portfolio_volatilities.index(min(portfolio_volatilities))
    scatter_plot_index = ticker_volatilities.index(min(ticker_volatilities))
    plt.figure(figsize=figsize)
    figure = plt.plot(portfolio_volatilities, returns, c='black', label='Constrained Efficient Frontier')
    plt.scatter(portfolio_volatilities[max_sharpe_ratio_index],
                returns[max_sharpe_ratio_index],
                marker='*',
                color='g',
                s=400,
                label='Maximum Sharpe Ratio')
    plt.scatter(portfolio_volatilities[min_volatility_index],
                returns[min_volatility_index],
                marker='*',
                color='r',
                s=400,
                label='Minimum Volatility')
    plt.scatter(np.sqrt(np.diag(cov_matrix)),
                exp_returns,
                marker='.',
                color='black',
                s=100,
                label='Individual Assets')
    plt.title('Efficient Frontier with Individual Assets')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.legend(loc='upper left')
    if save==True: plt.savefig(
        '../charts/efficient_frontier_{}.png'.format(datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if show==False: plt.close()


def monte_carlo_frontier(
        cov_matrix, exp_returns, ticker_adj = 1.0, upper_adj = 1.0, figsize=(11, 5), save=False, show=True):
    """
    Plots the efficient frontier and individual assets.

    :param tickers: (list) List of tickers.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param exp_returns: (pd.Series) Expected returns for each asset.
    :param ticker_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker
                               level. Defaults to 1.0.
    :param upper_adj: (float) Optional, multiple by which to multiply the maximum weighting constraints at the sector
                              level. Defaults to 1.0.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
                                   Defaults to (11, 5).
    :param save: (bool) Optional, width, height in inches. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :return: (fig) Plot of efficient frontier and individual assets.
    """
    (
        bounds,
        region_lower, region_upper, region_mapper,
        size_lower, size_upper, size_mapper,
        style_lower, style_upper, style_mapper,
        credit_lower, credit_upper, credit_mapper,
        duration_lower, duration_upper, duration_mapper,
        asset_lower, asset_upper, asset_mapper,
        type_lower, type_upper, type_mapper,
        holding_lower, holding_upper, holding_mapper,
        sector_lower, sector_upper, sector_mapper
    ) = constraints(ticker_adj, upper_adj)
    fig, ax = plt.subplots(figsize=figsize)
    ef = efficient_frontier.EfficientFrontier(exp_returns, cov_matrix, constraints()[0])
    ef.add_sector_constraints(region_mapper, region_lower, region_upper)
    ef.add_sector_constraints(size_mapper, size_lower, size_upper)
    ef.add_sector_constraints(style_mapper, style_lower, style_upper)
    ef.add_sector_constraints(credit_mapper, credit_lower, credit_upper)
    ef.add_sector_constraints(duration_mapper, duration_lower, duration_upper)
    ef.add_sector_constraints(asset_mapper, asset_lower, asset_upper)
    ef.add_sector_constraints(type_mapper, type_lower, type_upper)
    ef.add_sector_constraints(holding_mapper, holding_lower, holding_upper)
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    ax.get_lines()[0].set_color("black")
    ax.get_lines()[0].set_linewidth(2.0)

    # Find the tangency portfolio
    ef.max_sharpe()
    ret_tangent, std_tangent, _ = ef.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=400, c="red", label="Max Sharpe")

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(len(exp_returns)), n_samples)
    rets = w.dot(exp_returns)
    stds = np.sqrt(np.diag(w @ cov_matrix @ w.T))
    sharpes = rets / stds
    df = pd.DataFrame({'Expected Return': rets,
                       'Expected Volatility': stds,
                       'Sharpe Ratio': sharpes})
    df.plot(kind='scatter', x='Expected Volatility', y='Expected Return', c='Sharpe Ratio', cmap='viridis_r', ax=ax)

    # Output
    ax.set_title("Efficient Frontier with Random Portfolios")
    ax.legend(['Constrained Efficient Frontier', "Max Sharpe"], loc='upper left')
    if save==True: plt.savefig(
        '../charts/monte_carlo_frontier_{}.png'.format(datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if show==False: plt.close()


def name(api_source='yfinance', ticker=['SPY']):
    """
    Downloads the name of each ticker in ticker list. Only 1 ticker can be downloaded at a time if using yfinance as
    api_source. There is not such limit if api_source is bloomberg.

    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :param ticker: (list) List of single ticker if using yfinance, or, multiple tickers if using bloomberg.
    :return: (str) Name of the ticker, or, tickers.
    """
    if api_source=='yfinance':
        name = api.current_equity_data(ticker, ['longName'], api_source).squeeze()
    elif api_source=='bloomberg':
        name = api.current_equity_data(ticker, ['LONG_COMP_NAME'], api_source).squeeze()
    else:
        raise ValueError('api_source must be set to either yfinance or bloomberg')
    return name


def backtest_timeseries(prices, optimized_portfolios, api_source='yfinance', benchmark_ticker='SPY', pv=1000000):
    """
    Backtest function to compute the timeseries backtest of all efficient frontier portfolios.

    :param prices: (pd.DataFrame) Dataframe where each column is a series of prices for an asset.
    :param optimized_portfolios: (pd.DataFrame) Weightings for efficient frontier portfolios. Comes from the
                                                compute_efficient_frontier() function output, DataFrame called 'optimized portfolio.'
    :param api_source: (str) Optional, API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Defaults to yfinance.
    :param benchmark_ticker: (str) Optional, benchmark ticker. Takes only one ticker. Defaults to 'SPY'.
    :param pv: (int) Initial portfolio value in USD. Defaults to 1000000.
    :return: (pd.DateFrame) Backtest timeseries of portfolio values.
    """
    optimized_portfolios = optimized_portfolios.iloc[:, 10:]
    p0 = prices.iloc[:1, :].values.squeeze()  # initial prices at day 0
    weights, shares, values = {}, {}, {}
    for i in range(0, 20):
        weights[i + 1] = optimized_portfolios.iloc[:, i:i + 1].values.squeeze()
        shares[i + 1] = np.floor((weights[i + 1] * pv) / p0)
        portfolio_values = prices.multiply(shares[i + 1])
        portfolio_values['Cash Position'] = pv - portfolio_values.sum(axis=1)[0]
        portfolio_values = portfolio_values.sum(axis=1)
        values[i + 1] = portfolio_values
    portfolio_values = pd.concat([values.get(i) for i in range(1, 21)], axis=1) / pv
    portfolio_values.columns = list(range(1, 21))
    start_date = portfolio_values.index[0].strftime('%Y-%m-%d')
    end_date = (portfolio_values.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
    benchmark_prices = api.price_history([benchmark_ticker], start_date, end_date, api_source)
    benchmark_name = name(api_source, benchmark_ticker)
    benchmark_values = benchmark_prices.copy().squeeze()
    benchmark_values = benchmark_values / benchmark_values[0]
    benchmark_values.name = benchmark_ticker
    portfolio_values = pd.concat([portfolio_values, benchmark_values], axis=1)
    return portfolio_values


def days(start_date, end_date):
    """
    Counts the number of days between start and end dates. Helpful for splicing backtest timeseries pd.DataFrame.

    :param start_date: (str) Start date string or datetime. Date format 'MM-DD-YYYY'.
    :param end_date: (str) End date string or datetime. Date format 'MM-DD-YYYY'.
    :return: (int) Number of days between start_date and end_date.
    """
    date_format = "%m-%d-%Y"
    a = datetime.strptime(start_date, date_format)
    b = datetime.strptime(end_date, date_format)
    delta = b - a
    return delta.days


def backtest_chart1(backtest_timeseries, start_date, end_date, figsize=(15, 9), save=False, show=True, yscale='linear'):
    """
    Plots the performance for all efficient frontier portfolios.

    :param backtest_timeseries: (pd.DataFrame) Ex-post performance of efficient frontier portfolios.
    :param start_date: (str) Start date string or datetime. Date format 'MM-DD-YYYY'.
    :param end_date: (str) End date string or datetime. Date format 'MM-DD-YYYY'.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
                                   Defaults to (11, 5).
    :param save: (bool) Optional, width, height in inches. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :param yscale: (str) Optional, axis scale type to apply. Choose from 'linear', 'log', 'symlog', or 'logit'. Defaults
                         to linear.
    :return: (fig) Plot of performance for all efficient frontier portfolios.
    """
    backtest_timeseries = backtest_timeseries.loc[start_date: end_date]
    backtest_timeseries = backtest_timeseries / backtest_timeseries.iloc[0]
    x_values = backtest_timeseries.index
    y_values = backtest_timeseries.iloc[:, :-1]
    fig = plt.gcf()
    fig.set_size_inches(figsize)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.plot(x_values, y_values)
    plt.plot(backtest_timeseries.iloc[:, -1:], 'k')
    plt.legend(backtest_timeseries.columns)
    plt.title("\nBacktest of Portfolios {} to {}".format(
        x_values[0].strftime('%m-%d-%Y'),
        x_values[-1].strftime('%m-%d-%Y')))
    plt.yscale(yscale)
    plt.xlabel('Date')
    plt.ylabel('Value of $1 Investment')
    plt.gcf().autofmt_xdate()
    if save==True: plt.savefig(
        '../charts/backtest_chart1_{}.png'.format(datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if show==False: plt.close()


def backtest_chart2(
        backtest_timeseries, start_date, end_date, portfolios=[2, 5, 9, 13, 17],
        api_source='yfinance', figsize=(15, 9), save=False, show=True, yscale='linear'):
    """
    Plots the performance for 5 selected efficient frontier portfolios.

    :param backtest_timeseries: (pd.DataFrame) Ex-post performance of efficient frontier portfolios.
    :param start_date: (str) Start date string or datetime. Date format 'MM-DD-YYYY'.
    :param end_date: (str) End date string or datetime. Date format 'MM-DD-YYYY'.
    :param portfolios: (list) Optional, list of 5 integers anywhere from 1-20 corresponding to portfolios along the efficient
                              frontier in ascending order of risk. Portfolio 1 has the lowest risk while Portfolio 20
                              has the highest risk.
    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
                                   Defaults to (11, 5).
    :param save: (bool) Optional, width, height in inches. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :param yscale: (str) Optional, axis scale type to apply. Choose from 'linear', 'log', 'symlog', or 'logit'. Defaults
                         to linear.
    :return: (fig) Plot of performance for all efficient frontier portfolios.
    """

    backtest_timeseries = backtest_timeseries.loc[start_date: end_date]
    backtest_timeseries = backtest_timeseries / backtest_timeseries.iloc[0]
    plt.figure(figsize=figsize)
    backtest_timeseries[portfolios[0]].plot(color='#00B050', label='Portfolio {}'.format(portfolios[0]))
    backtest_timeseries[portfolios[1]].plot(color='#92D050', label='Portfolio {}'.format(portfolios[1]))
    backtest_timeseries[portfolios[2]].plot(color='#00B0F0', label='Portfolio {}'.format(portfolios[2]))
    backtest_timeseries[portfolios[3]].plot(color='#FFC000', label='Portfolio {}'.format(portfolios[3]))
    backtest_timeseries[portfolios[4]].plot(color='#FF0000', label='Portfolio {}'.format(portfolios[4]))
    backtest_timeseries[backtest_timeseries.columns[-1]].plot(
        color='#000000',
        label='{} ({})'.format(name(api_source, backtest_timeseries.columns[-1]), backtest_timeseries.columns[-1]))
    plt.yscale(yscale)
    plt.legend()
    plt.title("\nBacktest of Portfolios {} to {}".format(
        backtest_timeseries.index[0].strftime('%m-%d-%Y'),
        backtest_timeseries.index[-1].strftime('%m-%d-%Y')))
    plt.xlabel('Date')
    plt.ylabel('Value of $1 Investment')
    if save==True: plt.savefig(
        '../charts/backtest_chart2_{}.png'.format(datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if show==False: plt.close()


def backtest_linechart(
        backtest_timeseries, start_date, end_date, portfolios=[2, 5, 9, 13, 17],
        api_source='yfinance', chart=1, figsize=(15, 9), save=False, show=True, yscale='linear'):
    """
    Plots the performance for 5 selected efficient frontier portfolios.

    :param backtest_timeseries: (pd.DataFrame) Ex-post performance of efficient frontier portfolios.
    :param start_date: (str) Start date string or datetime. Date format 'MM-DD-YYYY'.
    :param end_date: (str) End date string or datetime. Date format 'MM-DD-YYYY'.
    :param portfolios: (list) Optional, list of 5 integers anywhere from 1-20 corresponding to portfolios along the efficient
                              frontier in ascending order of risk. Portfolio 1 has the lowest risk while Portfolio 20
                              has the highest risk.
    :param api_source: (str) API source to pull data from. Choose from 'yfinance' or 'bloomberg'. Default is yfinance.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
                                   Defaults to (11, 5).
    :param save: (bool) Optional, width, height in inches. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :param yscale: (str) Optional, axis scale type to apply. Choose from 'linear', 'log', 'symlog', or 'logit'. Defaults
                         to linear.
    :return: (fig) Plot of performance for all efficient frontier portfolios.
    """
    if chart==1:
        backtest_chart1(backtest_timeseries, start_date, end_date, figsize, save, show, yscale)
    elif chart==2:
        backtest_chart2(
            backtest_timeseries, start_date, end_date, portfolios,
            api_source, figsize, save, show, yscale)
    else:
        raise ValueError('chart parameter must be 1 or 2')
