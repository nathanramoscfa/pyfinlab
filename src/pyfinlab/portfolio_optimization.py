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
from pypfopt import efficient_frontier, plotting, objective_functions

"""
These functions optimize portfolios and generate output dataframes, displays, and plots for further analysis and usage. 
Available risk and return models used to compute cov_matrix come from either PyPortfolioOpt (free open source) or Hudson 
& Thames' PortfolioLab (subscription required). Available objective functions come only from PyPortfolioOpt.  

Available risk models:

        - 'sample_cov',
        - 'semicovariance',
        - 'exp_cov',
        - 'ledoit_wolf_constant_variance',
        - 'ledoit_wolf_single_factor'
        - 'ledoit_wolf_constant_correlation',
        - 'oracle_approximating'
        
Available return models:

        - 'avg_historical_return',
        - 'ema_historical_return',
        - 'capm_return'
        
Available objective functions:

            - 'min_volatility',
            - 'max_sharpe',
            - 'max_quadratic_utility'
            - 'efficient_risk'
            - 'efficient_return'
"""

# Global variables
inputs = pd.read_excel('../data/portopt_inputs.xlsx', engine='openpyxl', sheet_name=None)
mapping = inputs.get('mapping').sort_values(by='TICKER').dropna(how='all').fillna('')
tickers = list(mapping.TICKER.values)
classification = mapping.iloc[:, :11].set_index('TICKER')
classification = classification.replace({
    'Non-Equity': '',
    'Non-Specified': '',
    'Non-Fixed Income': ''
})
groups = list(classification.columns[1:])


def constraints(cov_matrix, restricted=False, banned=False, allow_cat2=False):
    """
    Returns dictionaries and tuples of ticker and sector-level constraints which are then input into the
    optimize_portfolio() function.

    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset.
    :param restricted: (bool) Optional, filters out tickers on the "restricted" tab in ('../data/portopt_inputs.xlsx'). Default is False.
    :param banned: (bool) Optional, filters out tickers on the "banned" tab in ('../data/portopt_inputs.xlsx'). Default is False.
    :return: (tuple) Returns tuple of ticker-level constraints as a list and sector-level constraints as tuples.
    """
    bounds = mapping[['TICKER', 'MIN', 'MAX']]
    if restricted==True:
        restricted_list = pd.read_excel('../data/portopt_inputs.xlsx', engine='openpyxl', sheet_name='restricted')[
            ['Symbol', 'Prohibition Reason', 'Restriction']].dropna().drop_duplicates(subset=['Symbol'])
        if allow_cat2==True:
            restricted_list = restricted_list[
                (restricted_list['Prohibition Reason']!='Category 2A ETF') &
                (restricted_list['Prohibition Reason']!='Category 2B ETF')
            ]
        restricted_tickers = list(restricted_list.Symbol)
        bounds = bounds[~bounds['TICKER'].isin(restricted_tickers)]
    if banned==True:
        banned_list = pd.read_excel('../data/portopt_inputs.xlsx', engine='openpyxl', sheet_name='banned')[
            ['Symbol', 'Prohibition Reason']].dropna().drop_duplicates(subset=['Symbol'])
        banned_list = list(banned_list.Symbol)
        bounds = bounds[~bounds['TICKER'].isin(banned_list)]
    keys = list(bounds.TICKER.values)
    bounds = bounds[bounds['TICKER'].isin(cov_matrix.index)].drop_duplicates()
    bounds = [tuple(x) for x in bounds[['MIN', 'MAX']].to_numpy()]

    region_constraint = inputs.get('region')
    size_constraint = inputs.get('size')
    style_constraint = inputs.get('style')
    credit_constraint = inputs.get('credit')
    duration_constraint = inputs.get('duration')
    asset_constraint = inputs.get('asset_class')
    type_constraint = inputs.get('sec_type')
    holding_constraint = inputs.get('holding')
    sector_constraint = inputs.get('sector')

    mapper = classification[classification.index.isin(cov_matrix.index)]

    region_lower = dict(zip(list(region_constraint['REGION']), list(region_constraint['MIN'])))
    region_upper = dict(zip(list(region_constraint['REGION']), list(region_constraint['MAX'])))
    region_mapper = dict(zip(list(mapper.index), list(mapper['REGION'])))
    size_lower = dict(zip(list(size_constraint['SIZE']), list(size_constraint['MIN'])))
    size_upper = dict(zip(list(size_constraint['SIZE']), list(size_constraint['MAX'])))
    size_mapper = dict(zip(list(mapper.index), list(mapper['SIZE'])))
    style_lower = dict(zip(list(style_constraint['STYLE']), list(style_constraint['MIN'])))
    style_upper = dict(zip(list(style_constraint['STYLE']), list(style_constraint['MAX'])))
    style_mapper = dict(zip(list(mapper.index), list(mapper['STYLE'])))
    credit_lower = dict(zip(list(credit_constraint['CREDIT']), list(credit_constraint['MIN'])))
    credit_upper = dict(zip(list(credit_constraint['CREDIT']), list(credit_constraint['MAX'])))
    credit_mapper = dict(zip(list(mapper.index), list(mapper['CREDIT'])))
    duration_lower = dict(zip(list(duration_constraint['DURATION']), list(duration_constraint['MIN'])))
    duration_upper = dict(zip(list(duration_constraint['DURATION']), list(duration_constraint['MAX'])))
    duration_mapper = dict(zip(list(mapper.index), list(mapper['DURATION'])))
    asset_lower = dict(zip(list(asset_constraint['ASSET_CLASS']), list(asset_constraint['MIN'])))
    asset_upper = dict(zip(list(asset_constraint['ASSET_CLASS']), list(asset_constraint['MAX'])))
    asset_mapper = dict(zip(list(mapper.index), list(mapper['ASSET_CLASS'])))
    type_lower = dict(zip(list(type_constraint['SECURITY_TYPE']), list(type_constraint['MIN'])))
    type_upper = dict(zip(list(type_constraint['SECURITY_TYPE']), list(type_constraint['MAX'])))
    type_mapper = dict(zip(list(mapper.index), list(mapper['SECURITY_TYPE'])))
    holding_lower = dict(zip(list(holding_constraint['HOLDING']), list(holding_constraint['MIN'])))
    holding_upper = dict(zip(list(holding_constraint['HOLDING']), list(holding_constraint['MAX'])))
    holding_mapper = dict(zip(list(mapper.index), list(mapper['HOLDING'])))
    sector_lower = dict(zip(list(sector_constraint['SECTOR']), list(sector_constraint['MIN'])))
    sector_upper = dict(zip(list(sector_constraint['SECTOR']), list(sector_constraint['MAX'])))
    sector_mapper = dict(zip(list(mapper.index), list(mapper['SECTOR'])))

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

    mapper_list = [
        region_mapper,
        size_mapper,
        style_mapper,
        credit_mapper,
        duration_mapper,
        asset_mapper,
        type_mapper,
        holding_mapper,
        sector_mapper
    ]

    mapper_list = [{k: v for k, v in i.items() if k in keys} for i in mapper_list]
    for mapper in mapper_list:
        mapper = {k: mapper[k] for k in mapper if type(k) is str}

    region_mapper = mapper_list[0]
    size_mapper = mapper_list[1]
    style_mapper = mapper_list[2]
    credit_mapper = mapper_list[3]
    duration_mapper = mapper_list[4]
    asset_mapper = mapper_list[5]
    type_mapper = mapper_list[6]
    holding_mapper = mapper_list[7]
    sector_mapper = mapper_list[8]

    def style_constraint(style_mapper):
        value = dict(sorted(style_mapper.copy().items()))
        growth = dict(sorted(style_mapper.copy().items()))
        blend = dict(sorted(style_mapper.copy().items()))
        for key in value:
            if value.get(key) == 'Value':
                value[key], growth[key], blend[key] = 1, 0, 0
            elif value.get(key) == 'Growth':
                value[key], growth[key], blend[key] = 0, 1, 0
            elif value.get(key) == 'Blend':
                value[key], growth[key], blend[key] = 0, 0, 1
            elif value.get(key) == 'Non-Equity':
                value[key], growth[key], blend[key] = 0, 0, 0
            elif str(value.get(key)) == '':
                value[key], growth[key], blend[key] = 0, 0, 0
            else:
                pass
        value = list(value.values())
        growth = list(growth.values())
        blend = list(blend.values())
        return value, growth, blend

    def size_constraint(size_mapper, type_mapper):
        sec_type = dict(sorted(type_mapper.copy().items()))
        broadmkt_etp = dict(sorted(size_mapper.copy().items()))
        largecap_etp = dict(sorted(size_mapper.copy().items()))
        midcap_etp = dict(sorted(size_mapper.copy().items()))
        smallcap_etp = dict(sorted(size_mapper.copy().items()))
        largecap_stock = dict(sorted(size_mapper.copy().items()))
        midcap_stock = dict(sorted(size_mapper.copy().items()))
        smallcap_stock = dict(sorted(size_mapper.copy().items()))
        for key in sec_type:
            if (broadmkt_etp.get(key) == 'Broad Market') & (sec_type.get(key) == 'ETP'):
                broadmkt_etp[key], largecap_etp[key], midcap_etp[key], smallcap_etp[key], largecap_stock[key], \
                midcap_stock[key], smallcap_stock[key] = 1, 1, 0, 0, 0, 0, 0
            elif (broadmkt_etp.get(key) == 'Large-cap') & (sec_type.get(key) == 'ETP'):
                broadmkt_etp[key], largecap_etp[key], midcap_etp[key], smallcap_etp[key], largecap_stock[key], \
                midcap_stock[key], smallcap_stock[key] = 0, 1, 0, 0, 0, 0, 0
            elif (broadmkt_etp.get(key) == 'Mid-cap') & (sec_type.get(key) == 'ETP'):
                broadmkt_etp[key], largecap_etp[key], midcap_etp[key], smallcap_etp[key], largecap_stock[key], \
                midcap_stock[key], smallcap_stock[key] = 0, 0, 1, 0, 0, 0, 0
            elif (broadmkt_etp.get(key) == 'Small-cap') & (sec_type.get(key) == 'ETP'):
                broadmkt_etp[key], largecap_etp[key], midcap_etp[key], smallcap_etp[key], largecap_stock[key], \
                midcap_stock[key], smallcap_stock[key] = 0, 0, 0, 1, 0, 0, 0
            elif (broadmkt_etp.get(key) == 'Large-cap') & (sec_type.get(key) == 'Common Stock'):
                broadmkt_etp[key], largecap_etp[key], midcap_etp[key], smallcap_etp[key], largecap_stock[key], \
                midcap_stock[key], smallcap_stock[key] = 0, 0, 0, 0, 1, 0, 0
            elif (broadmkt_etp.get(key) == 'Mid-cap') & (sec_type.get(key) == 'Common Stock'):
                broadmkt_etp[key], largecap_etp[key], midcap_etp[key], smallcap_etp[key], largecap_stock[key], \
                midcap_stock[key], smallcap_stock[key] = 0, 0, 0, 0, 0, 1, 0
            elif (broadmkt_etp.get(key) == 'Small-cap') & (sec_type.get(key) == 'Common Stock'):
                broadmkt_etp[key], largecap_etp[key], midcap_etp[key], smallcap_etp[key], largecap_stock[key], \
                midcap_stock[key], smallcap_stock[key] = 0, 0, 0, 0, 0, 0, 1
            else:
                broadmkt_etp[key], largecap_etp[key], midcap_etp[key], smallcap_etp[key], largecap_stock[key], \
                midcap_stock[key], smallcap_stock[key] = 0, 0, 0, 0, 0, 0, 0
        broadmkt_etp = list(broadmkt_etp.values())
        largecap_etp = list(largecap_etp.values())
        midcap_etp = list(midcap_etp.values())
        smallcap_etp = list(smallcap_etp.values())
        largecap_stock = list(largecap_stock.values())
        midcap_stock = list(midcap_stock.values())
        smallcap_stock = list(smallcap_stock.values())
        return broadmkt_etp, largecap_etp, midcap_etp, smallcap_etp, largecap_stock, midcap_stock, smallcap_stock

    def type_constraint(asset_mapper, type_mapper):
        sec_type = dict(sorted(type_mapper.copy().items()))
        equity_etp = dict(sorted(asset_mapper.copy().items()))
        equity_stock = dict(sorted(asset_mapper.copy().items()))
        for key in sec_type:
            if (equity_etp.get(key) == 'Equity') & (sec_type.get(key) == 'ETP'):
                equity_etp[key], equity_stock[key] = 1, 0
            elif (equity_stock.get(key) == 'Equity') & (sec_type.get(key) == 'Common Stock'):
                equity_etp[key], equity_stock[key] = 0, 1
            else:
                equity_etp[key], equity_stock[key] = 0, 0
        equity_etp = list(equity_etp.values())
        equity_stock = list(equity_stock.values())
        return equity_etp, equity_stock

    def region_constraint(region_mapper):
        region_dict = dict(sorted(region_mapper.copy().items()))
        usa_global = dict(sorted(region_mapper.copy().items()))
        developed = dict(sorted(region_mapper.copy().items()))
        emerging = dict(sorted(region_mapper.copy().items()))
        for key in region_dict:
            if region_dict.get(key) == 'U.S.':
                usa_global[key], developed[key], emerging[key] = 1, 0, 0
            elif region_dict.get(key) == 'Global':
                usa_global[key], developed[key], emerging[key] = 1, 0, 0
            elif region_dict.get(key) == 'Developed Markets':
                usa_global[key], developed[key], emerging[key] = 0, 1, 0
            elif region_dict.get(key) == 'Emerging Markets':
                usa_global[key], developed[key], emerging[key] = 0, 0, 1
            elif region_dict.get(key) == 'Emerging Markets Local Currency':
                usa_global[key], developed[key], emerging[key] = 0, 0, 1
            else:
                usa_global[key], developed[key], emerging[key] = 0, 0, 0
        usa_global = list(usa_global.values())
        developed = list(developed.values())
        emerging = list(emerging.values())
        return usa_global, developed, emerging

    value, growth, blend = style_constraint(style_mapper)
    equity_etp, equity_stock = type_constraint(asset_mapper, type_mapper)
    broadmkt_etp, largecap_etp, midcap_etp, smallcap_etp, largecap_stock, midcap_stock, smallcap_stock = size_constraint(size_mapper, type_mapper)
    usa_global, developed, emerging = region_constraint(region_mapper)

    return (
        bounds,
        region_lower, region_upper, region_mapper, size_lower, size_upper, size_mapper,
        style_lower, style_upper, style_mapper, credit_lower, credit_upper, credit_mapper,
        duration_lower, duration_upper, duration_mapper, asset_lower, asset_upper, asset_mapper,
        type_lower, type_upper, type_mapper, holding_lower, holding_upper, holding_mapper,
        sector_lower, sector_upper, sector_mapper, value, growth, blend, equity_etp, equity_stock,
        broadmkt_etp, largecap_etp, midcap_etp, smallcap_etp, largecap_stock, midcap_stock, smallcap_stock,
        usa_global, developed, emerging
    )


def clean_weights(weights, cutoff=0.0001, rounding=4):
    """
    Helper method to clean the raw weights, setting any weights whose absolute
    values are below the cutoff to zero, and rounding the rest.

    :param weightings: (pd.DataFrame) DataFrame of weightings.
    :type cutoff: float, optional
    :param rounding: number of decimal places to round the weights, defaults to 5.
                     Set to None if rounding is not desired.
    :return: (pd.DataFrame) DataFrame of cleaned weights.
    """
    if weights is None:
        raise AttributeError("Weights not yet computed")
    clean_weights = weights.copy()
    clean_weights[np.abs(clean_weights)<cutoff]=0
    if rounding is not None:
        if not isinstance(rounding, int) or rounding<1:
            raise ValueError("rounding must be a positive integer")
        clean_weights = np.round(clean_weights, rounding)
    clean_weights = clean_weights.div(clean_weights.sum())
    clean_weights = clean_weights[clean_weights!=0]
    return clean_weights


def optimize_portfolio(
        exp_returns, cov_matrix,
        obj_function='max_sharpe',
        target_volatility=0.4, target_return=0.2,
        risk_free_rate=0.02, risk_aversion=1, market_neutral=False,
        restricted=False, banned=False, allow_cat2=False, gamma=0.0, add_custom_constraints=False, min_weight=0.0
):
    """
    Compute the optimal portfolio.

    :param exp_returns: (pd.Series) Expected returns for each asset.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
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
    :param restricted: (bool) Optional, filters out tickers on the "restricted" tab in ('../data/portopt_inputs.xlsx'). Default is False.
    :param banned: (bool) Optional, filters out tickers on the "banned" tab in ('../data/portopt_inputs.xlsx'). Default is False.
    :param gamma: (float) Optional, L2 regularisation parameter, defaults to 0. Increase if you want more non-negligible weights.
    :param add_custom_constraints: (bool) Optional, adds custom constraints to the optimization problem.
    :param min_weight: (float) Optional, minimum weight to apply. Weights below this level are reallocated to weights which are
                               above this level.
    :return: (tuple) Tuple of weightings (pd.DataFrame) and results (pd.DataFrame) showing risk, return, sharpe ratio
                     metrics.
    """
    # Generate constraints
    (
        bounds,
        region_lower, region_upper, region_mapper, size_lower, size_upper, size_mapper,
        style_lower, style_upper, style_mapper, credit_lower, credit_upper, credit_mapper,
        duration_lower, duration_upper, duration_mapper, asset_lower, asset_upper, asset_mapper,
        type_lower, type_upper, type_mapper, holding_lower, holding_upper, holding_mapper,
        sector_lower, sector_upper, sector_mapper, value, growth, blend, equity_etp, equity_stock,
        broadmkt_etp, largecap_etp, midcap_etp, smallcap_etp, largecap_stock, midcap_stock, smallcap_stock,
        usa_global, developed, emerging
    ) = constraints(cov_matrix, restricted, banned, allow_cat2)

    # Empty DataFrames
    weightings = pd.DataFrame()
    results = pd.DataFrame()

    # Instantiate efficient_frontier class
    ef = efficient_frontier.EfficientFrontier(exp_returns, cov_matrix, bounds)

    # Add objective for L2 regularisation
    ef.add_objective(objective_functions.L2_reg, gamma=gamma)

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

    # Add Custom Constraints
    if add_custom_constraints==True:
        ef.add_constraint(lambda w: w @ growth <= w @ value)
        ef.add_constraint(lambda w: w @ largecap_etp >= 2 * w @ midcap_etp)
        ef.add_constraint(lambda w: w @ midcap_etp >= 2 * w @ smallcap_etp)
        ef.add_constraint(lambda w: w @ largecap_stock >= 2 * w @ midcap_stock)
        ef.add_constraint(lambda w: w @ midcap_stock >= 2 * w @ smallcap_stock)
        ef.add_constraint(lambda w: w @ equity_etp >= 2 * w @ equity_stock)
        ef.add_constraint(lambda w: w @ usa_global >= 2 * w @ developed)
        ef.add_constraint(lambda w: w @ developed >= 2 * w @ emerging)

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
    weights = ef.clean_weights(min_weight)
    weights = pd.DataFrame.from_dict(weights, orient='index', columns=[int(1)]).round(4)
    weights = clean_weights(weights, cutoff=min_weight)
    performance = pd.DataFrame(ef.portfolio_performance(risk_free_rate=risk_free_rate)).round(4)
    weightings = pd.concat([weightings, weights], axis=1).round(4)
    results = pd.concat([results, performance], axis=1)
    results.columns = ['PORTFOLIO']
    results = results.rename(index={0: 'Expected_Return', 1: 'Volatility', 2: 'Sharpe_Ratio'})
    weightings.index.name = 'TICKER'
    weightings = pd.merge(classification, weightings, on='TICKER').fillna(0)
    return weightings, results


def display_portfolio(portfolio, results):
    """
    Display the optimal portfolio.

    :param portfolio: (pd.DataFrame) Weightings DataFrame from optimize_portfolio() function.
    :param results: (pd.DataFrame) Results DataFrame from optimize_portfolio() function.
    :return: (pd.DataFrame) Displays the optimized portfolio.
    """
    portfolio.rename(columns={int(1): 'WEIGHTING'}, inplace=True)
    portfolio = portfolio[portfolio['WEIGHTING']!=0].sort_values(by='WEIGHTING', ascending=False)
    print(results)
    return portfolio


def min_risk(
        exp_returns, cov_matrix,
        obj_function='min_volatility',
        restricted=False, banned=False, allow_cat2=False, gamma=0.0, min_weight=0.0
):
    """
    Computes the minimum volatility of the minimum volatility portfolio.

    :param exp_returns: (pd.Series) Expected returns for each asset.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param obj_function: (str) Objective function used in the portfolio optimization. Defaults to 'min_volatility'.
    :param gamma: (float) Optional, L2 regularisation parameter, defaults to 0. Increase if you want more non-negligible weights.
    :param min_weight: (float) Optional, minimum weight to apply. Weights below this level are reallocated to weights which are
                               above this level.
    :return: (float) Volatility of the minimum risk portfolio.
    """
    min_volatility = optimize_portfolio(
        exp_returns, cov_matrix,
        obj_function,
        restricted=restricted, banned=banned, allow_cat2=allow_cat2, gamma=gamma, min_weight=min_weight
    )
    return min_volatility[1].loc['Volatility'].squeeze()


def max_risk(
        exp_returns, cov_matrix,
        obj_function='efficient_risk',
        target_volatility=0.4,
        restricted=False, banned=False, allow_cat2=False, gamma=0.0, min_weight=0.0
):
    """
    Computes the maximum volatility of the maximum volatility portfolio. The default target volatility used to optimize
    the maximum volatility portfolio is 0.40.

    :param exp_returns: (pd.Series) Eexpected returns for each asset.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param obj_function: (str) Objective function used in the portfolio optimization. Defaults to 'efficient_risk'.
    :param target_volatility: (float) Optional, the desired maximum volatility of the resulting portfolio. Required if
                                      objective function is 'efficient_risk', otherwise, parameter is ignored. Defaults
                                      to 0.01.
    :param gamma: (float) Optional, L2 regularisation parameter, defaults to 0. Increase if you want more non-negligible weights.
    :param min_weight: (float) Optional, minimum weight to apply. Weights below this level are reallocated to weights which are
                               above this level.
    :return: (float) Portfolio volatility of the maximum risk portfolio.
    """
    max_volatility = optimize_portfolio(
        exp_returns, cov_matrix,
        obj_function,
        target_volatility,
        restricted=restricted, banned=banned, allow_cat2=allow_cat2, gamma=gamma, min_weight=min_weight
    )
    return max_volatility[1].loc['Volatility'].squeeze()


def compute_efficient_frontier(
        exp_returns, cov_matrix,
        restricted=False, banned=False, allow_cat2=False, gamma=0.0, add_custom_constraints=False, min_weight=0.0):
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
    :param restricted: (bool) Optional, filters out tickers on the "restricted" tab in ('../data/portopt_inputs.xlsx'). Default is False.
    :param banned: (bool) Optional, filters out tickers on the "banned" tab in ('../data/portopt_inputs.xlsx'). Default is False.
    :param gamma: (float) Optional, L2 regularisation parameter, defaults to 0. Increase if you want more non-negligible weights.
    :param add_custom_constraints: (bool) Optional, adds custom constraints to the optimization problem.
    :param min_weight: (float) Optional, minimum weight to apply. Weights below this level are reallocated to weights which are
                               above this level.
    :return: (tuple) Tuple of pd.DataFrames representing the optimized portfolio weightings and ex-ante risk, return,
                     and Sharpe ratio.
    """
    min_risk_ = min_risk(
        exp_returns, cov_matrix,
        restricted=restricted, banned=banned, allow_cat2=allow_cat2, gamma=gamma, min_weight=min_weight)
    max_risk_ = max_risk(
        exp_returns, cov_matrix,
        restricted=restricted, banned=banned, allow_cat2=allow_cat2, gamma=gamma, min_weight=min_weight)
    cash_weightings = pd.DataFrame()
    results = pd.DataFrame()
    counter = 1
    for i in tqdm(np.linspace(min_risk_ + .001, max_risk_, 20).round(4)):
        optimized_portfolio, optimized_performance = optimize_portfolio(
            exp_returns, cov_matrix,
            obj_function='efficient_risk',
            target_volatility=i,
            restricted=restricted, banned=banned, allow_cat2=allow_cat2, gamma=gamma,
            add_custom_constraints=add_custom_constraints,
            min_weight=min_weight
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
    # classification_schema = classification_schema().rename(columns={'SymbolCusip': 'TICKER'}).set_index('TICKER')
    optimized_portfolios = classification.join(cash_weightings, how='inner')
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
    optimized_portfolios = optimized_portfolios[(optimized_portfolios.T!=0).any()]
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
    optimized_portfolios = optimized_portfolios[(optimized_portfolios.T!=0).any()]
    cash_weightings = optimized_portfolios.iloc[:, 10:]
    risk_weightings = pd.DataFrame(index=cash_weightings.index)
    classifications = classification[classification.index.isin(risk_weightings.index)]
    for i in range(1, cash_weightings.shape[1] + 1):
        try:
            w = cash_weightings[i]
        except KeyError:
            w = cash_weightings['WEIGHTING']
        pvar = np.dot(w.T, np.dot(cov_matrix, w))
        pvol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        pvolw = ((np.dot(w, cov_matrix)) / pvar) * w
        risk_weightings = pd.concat([risk_weightings, pvolw], axis=1)
    risk_weightings = pd.concat([classifications, risk_weightings.round(4)], axis=1)
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
    Computes the cash and risk weightings of all efficient frontier portfolios.

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
    cash_focus_df = cash_focus_df.loc[cash_focus_df.index!='']
    risk_focus_df = risk_focus_df.loc[risk_focus_df.index!='']
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


def eff_frontier_plot(cov_matrix, exp_returns, results, figsize=(12, 6), save=False, show=True):
    """
    Plots the efficient frontier and individual assets.

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
    ticker_vols = compute_ticker_vols(cov_matrix.index, cov_matrix)
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
        cov_matrix, exp_returns,
        figsize=(11, 5),
        save=False, show=True, restricted=False, banned=False, allow_cat2=False, gamma=0.0, add_custom_constraints=False):
    """
    Plots the efficient frontier and individual assets.

    :param tickers: (list) List of tickers.
    :param cov_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param exp_returns: (pd.Series) Expected returns for each asset.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
                                   Defaults to (11, 5).
    :param save: (bool) Optional, width, height in inches. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :param restricted: (bool) Optional, filters out tickers on the "restricted" tab in ('../data/portopt_inputs.xlsx'). Default is False.
    :param banned: (bool) Optional, filters out tickers on the "banned" tab in ('../data/portopt_inputs.xlsx'). Default is False.
    :param gamma: (float) Optional, L2 regularisation parameter, defaults to 0. Increase if you want more non-negligible weights.
    :param add_custom_constraints: (bool) Optional, adds custom constraints to the optimization problem.
    :return: (fig) Plot of efficient frontier and individual assets.
    """
    (
        bounds,
        region_lower, region_upper, region_mapper, size_lower, size_upper, size_mapper,
        style_lower, style_upper, style_mapper, credit_lower, credit_upper, credit_mapper,
        duration_lower, duration_upper, duration_mapper, asset_lower, asset_upper, asset_mapper,
        type_lower, type_upper, type_mapper, holding_lower, holding_upper, holding_mapper,
        sector_lower, sector_upper, sector_mapper, value, growth, blend, equity_etp, equity_stock,
        broadmkt_etp, largecap_etp, midcap_etp, smallcap_etp, largecap_stock, midcap_stock, smallcap_stock,
        usa_global, developed, emerging
    ) = constraints(cov_matrix, restricted, banned, allow_cat2)
    fig, ax = plt.subplots(figsize=figsize)
    ef = efficient_frontier.EfficientFrontier(exp_returns, cov_matrix, bounds)
    ef.add_objective(objective_functions.L2_reg, gamma=gamma)
    ef.add_sector_constraints(region_mapper, region_lower, region_upper)
    ef.add_sector_constraints(size_mapper, size_lower, size_upper)
    ef.add_sector_constraints(style_mapper, style_lower, style_upper)
    ef.add_sector_constraints(credit_mapper, credit_lower, credit_upper)
    ef.add_sector_constraints(duration_mapper, duration_lower, duration_upper)
    ef.add_sector_constraints(asset_mapper, asset_lower, asset_upper)
    ef.add_sector_constraints(type_mapper, type_lower, type_upper)
    ef.add_sector_constraints(holding_mapper, holding_lower, holding_upper)
    ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    if add_custom_constraints==True:
        ef.add_constraint(lambda w: w @ growth <= w @ value)
        ef.add_constraint(lambda w: w @ largecap_etp >= 2 * w @ midcap_etp)
        ef.add_constraint(lambda w: w @ midcap_etp >= 2 * w @ smallcap_etp)
        ef.add_constraint(lambda w: w @ largecap_stock >= 2 * w @ midcap_stock)
        ef.add_constraint(lambda w: w @ midcap_stock >= 2 * w @ smallcap_stock)
        ef.add_constraint(lambda w: w @ usa_global >= 2 * w @ developed)
        ef.add_constraint(lambda w: w @ developed >= 2 * w @ emerging)
        ef.add_constraint(lambda w: w @ equity_etp >= 3 * w @ equity_stock)
    else:
        pass
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    ax.get_lines()[0].set_color("black")
    ax.get_lines()[0].set_linewidth(2.0)

    # Find the tangency portfolio
    try:
        ef.max_sharpe()
        ret_tangent, std_tangent, _ = ef.portfolio_performance()
        ax.scatter(std_tangent, ret_tangent, marker="*", s=400, c="red", label="Max Sharpe")
        ax.legend(['Constrained Efficient Frontier', "Max Sharpe"], loc='upper left')
    except:
        ax.legend(['Constrained Efficient Frontier'], loc='upper left')

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
    if save==True: plt.savefig(
        '../charts/monte_carlo_frontier_{}.png'.format(datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if show==False: plt.close()


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
    benchmark_name = api.name(api_source, benchmark_ticker)
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


def benchmarks(prices, method='buy_and_hold_equal_weight', weights=None, resample_by=None, verbose=False):
    if method=='buy_and_hold_equal_weight':
        s = pd.Series(np.repeat(1 / len(prices.columns), len(prices.columns)), index=prices.columns)
        s.name = method
        return s
    elif method=='beststock':
        beststock = BestStock()
        beststock.allocate(prices, weights, resample_by, verbose)
        s = pd.Series(beststock.weights, index=prices.columns)
        s.name = method
        return s
    elif method=='CRP':
        crp = CRP()
        crp.allocate(prices, weights, resample_by, verbose)
        s = pd.Series(crp.weights, index=prices.columns)
        s.name = method
        return s
    elif method=='BCRP':
        bcrp = BestStock()
        bcrp.allocate(prices, weights, resample_by, verbose)
        s = pd.Series(bcrp.weights, index=prices.columns)
        s.name = method
        return s
    else:
        raise ValueError('Please input a valid method (e.g. equal_weight, beststock, CRP, BCRP).')
