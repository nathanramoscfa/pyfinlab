import pandas as pd
from datetime import datetime
from pyfinlab import portfolio_optimization as opt

"""
These functions generate formatted Excel files. 
"""

def generate_excel_report(
        optimized_portfolios, risk_weightings, results, backtest_timeseries, cash_focus, risk_focus, periodic_stats,
        label='large_acct'
):
    """
    Generates a formatted Excel portfolio optimization analysis.

    :param optimized_portfolios: (pd.DataFrame) Cash-weightings for efficient frontier portfolios.
    :param risk_weightings: (pd.DataFrame) Risk-weightings for efficient frontier portfolios.
    :param results: (pd.DataFrame) Risk, return, and sharpe ratio for all efficient frontier portfolios. Input the
                                   results pd.DataFrame computed using the opt.optimize_portfolio() function.
    :param backtest_timeseries: (pd.DateFrame) Backtest timeseries of portfolio values.
    :cash_focus: (pd.DataFrame) Cash-weightings weighted by group for efficient frontier portfolios.
    :risk_focus: (pd.DataFrame) Risk-weightings weighted by group for efficient frontier portfolios.
    :periodic_stats: (pd.DataFrame) Periodic stats computed using the performance.compile_periodic_stats() function
                                    from pyfinlab library.
    :label: (str) Label added to filename as a descriptor.
    :return: (obj) Creates Excel workbook objects and saves detailed, formatted results of portfolio optimization.
    """
    report_description = 'optimized_portfolios'
    today = datetime.today().strftime('%m-%d-%Y')
    filename = '../excel/{}_{}_{}.xlsx'.format(report_description, today, label)
    cash_portfolios = optimized_portfolios.loc[~(optimized_portfolios.iloc[:, 10:]==0).all(axis=1)]
    cash_portfolios.index.name = 'TICKER'
    risk_portfolios = risk_weightings.loc[~(risk_weightings.iloc[:, 10:]==0).all(axis=1)]
    dashboard = results.append(opt.cash_focus(optimized_portfolios).get('ASSET_CLASS'))
    dashboard.index.name = 'Dashboard'

    # Create mew Excel file.
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    # Add worksheets in the order you want them here.
    dashboard.to_excel(writer, sheet_name='dashboard')
    cash_portfolios.to_excel(writer, sheet_name='cash_weighting')
    risk_portfolios.to_excel(writer, sheet_name='risk_weighting')
    cash_focus.to_excel(writer, sheet_name='cash_exposure')
    risk_focus.to_excel(writer, sheet_name='risk_exposure')
    pd.DataFrame().to_excel(writer, sheet_name='efficient_frontier')
    periodic_stats['cagr'].to_excel(writer, sheet_name='cagr')
    periodic_stats['sharpe'].to_excel(writer, sheet_name='sharpe')
    periodic_stats['vol'].to_excel(writer, sheet_name='vol')
    periodic_stats['beta'].to_excel(writer, sheet_name='beta')
    periodic_stats['drawdown'].to_excel(writer, sheet_name='drawdown')
    periodic_stats['m2_alpha'].to_excel(writer, sheet_name='m2_alpha')
    periodic_stats['jensen_alpha'].to_excel(writer, sheet_name='jensen_alpha')
    periodic_stats['treynor'].to_excel(writer, sheet_name='treynor')
    periodic_stats['sortino'].to_excel(writer, sheet_name='sortino')
    periodic_stats['capture_ratio'].to_excel(writer, sheet_name='capture_ratio')
    periodic_stats['appraisal_ratio'].to_excel(writer, sheet_name='appraisal_ratio')
    periodic_stats['ulcer'].to_excel(writer, sheet_name='ulcer')
    periodic_stats['info_ratio'].to_excel(writer, sheet_name='info_ratio')
    periodic_stats['m2'].to_excel(writer, sheet_name='m2')
    periodic_stats['capm'].to_excel(writer, sheet_name='capm')

    # Create workbook objects
    workbook = writer.book
    worksheet1 = writer.sheets['dashboard']
    worksheet2 = writer.sheets['cash_weighting']
    worksheet3 = writer.sheets['risk_weighting']
    worksheet4 = writer.sheets['cash_exposure']
    worksheet5 = writer.sheets['risk_exposure']
    worksheet6 = writer.sheets['efficient_frontier']
    worksheet7 = writer.sheets['cagr']
    worksheet8 = writer.sheets['drawdown']
    worksheet9 = writer.sheets['vol']
    worksheet10 = writer.sheets['beta']
    worksheet13 = writer.sheets['sharpe']
    worksheet11 = writer.sheets['m2_alpha']
    worksheet12 = writer.sheets['jensen_alpha']
    worksheet14 = writer.sheets['treynor']
    worksheet15 = writer.sheets['sortino']
    worksheet16 = writer.sheets['info_ratio']
    worksheet17 = writer.sheets['capture_ratio']
    worksheet18 = writer.sheets['appraisal_ratio']
    worksheet19 = writer.sheets['ulcer']
    worksheet20 = writer.sheets['m2']
    worksheet21 = writer.sheets['capm']

    # Workbook Formats
    format1 = workbook.add_format({'fg_color': '#F2F2F2'})  # Background Color Left Aligned
    format1.set_align('left')
    format2 = workbook.add_format({'num_format': '0.00%',  # Percentage Style
                                   'fg_color': '#F2F2F2'})
    format3 = workbook.add_format({'num_format': '0.0000',  # Decimal Style
                                   'fg_color': '#F2F2F2'})
    format4 = workbook.add_format({'num_format': '0.00%',  # Percentage Style Green Bold
                                   'font_color': '#00B050',
                                   'fg_color': '#F2F2F2',
                                   'bold': True})
    format5 = workbook.add_format({'num_format': '0.00%',  # Percentage Style Red Bold
                                   'font_color': '#FF0000',
                                   'fg_color': '#F2F2F2',
                                   'bold': True})
    format6 = workbook.add_format({'fg_color': '#F2F2F2'})  # Background Color
    format7 = workbook.add_format({'fg_color': '#F2F2F2'})  # Background Color
    format7.set_right(2)

    # dashboard
    worksheet1.hide_gridlines()
    worksheet1.set_zoom(110)
    worksheet1.set_column('A:A', 22, format1)
    worksheet1.set_column('B:U', 7.5, format1)
    worksheet1.set_row(1, None, format4)
    worksheet1.set_row(2, None, format5)
    worksheet1.set_row(3, None, format3)
    worksheet1.set_row(4, None, format2)
    worksheet1.set_row(5, None, format2)
    worksheet1.set_row(6, None, format2)
    worksheet1.set_row(7, None, format2)
    worksheet1.set_row(8, None, format2)
    worksheet1.set_row(9, None, format2)
    worksheet1.set_row(10, None, format2)
    for row in range(dashboard.shape[0] + 2, 43):
        worksheet1.write('A{}'.format(row), ' ')
    worksheet1.set_default_row(hide_unused_rows=True)
    worksheet1.set_column('V:XFD', None, None, {'hidden': True})
    worksheet1.conditional_format('B4:U4', {'type': '3_color_scale'})
    worksheet1.insert_image(
        'B12', '../charts/linechart_{}.png'.format(datetime.today().strftime('%m-%d-%Y')))
    worksheet1.conditional_format('B5:U{}'.format(dashboard.shape[0] + 1), {'type': '3_color_scale',
                                                                            'min_color': '#63BE7B',
                                                                            'mid_color': '#FFEB84',
                                                                            'max_color': '#F8696B'})


    # cash_weighting
    worksheet2.hide_gridlines()
    worksheet2.freeze_panes(1, 2, 1, 7)
    worksheet2.set_column('A:A', 16, format1)
    worksheet2.set_column('B:B', 55, format1)
    worksheet2.set_column('C:C', 30, format1, {'level': 3})
    worksheet2.set_column('D:D', 13, format1, {'level': 3})
    worksheet2.set_column('E:E', 12, format1, {'level': 2})
    worksheet2.set_column('F:F', 10, format1, {'level': 2})
    worksheet2.set_column('G:G', 22, format1, {'level': 1})
    worksheet2.set_column('H:H', 19, format1, {'level': 1})
    worksheet2.set_column('I:I', 16, format1, {'level': 1})
    worksheet2.set_column('J:J', 14, format1, {'level': 1})
    worksheet2.set_column('K:K', 16, format1, {'level': 1})
    worksheet2.set_column('L:AE', 7, format2)
    worksheet2.set_column('AF:XFD', None, None, {'hidden': True})
    worksheet2.autofilter('A1:AE{}'.format(optimized_portfolios.shape[1] + 1))
    worksheet2.set_default_row(hide_unused_rows=True)
    worksheet2.conditional_format('L2:AE{}'.format(optimized_portfolios.shape[0] + 1), {'type': '3_color_scale',
                                                                                        'min_color': '#63BE7B',
                                                                                        'mid_color': '#FFEB84',
                                                                                        'max_color': '#F8696B'})

    # risk_weighting
    worksheet3.hide_gridlines()
    worksheet3.freeze_panes(1, 2, 1, 7)
    worksheet3.set_column('A:A', 16, format1)
    worksheet3.set_column('B:B', 55, format1)
    worksheet3.set_column('C:C', 30, format1, {'level': 3})
    worksheet3.set_column('D:D', 13, format1, {'level': 3})
    worksheet3.set_column('E:E', 12, format1, {'level': 2})
    worksheet3.set_column('F:F', 10, format1, {'level': 2})
    worksheet3.set_column('G:G', 22, format1, {'level': 1})
    worksheet3.set_column('H:H', 19, format1, {'level': 1})
    worksheet3.set_column('I:I', 16, format1, {'level': 1})
    worksheet3.set_column('J:J', 14, format1, {'level': 1})
    worksheet3.set_column('K:K', 16, format1, {'level': 1})
    worksheet3.set_column('L:AE', 7, format2)
    worksheet3.set_column('AF:XFD', None, None, {'hidden': True})
    worksheet3.autofilter('A1:AE{}'.format(optimized_portfolios.shape[1] + 1))
    worksheet3.set_default_row(hide_unused_rows=True)
    worksheet3.conditional_format('L2:AE{}'.format(optimized_portfolios.shape[0] + 1), {'type': '3_color_scale',
                                                                                        'min_color': '#63BE7B',
                                                                                        'mid_color': '#FFEB84',
                                                                                        'max_color': '#F8696B'})

    # cash exposure
    worksheet4.hide_gridlines()
    worksheet4.freeze_panes(1, 1)
    worksheet4.set_zoom(130)
    worksheet4.set_column('A:A', 31, format1)
    worksheet4.set_column('B:U', None, format6)
    [worksheet4.set_row(i, None, format2) for i in range(1, cash_focus.shape[0] + 1)]
    worksheet4.autofilter('A1:U{}'.format(cash_focus.shape[1] + 1))
    worksheet4.set_default_row(hide_unused_rows=True)
    worksheet4.set_column('V:XFD', None, None, {'hidden': True})
    worksheet4.conditional_format('B2:U{}'.format(cash_focus.shape[0] + 1), {'type': '3_color_scale',
                                                                             'min_color': '#63BE7B',
                                                                             'mid_color': '#FFEB84',
                                                                             'max_color': '#F8696B'})

    # risk_exposure
    worksheet5.hide_gridlines()
    worksheet5.freeze_panes(1, 1)
    worksheet5.set_zoom(130)
    worksheet5.set_column('A:A', 31, format1)
    worksheet5.set_column('B:U', None, format6)
    [worksheet5.set_row(i, None, format2) for i in range(1, risk_focus.shape[0] + 1)]
    worksheet5.autofilter('A1:U{}'.format(risk_focus.shape[1] + 1))
    worksheet5.set_default_row(hide_unused_rows=True)
    worksheet5.set_column('V:XFD', None, None, {'hidden': True})
    worksheet5.conditional_format('B2:U{}'.format(risk_focus.shape[0] + 1), {'type': '3_color_scale',
                                                                             'min_color': '#63BE7B',
                                                                             'mid_color': '#FFEB84',
                                                                             'max_color': '#F8696B'})

    # efficient_frontier
    worksheet6.hide_gridlines()
    worksheet6.set_column('A:Q', None, format1)
    for row in range(1, 35):
        worksheet6.write('A{}'.format(row), ' ')
    worksheet6.set_default_row(hide_unused_rows=True)
    worksheet6.set_column('R:XFD', None, None, {'hidden': True})
    worksheet6.insert_image(
        'A1', '../charts/efficient_frontier_{}.png'.format(datetime.today().strftime('%m-%d-%Y')))

    # vol
    worksheet9.hide_gridlines()
    worksheet9.freeze_panes(1, 1)
    worksheet9.set_zoom(160)
    worksheet9.set_column('A:H', None, format2)
    worksheet9.set_default_row(hide_unused_rows=True)
    worksheet9.set_column('I:XFD', None, None, {'hidden': True})
    worksheet9.conditional_format(
        'B2:B22', {'type': '3_color_scale', 'min_color': '#63BE7B', 'mid_color': '#FFEB84', 'max_color': '#F8696B'})
    worksheet9.conditional_format(
        'C2:C22', {'type': '3_color_scale', 'min_color': '#63BE7B', 'mid_color': '#FFEB84', 'max_color': '#F8696B'})
    worksheet9.conditional_format(
        'D2:D22', {'type': '3_color_scale', 'min_color': '#63BE7B', 'mid_color': '#FFEB84', 'max_color': '#F8696B'})
    worksheet9.conditional_format(
        'E2:E22', {'type': '3_color_scale', 'min_color': '#63BE7B', 'mid_color': '#FFEB84', 'max_color': '#F8696B'})
    worksheet9.conditional_format(
        'F2:F22', {'type': '3_color_scale', 'min_color': '#63BE7B', 'mid_color': '#FFEB84', 'max_color': '#F8696B'})
    worksheet9.conditional_format(
        'G2:G22', {'type': '3_color_scale', 'min_color': '#63BE7B', 'mid_color': '#FFEB84', 'max_color': '#F8696B'})
    worksheet9.conditional_format(
        'H2:H22', {'type': '3_color_scale', 'min_color': '#63BE7B', 'mid_color': '#FFEB84', 'max_color': '#F8696B'})
    worksheet9.autofilter('A1:H{}'.format(periodic_stats['vol'].shape[1] + 1))


    def worksheet_format1(worksheet, formatting, periodic_stats):
        """
        Formats given worksheet as specified in the code.

        :param worksheet: (obj) Worksheet object corresponding to a particular spreadsheet.
        :param formatting: (obj) Risk-weightings for efficient frontier portfolios.
        :periodic_stats: (pd.DataFrame) Periodic stats computed using the performance.compile_periodic_stats() function
                                    from pyfinlab library.
        :return: (obj) Formatted Worksheet object.
        """
        worksheet.hide_gridlines()
        worksheet.freeze_panes(1, 1)
        worksheet.set_zoom(160)
        worksheet.set_column('A:H', None, formatting)
        worksheet.set_column('A:A', 15, formatting)
        worksheet.set_default_row(hide_unused_rows=True)
        worksheet.set_column('I:XFD', None, None, {'hidden': True})
        worksheet.conditional_format('B2:B22', {'type': '3_color_scale'})
        worksheet.conditional_format('C2:C22', {'type': '3_color_scale'})
        worksheet.conditional_format('D2:D22', {'type': '3_color_scale'})
        worksheet.conditional_format('E2:E22', {'type': '3_color_scale'})
        worksheet.conditional_format('F2:F22', {'type': '3_color_scale'})
        worksheet.conditional_format('G2:G22', {'type': '3_color_scale'})
        worksheet.conditional_format('H2:H22', {'type': '3_color_scale'})
        worksheet.autofilter('A1:H{}'.format(periodic_stats['cagr'].shape[1] + 1))


    worksheet_format1(worksheet7, format2, periodic_stats)  # cagr
    worksheet_format1(worksheet8, format2, periodic_stats)  # drawdown
    worksheet_format1(worksheet10, format3, periodic_stats)  # beta
    worksheet_format1(worksheet11, format2, periodic_stats)  # m2_alpha
    worksheet_format1(worksheet12, format2, periodic_stats)  # jensen_alpha
    worksheet_format1(worksheet13, format3, periodic_stats)  # sharpe
    worksheet_format1(worksheet14, format3, periodic_stats)  # treynor
    worksheet_format1(worksheet15, format3, periodic_stats)  # sortino
    worksheet_format1(worksheet16, format3, periodic_stats)  # info_ratio
    worksheet_format1(worksheet17, format3, periodic_stats)  # capture_ratio
    worksheet_format1(worksheet18, format3, periodic_stats)  # appraisal_ratio
    worksheet_format1(worksheet19, format3, periodic_stats)  # ulcer
    worksheet_format1(worksheet20, format2, periodic_stats)  # m2
    worksheet_format1(worksheet21, format2, periodic_stats)  # capm

    writer.save()
