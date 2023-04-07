"""
@Time : 2023/2/7 14:40
@Author : 十三
@Email : mapledok@outlook.com
@File : Index_calculation_industry.py
@Project : Quant
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import *
from functools import lru_cache


class CFG:
    path_industry_mean = r'...\CSI500\行业涨跌幅\等权行业涨跌幅_后三个月.xlsx'
    path_industry_mv = r'...\CSI500\行业涨跌幅\市值权重行业涨跌幅_后三个月.xlsx'

    returns_mean = pd.ExcelFile(path_industry_mean)
    returns_mv = pd.ExcelFile(path_industry_mv)

    path_weight_mean = r'...\CSI500\optimal_weight\optimal_weight_mean.xlsx'
    path_weight_mv = r'...\CSI500\optimal_weight\optimal_weight_mv.xlsx'

    optimal_weights_mean = pd.read_excel(path_weight_mean)
    optimal_weights_mv = pd.read_excel(path_weight_mv)

    path_return = r'...\CSI500\涨跌幅市值行业分类\涨跌幅市值行业分类_后三个月.xlsx'
    returns = pd.ExcelFile(path_return)

    sheet_names = returns_mean.sheet_names


class IndexCalculationIndustry:
    """
    Main function: reproduce the results
    Note:
    external_weighting='mean' means weighting industries equally.
    external_weighting='mv' means market capitalization weighting between industries.
    external_weighting='ow' means weighting industries with optimal weights.
    """

    def __init__(self, internal_weighting: str, external_weighting: str):
        self.internal_weighting = internal_weighting
        self.external_weighting = external_weighting

    @lru_cache(maxsize=None)
    def build_dataset(self, sheet_name: str) -> pd.DataFrame:
        if self.internal_weighting == 'mean':
            sr = CFG.returns_mean.parse(sheet_name=sheet_name)
        elif self.internal_weighting == 'mv':
            sr = CFG.returns_mv.parse(sheet_name=sheet_name)
        cols = sr.columns
        industries = sr.iloc[:, 0].values
        sr.drop(columns=cols[0], axis=1, inplace=True)
        sr = pd.DataFrame(sr.values.T, columns=industries, dtype='float')
        return sr

    @lru_cache(maxsize=None)
    def build_optimal_weights(self):
        if self.internal_weighting == 'mean':
            return CFG.optimal_weights_mean
        elif self.internal_weighting == 'mv':
            return CFG.optimal_weights_mv

    @lru_cache(maxsize=None)
    def build_series(self):
        """
        Calculate the total rise or fall for each trading day on a different basis.
        """

        @lru_cache(maxsize=None)
        def mvw(sn):
            return CFG.returns.parse(sheet_name=sn, usecols=[2, 3]).groupby(by='中信二级行业分类').mean()['市值']

        if self.external_weighting == 'mean':
            weight_func = lambda sr, sn: sr.shape[-1] * [1 / sr.shape[-1]]
        elif self.external_weighting == 'mv':
            weight_func = lambda sr, sn: mvw(sn)/mvw(sn).sum()
        elif self.external_weighting == 'ow':
            weight_func = lambda sr, sn: self.build_optimal_weights()[sn][:sr.shape[-1]].values

        def weighted_return(sn, wf):
            sr = self.build_dataset(sheet_name=sn)
            weights = wf(sr, sn)
            wr = pd.DataFrame(np.dot(sr, weights))
            return wr

        sr_list = list(map(lambda sn: weighted_return(sn, wf=weight_func), CFG.sheet_names))
        sr_groups = pd.concat(sr_list, axis=0)
        sr_groups.reset_index(drop=True, inplace=True)
        return sr_groups

    @lru_cache(maxsize=None)
    def compute_networth(self) -> pd.Series:
        """
        Calculate net worth.
        """

        time_series = self.build_series().to_numpy()
        growth_rates = time_series * 0.01 + 1
        net_worth = np.cumprod(growth_rates) * 1000
        return pd.Series(net_worth)

    @lru_cache(maxsize=None)
    def accumulated_returns(self) -> float:
        net_worth = self.compute_networth()
        return (net_worth.iloc[-1] / net_worth.iloc[0]) - 1

    @lru_cache(maxsize=None)
    def annualized_returns(self) -> float:
        accumulated_returns = self.accumulated_returns()
        time_series = self.build_series()
        return (accumulated_returns + 1) ** (1 / (len(time_series) / 250)) - 1

    @lru_cache(maxsize=None)
    def annualized_fluctuation(self) -> float:
        time_series = self.build_series()
        return (time_series.values.std() * np.sqrt(250)) / 100

    @lru_cache(maxsize=None)
    def sharpe_ratio(self) -> float:
        return self.annualized_returns() / self.annualized_fluctuation()

    def withdrawal_trend(self):
        """
        Plot the backtest trend.
        """

        net_worth = self.compute_networth()
        x_ticks = list(range(0, 3580, 247))
        date_index = pd.date_range(start='2007-12-28', end='2022-12-28', freq='Y').strftime('%Y/%m/%d')
        x_labels = [str(date_index[n]) for n in range(len(date_index))]
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(13, 5), dpi=540)
        net_worth.plot(label=f'internal:{self.internal_weighting} external:{self.external_weighting}')
        plt.title('Backtesting trend of historical portfolio returns',
                  fontsize=15,
                  fontweight='bold')
        plt.grid(False, axis='x')
        plt.legend(loc='best')
        plt.xticks(ticks=x_ticks, labels=x_labels, rotation=30)
        plt.show()

    @lru_cache(maxsize=None)
    def maximum_withdrawal(self) -> float:
        net_worth = self.compute_networth()
        rolling_max = net_worth.expanding(min_periods=1).max()
        draw_down = np.abs(net_worth - rolling_max) / rolling_max
        return draw_down.max()

    def results_summary(self):
        print('累计收益: {:.2%}'.format(self.accumulated_returns()),
              '年化收益: {:.2%}'.format(self.annualized_returns()),
              '夏普比率: {}'.format(round(self.sharpe_ratio(), 5)),
              '波动率: {:.2%}'.format(self.annualized_fluctuation()),
              '最大回撤: {:.2%}'.format(self.maximum_withdrawal()),
              sep='\n')


begin = time.time()
ici = IndexCalculationIndustry(internal_weighting='mean', external_weighting='ow')
ici.results_summary()
print(time.time() - begin)
