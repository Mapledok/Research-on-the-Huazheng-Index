"""
@Time : 2023/2/7 14:40
@Author : 十三
@Email : mapledok@outlook.com
@File : Index_calculation_individual.py
@Project : Quant
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import *
from functools import lru_cache


class CFG:
    path_return = r'...\CSI500\涨跌幅市值行业分类\涨跌幅市值行业分类_后三个月.xlsx'
    path_weight = r'...\CSI500\optimal_weight\individual_share.xlsx'

    returns = pd.ExcelFile(path_return)
    optimal_weights = pd.read_excel(path_weight)

    sheet_names = returns.sheet_names


class IndexCalculationIndividual:
    """
    Main function: reproduce the results.
    Note:
    external_weighting='mean' means weighting individual stocks equally.
    external_weighting='mv' represents weighting the market value of individual stocks.
    external_weighting='ow' means weighting the individual stocks by the optimal weight.
    """

    def __init__(self, external_weighting: str):
        self.external_weighting = external_weighting

    @staticmethod
    @lru_cache(maxsize=None)
    def build_dataset(sheet_name: str) -> Tuple[pd.DataFrame, np.array]:
        df = CFG.returns.parse(sheet_name=sheet_name)
        cols = df.columns
        stock_code = df.iloc[:, 0].values
        market_value = df.iloc[:, 3].values
        df.drop(columns=[cols[0], cols[1], cols[2], cols[3]], axis=1, inplace=True)
        df = pd.DataFrame(df.values.T, columns=stock_code, dtype='float')
        return df, market_value

    @lru_cache(maxsize=None)
    def build_series(self):
        """
        Calculate the total rise or fall for each trading day on a different basis.
        """

        if self.external_weighting == 'mean':
            weight_func = lambda sr, mv, sn: sr.shape[-1] * [1 / sr.shape[-1]]
        elif self.external_weighting == 'mv':
            weight_func = lambda sr, mv, sn: mv / mv.sum()
        elif self.external_weighting == 'ow':
            weight_func = lambda sr, mv, sn: CFG.optimal_weights[sn].values

        def weighted_return(sn, wf):
            sr, mv = self.build_dataset(sheet_name=sn)
            weights = wf(sr, mv, sn)
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
        net_worth.plot(label=f'{self.external_weighting}')
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
ici = IndexCalculationIndividual(external_weighting='ow')
ici.results_summary()
print(time.time()-begin)
