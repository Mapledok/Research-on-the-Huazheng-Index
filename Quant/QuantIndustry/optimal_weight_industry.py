"""
@Time : 2023/4/6 9:41
@Author : 十三
@Email : mapledok@outlook.com
@File : optimal_weight_industry.py
@Project : Quant
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import *
from multiprocessing import Pool
from tqdm import trange
from scipy.optimize import minimize, approx_fprime
from pprint import pprint
from sklearn.covariance import LedoitWolf, GraphicalLasso, GraphicalLassoCV, EmpiricalCovariance
from sklearn.linear_model import Lasso, Ridge
from functools import lru_cache


class CFG:
    path_industry_mean = r'...\CSI500\行业涨跌幅\等权行业涨跌幅_前半年.xlsx'
    path_industry_mv = r'...\CSI500\行业涨跌幅\市值权重行业涨跌幅_前半年.xlsx'

    returns_mean = pd.ExcelFile(path_industry_mean)
    returns_mv = pd.ExcelFile(path_industry_mv)

    sheet_names = returns_mean.sheet_names


class QuantitativeTradingIndustry:
    """
    Role: Research on the industry.
    """

    def __init__(self, sheet_name: str, internal_weighting: str):
        self.sheet_name = sheet_name
        self.internal_weighting = internal_weighting

    @lru_cache(maxsize=None)
    def build_dataset(self) -> pd.DataFrame:
        if self.internal_weighting == 'mean':
            sr = CFG.returns_mean.parse(sheet_name=self.sheet_name)
        elif self.internal_weighting == 'mv':
            sr = CFG.returns_mv.parse(sheet_name=self.sheet_name)
        cols = sr.columns
        industries = sr.iloc[:, 0].values
        sr.drop(columns=cols[0], axis=1, inplace=True)
        sr = pd.DataFrame(sr.values.T, columns=industries, dtype='float')
        return sr

    @lru_cache(maxsize=None)
    def compute_cov_mean(self) -> Tuple[np.array, np.array]:
        sr = self.build_dataset()
        # Maximum likelihood estimation of the covariance matrix
        cov_mle = EmpiricalCovariance().fit(sr).covariance_
        mean_return = sr.mean(axis=0)
        return cov_mle, mean_return

    def stochastic_simulation(self, number_simulations: int, ins: int):
        cov, mean_return = self.compute_cov_mean()
        cov, mean_return = cov[:ins, :ins], mean_return[:ins]

        weights = np.random.random(size=(number_simulations, len(mean_return)))
        weights /= weights.sum(axis=1)[:, None]

        portfolio_return = np.dot(weights, mean_return)
        portfolio_volatility = np.sqrt(np.einsum('ii -> i', np.dot(np.dot(weights, cov), weights.T)))
        sharpe_ratio = portfolio_return / portfolio_volatility

        plt.style.use('seaborn-v0_8')
        plt.figure(dpi=540)
        plt.scatter(x=portfolio_volatility, y=portfolio_return,
                    c=sharpe_ratio,
                    cmap=plt.cm.cool,
                    edgecolors='none',
                    s=15)
        plt.title('Stochastic Simulation')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Portfolio Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.grid(False)
        plt.show()

    def minimize_function(self):
        cov, mean_return = self.compute_cov_mean()
        optimization_objective = [lambda x: -np.dot(mean_return, x),
                                  lambda x: np.sqrt(np.dot(np.dot(x.T, cov), x)),
                                  lambda x: -np.dot(mean_return, x) / np.sqrt(np.dot(np.dot(x.T, cov), x))]

        print('mode:',
              '0: Maximize portfolio_return',
              '1: Minimize portfolio_volatility',
              '2: Maximize sharpe_ratio',
              sep='\n')
        mode = int(input('Please select mode:'))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for _ in range(len(mean_return)))
        optimizer = minimize(optimization_objective[mode],
                             x0=np.array(len(mean_return) * [1 / len(mean_return)]),
                             method='SLSQP',
                             bounds=bnds,
                             constraints=cons)
        code_list = ['Maximum Return', 'Minimum Volatility', 'Maximum Sharpe']
        print(f'{code_list[mode]}: {optimizer.fun}',
              f'Was the optimization successful?: {optimizer.success}',
              f'Optimal weight: \n{optimizer.x}',
              sep='\n')

    def compute_optimal_weight(self) -> list:
        cov, mean_return = self.compute_cov_mean()
        optimization_objective = lambda x: np.sqrt(np.dot(np.dot(x.T, cov), x))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for _ in range(len(mean_return)))
        optimizer = minimize(optimization_objective,
                             x0=np.array(len(mean_return) * [1 / len(mean_return)]),
                             method='SLSQP',
                             bounds=bnds,
                             constraints=cons)
        return optimizer.x


# The first ins industries are taken for Monte Carlo simulation to draw the efficient frontier.
# qti = QuantitativeTradingIndustry(sheet_name=CFG.sheet_names[0], internal_weighting='mean')
# qti.stochastic_simulation(number_simulations=5000, ins=5)

# The optimal weights are calculated for 59 time nodes.
def compute_optimal_weight(sheet_name: str) -> list:
    qti = QuantitativeTradingIndustry(sheet_name=sheet_name, internal_weighting='mean')
    return qti.compute_optimal_weight()


#  Distributed computing.
if __name__ == '__main__':
    with Pool() as p:
        optimal_weights = p.map(compute_optimal_weight, CFG.sheet_names)
    df_weights = pd.DataFrame(pd.DataFrame(optimal_weights).values.T, columns=CFG.sheet_names)
    df_weights.to_excel(r'...\CSI500\optimal_weight\optimal_weight_mean.xlsx',
                        index=False)
