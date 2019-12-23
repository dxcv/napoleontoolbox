#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod

import pandas as pd

import numpy as np

from napoleontoolbox.utility import metrics
from scipy.optimize import Bounds, LinearConstraint, minimize

import torch


class AbstractAssembler(ABC):
    def __init__(self, features_path, returns_path, root='../data/', user = 'napoleon', low_bound = 0.02, up_bound = 0.4, supervision_saving_path='_supervision.npy'):

        super().__init__()
        self.root =  root
        self.user =  user
        self.features_path = features_path
        self.returns_path = returns_path
        self.low_bound = low_bound
        self.up_bound = up_bound
        self.supervision_saving_path = supervision_saving_path


    @abstractmethod
    def computeUtility(self, s):
        pass


class UtilitySuperviser(AbstractAssembler):
    def computeUtility(self, s):
        print('quotes loading')

        df = pd.read_pickle(self.root + self.returns_path)

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        strats = [col for col in list(df.columns) if col != 'Date']
        df = df.fillna(method='ffill')

        print('advanced features loading')
        advanced_features = pd.read_pickle(self.root + self.features_path)
        features_names = [col for col in list(advanced_features.columns) if col!='Date']
        advanced_features['Date'] = pd.to_datetime(advanced_features['Date'])


        # quotes_df=quotes_df.sort_values(by='date', ascending=True)
        # quotes_df.head()

        print(df.columns)
        print(df.shape)

        # Computationnal period (default 1 year)

        np.random.seed(0)
        torch.manual_seed(0)

        ##===================##
        ##  Setting targets  ##
        ##===================##

        #
        df_bis = df.copy()
        # df_ret = df_bis.pct_change().fillna(0.)
        # ret = df_ret.values

        print('merging')
        df_bis = pd.merge(df_bis, advanced_features, how='left', on=['Date'])

        print('merging done')
        df_bis.index = df_bis['Date']
        df_bis = df_bis.drop(columns=['Date'])
        print('return')

        df_bis = df_bis.fillna(method='ffill').fillna(method='bfill')
        # df_ret = df_bis.pct_change().fillna(0.)
        df_ret = df_bis.copy()
        prices = df_bis[strats].values

        for col in strats:
            print(col + str(len(df_bis.columns)))
            df_ret[col] = df_bis[col].pct_change().fillna(0.)

        print('return done')
        print('return computed')

        ret_df = df_ret[strats]
        ret = ret_df.values

        feat = df_ret[features_names].values

        dates = df_ret.index


        T = df.index.size
        N = df.columns.size
        w0 = np.ones([N]) / N
        w_ = w0
        previous_w_ = w_

        # print('saving df_ret')
        # df_ret.to_pickle('../data/df_ret.pkl')

        print("recomputing supervision weights")
        # Set constraints and minimze
        const_sum = LinearConstraint(np.ones([1, N]), [1], [1])
        const_ind = Bounds(self.low_bound * np.ones([N]), self.up_bound * np.ones([N]))

        # features = np.zeros([T , N * N + N+ len(features_names)], np.float32)
        # features = np.zeros([T, N * N], np.float32)

        utility_size = 6
        result = np.zeros([T, N, utility_size], np.float32)

        def process(series):
            # True if less than 50% of obs. are constant
            return series.value_counts(dropna=False).max() < 0.7 * s

        # for t in range(max(n_past_features, s), T - s):
        for t in range(s, T):
            np.random.seed(0)
            torch.manual_seed(0)
            if t % 500 == 0:
                print('{:.2%}'.format(t / T))
            # we compute the utility output to predict only if not in future
            if t + s <= T:
                t_s = min(t + s, T)
                X = ret[t: t_s, :]
                mat_cov = np.cov(X, rowvar=False)
                # Avoid constant assets
                sub_X = ret_df.iloc[t: t_s, :].copy()
                assets = sub_X.apply(process).values
                N_ = assets.sum()
                if N_ != 0:
                    def f_minVar(w):
                        w = w.reshape([N_, 1])
                        return np.sqrt(w.T @ mat_cov[assets][:, assets] @ w)
                    def f_maxMean(w):
                        w = w.reshape([N_, 1])
                        return - np.mean(np.cumprod(X[:, assets] @ w + 1, axis=0)[-1, :])
                    def f_MeanVar(w):
                        w = w.reshape([N_, 1])
                        std_dev = np.sqrt(w.T @ mat_cov[assets][:, assets] @ w)
                        mean_ret = np.mean(np.cumprod(X[:, assets] @ w + 1, axis=0)[-1, :])
                        return np.sqrt(252) * std_dev - (np.float_power(mean_ret, s / 252) - 1)
                    def f_sharpe(w):
                        w = w.reshape([N_, 1])
                        return - metrics.sharpe(np.cumprod(np.prod(X[:, assets] @ w + 1, axis=1)))
                    def f_calmar(w):
                        w = w.reshape([N_, 1])
                        return - metrics.calmar(np.cumprod(np.prod(X[:, assets] @ w + 1, axis=1)))
                    def f_drawdown(w):
                        w = w.reshape([N_, 1])
                        return metrics.drawdown(np.cumprod(np.prod(X[:, assets] @ w + 1, axis=1))).max()
                    # Set constraints
                    const_sum = LinearConstraint(np.ones([1, N_]), [1], [1])
                    up_bound_ = max(self.up_bound, 1 / N_)
                    low_bound_ = min(self.low_bound, 1 / N_)
                    const_ind = Bounds(low_bound_ * np.ones([N_]), up_bound_ * np.ones([N_]))
                    # Search optimal weights => target
                    f_list = [f_minVar, f_maxMean, f_sharpe, f_MeanVar, f_calmar, f_drawdown]
                    i = 0
                    previous_w_ = w_
                    w__ = w_[assets]
                    for f in f_list:
                        np.random.seed(0)
                        torch.manual_seed(0)
                        # Optimize f
                        w__ = minimize(
                            f,
                            # w0[assets],
                            w__,
                            method='SLSQP',
                            constraints=[const_sum],
                            bounds=const_ind
                        ).x

                        if np.isnan(w__.sum()):
                            print('Trouble converging : investigate')
                            w__ = previous_w_[assets]

                        w_[assets] = w__
                        w_[~assets] = 0.
                        s_w = w_.sum()

                        # Verify if sum(w) = 1 and set output
                        if s_w == 1.:
                            result[t: t + 1, :, i] = w_

                        elif s_w != 0.:
                            result[t: t + 1, :, i] = w_ / s_w

                        else:
                            result[t: t + 1, :, i] = w0
                        i += 1
                else:
                    for ii in range(utility_size):
                        result[t: t + 1, :, ii] = w0

        print('saving file')
        np.save(self.root + self.user + '_' + str(s) + self.supervision_saving_path, result)
        print('saved files')
        print('number of nan/infinity output')
        print(np.isnan(result).sum(axis=0).sum())
        print(np.isinf(result).sum(axis=0).sum())
        if np.isnan(result).sum(axis=0).sum() > 0:
            raise Exception('trouble : nan in supervised output')
        if np.isinf(result).sum(axis=0).sum() > 0:
            raise Exception('trouble : nan in supervised output')





