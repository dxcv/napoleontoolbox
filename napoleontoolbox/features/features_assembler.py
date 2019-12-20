#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod

import sys

import pandas as pd

import numpy as np

import torch


class AbstractAssembler(ABC):
    def __init__(self, features_path, returns_path, root='../data/', user = 'napoleon', lr=0.001):
        super().__init__()
        self.root =  root
        self.user =  user
        self.features_path = features_path
        self.returns_path = returns_path
        self.lr = lr

        #advanced_features = pd.read_excel(root + '/features_completed.xlsx',
        #                                  sheet_name='Features')

    @abstractmethod
    def assembleFeature(self,saver, seed, normalize, advanced_features, whole_history, n_past_features, s):
        pass


class FeaturesAssembler(AbstractAssembler):
    def assembleFeature(self, saver, seed, normalize, advanced_features, whole_history, n_past_features, s):
        print('advanced_features_in' + str(advanced_features))
        print('whole_history' + str(whole_history))
        print('n_past_features' + str(n_past_features))
        print('rebalancing ' + str(s))


        print('quotes loading')

        # path = root + '/sp_blend_under_blends_quotes.pkl'
        # df = pd.read_pickle(path)
        # pd.read_excel(open('tmp.xlsx', 'rb'),sheet_name = 'Sheet3')

        # df = pd.read_excel(root+'/SG2.xlsx',sheet_name='data')
        # df = pd.read_pickle(root + '/SG2.pkl')
        df = pd.read_pickle(self.root + self.returns_path)

        print(df.columns)

        # df = pd.read_excel(root+'/SG2.xlsx',sheet_name='data2')
        # df = pd.read_pickel(root + '/SG2_restreint.pkl', sheet_name='data')

        df['Date'] = pd.to_datetime(df['Date'])
        strats = list(df.columns)
        df = df.set_index('Date')

        df = df.fillna(method='ffill')
        T = df.index.size

        print('advanced features loading')
        # advanced_features = pd.read_excel('../data/Advanced_Market_Features.xlsx', sheet_name='Features')
        advanced_features = pd.read_pickle(self.root + self.features_path)
        features_names = list(advanced_features.columns)
        advanced_features['Date'] = pd.to_datetime(advanced_features['Date'])

        # quotes_df=quotes_df.sort_values(by='date', ascending=True)
        # quotes_df.head()

        print(df.columns)
        print(df.shape)

        print(advanced_features.columns)
        print(advanced_features.shape)

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

        if normalize:
            print('normalizing')
            feat[feat == -np.inf] = 0
            feat[feat == np.inf] = 0
            feat_stds = np.std(feat, axis=0)
            feat_means = np.mean(feat, axis=0)
            feat = (feat - feat_means) / feat_stds

        T = df.index.size
        N = df.columns.size

        if whole_history:
            if advanced_features:
                features = np.zeros([T, n_past_features, N + len(features_names)], np.float32)
            if not advanced_features:
                features = np.zeros([T, n_past_features, N], np.float32)
        else:
            if advanced_features:
                features = np.zeros([T, N + N * N + len(features_names)], np.float32)
            if not advanced_features:
                features = np.zeros([T, N + N * N], np.float32)


        # for t in range(max(n_past_features, s), T - s):
        for t in range(max(n_past_features, s), T):
            np.random.seed(0)
            torch.manual_seed(0)
            # the output to predict cannot be computed in the future
            # we still assemble the predictors
            # Set input data
            t_n = min(max(t - n_past_features, 0), T)
            F = feat[t_n: t, :]
            X_back = ret[t_n: t, :]
            if whole_history:
                if advanced_features:
                    # features = np.zeros([T, n_past_features, N + len(features_names)], np.float32)
                    X_back = np.concatenate((X_back, F), axis=1)
                if not advanced_features:
                    # features = np.zeros([T, n_past_features, N], np.float32)
                    features[t: t + 1] = X_back

            else:
                F_mean = np.nanmean(F, axis=0)
                std = np.std(X_back, axis=0).reshape([N, 1])
                mean = np.mean(X_back, axis=0)
                mat_std = std @ std.T
                mat_std[mat_std == 0.] = 1.
                mat_corr = np.cov(X_back, rowvar=False) / mat_std

                if advanced_features:
                    # features = np.zeros(
                    #     [T, n_past_features, N + len(features_names) + len(signal_features_names)],
                    #     np.float32)
                    features[t: t + 1] = np.transpose(
                        np.concatenate((mean, mat_corr.flatten(), F_mean), axis=0))
                if not advanced_features:
                    # features = np.zeros([T, n_past_features, N], np.float32)
                    features[t: t + 1] = np.transpose(
                        np.concatenate((mean, mat_corr.flatten()), axis=0))
            # features[t: t + 1] = mat_corr.flatten()
            # print('{:.2%}'.format(t / T))
            # Set input data
            t_n = min(max(t - n_past_features, 0), T)
            F = feat[t_n: t, :]
            X_back = ret[t_n: t, :]

            if whole_history:
                if advanced_features:
                    # features = np.zeros(
                    #     [T, n_past_features, N + len(features_names) + len(signal_features_names)],
                    #     np.float32)
                    X_back = np.concatenate((X_back, F), axis=1)
                if not advanced_features:
                    # features = np.zeros([T, n_past_features, N], np.float32)
                    features[t: t + 1] = X_back
                # if advanced_features_in:
                #     X_back = np.concatenate((X_back, F), axis=1)
                # if signal_features_in:
                #     X_back = np.concatenate((X_back, S), axis=1)
            else:
                F_mean = np.nanmean(F, axis=0)

                std = np.std(X_back, axis=0).reshape([N, 1])
                mean = np.mean(X_back, axis=0)
                mat_std = std @ std.T
                mat_std[mat_std == 0.] = 1.
                mat_corr = np.cov(X_back, rowvar=False) / mat_std

                if advanced_features:
                    # features = np.zeros([T, n_past_features, N + len(features_names)], np.float32)
                    features[t: t + 1] = np.transpose(
                        np.concatenate((mean, mat_corr.flatten(), F_mean), axis=0))
                if not advanced_features:
                    # features = np.zeros([T, n_past_features, N], np.float32)
                    features[t: t + 1] = np.transpose(
                        np.concatenate((mean, mat_corr.flatten()), axis=0))
            # features[t: t + 1] = mat_corr.flatten()
            if t % 500 == 0:
                print('{:.2%}'.format(t / T))
            # we compute the utility output to predict only if not in future

        print('saved files')
        print('number of nan/infinity features')
        print(np.isnan(features).sum(axis=0).sum())
        print(np.isinf(features).sum(axis=0).sum())
        if np.isnan(features).sum(axis=0).sum() > 0:
            raise Exception('nan values for assembled features')
        if np.isinf(features).sum(axis=0).sum() > 0:
            raise Exception('inf values for assembled features')

        print('saving file')
        np.save(self.root  + self.user + '_' + str(normalize) + '_' + str(whole_history) + '_' + str(
            advanced_features) + '_' + str(n_past_features) + '_features.npy', features)







