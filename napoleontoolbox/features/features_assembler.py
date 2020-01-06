#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod
from napoleontoolbox.features import features_type
import sys

import pandas as pd

import numpy as np

import torch
from napoleontoolbox.boosted_trees import roll_lightgbm
from napoleontoolbox.features import features_type

class AbstractAssembler(ABC):
    def __init__(self, features_path, returns_path, root='../data/', user = 'napoleon', features_saving_path = '_features.npy', features_names_saving_path = '_features_names.npy'):
        super().__init__()
        self.root =  root
        self.user =  user
        self.features_path = features_path
        self.returns_path = returns_path
        self.features_saving_path = features_saving_path
        self.features_names_saving_path = features_names_saving_path


    @abstractmethod
    def assembleFeature(self, feature_type, n_past_features):
        pass


class FeaturesAssembler(AbstractAssembler):

    def reassembleAdvancedFeaturesForClusterization(self, feature_type, n_past_features = 21, simple = True):

        if (feature_type is features_type.FeaturesType.HISTORY or feature_type is features_type.FeaturesType.HISTORY_ADVANCED) and n_past_features is None:
            return

        if (feature_type is features_type.FeaturesType.STANDARD or feature_type is features_type.FeaturesType.STANDARD_ADVANCED) and n_past_features is not None:
            return

        print('feature_type' + str(feature_type))
        print('n_past_features' + str(n_past_features))


        np.random.seed(0)
        torch.manual_seed(0)

        advanced_features, features_names = self.preprocessFeature()
        df, strats, strat_features_names = self.preprocessReturns()

        # quotes_df=quotes_df.sort_values(by='date', ascending=True)
        # quotes_df.head()

        print(df.columns)
        print(df.shape)

        print(advanced_features.columns)
        print(advanced_features.shape)

        # Computationnal period (default 1 year)


        ##===================##
        ##  Setting targets  ##
        ##===================##


        print('merging')
        df_ret = pd.merge(df, advanced_features, how='left', on=['Date'])

        df_ret = df_ret.replace([np.inf, -np.inf], np.nan)
        df_ret=df_ret.fillna(0.)

        print('number of infs in advanced features')
        print(np.isinf(df_ret[features_names]).sum(axis=0))

        print('number of infs in returns')
        print(np.isinf(df_ret[strats]).sum(axis=0))

        print('number of nans in advanced features')
        print(np.isnan(df_ret[features_names]).sum(axis=0))

        print('number of nans in returns')
        print(np.isnan(df_ret[strats]).sum(axis=0))

        if simple:
            return strats, strat_features_names, features_names, df_ret


        ret_df = df_ret[strats]
        ret = ret_df.values

        feat = df_ret[strat_features_names + features_names].values

        T = df.index.size
        N = len(strats)

        all_features_length = len(features_names) + len(strat_features_names)

        if feature_type is features_type.FeaturesType.HISTORY_ADVANCED:
            features = np.zeros([T, n_past_features, N + all_features_length], np.float32)
            predictor_names = strats  + features_names + strat_features_names

        if feature_type is features_type.FeaturesType.HISTORY:
            features = np.zeros([T, n_past_features, N], np.float32)


        if feature_type is features_type.FeaturesType.HISTORY or feature_type is features_type.FeaturesType.HISTORY_ADVANCED:
            for t in range(n_past_features, T):
                np.random.seed(0)
                torch.manual_seed(0)
                # the output to predict cannot be computed in the future
                # we still assemble the predictors
                # Set input data
                t_n = min(max(t - n_past_features, 0), T)
                F = feat[t_n: t, :]
                X_back = ret[t_n: t, :]

                if feature_type is features_type.FeaturesType.HISTORY_ADVANCED or feature_type is features_type.FeaturesType.HISTORY:
                    if feature_type is features_type.FeaturesType.HISTORY_ADVANCED:
                        X_back = np.concatenate((X_back, F), axis=1)
                        predictor_names = features_names + strat_features_names
                    if feature_type is features_type.FeaturesType.HISTORY:
                        predictor_names = strats

                    features[t: t + 1] = X_back

                if t % 500 == 0:
                    print('{:.2%}'.format(t / T))
                # we compute the utility output to predict only if not in future
        else :
            if feature_type is features_type.FeaturesType.STANDARD_ADVANCED:

                #features = np.zeros([T, N + all_features_length], np.float32)
                features = feat
                predictor_names =  features_names + strat_features_names

            if feature_type is features_type.FeaturesType.STANDARD:
                #features = np.zeros([T, N], np.float32)
                features = ret
                predictor_names = strats

        return  predictor_names, features





    def preprocessFeature(self):

        features = pd.read_pickle(self.root + self.features_path)
        features_names = [col for col in list(features.columns) if col!='Date']

        features['Date'] = pd.to_datetime(features['Date'])


        # MM 3 mois
        for col in features_names:
            features["MM63_{}".format(col)] = features[col] / features[col].shift(1).rolling(window=63).mean() - 1
        # Z score 21 jours
        for col in features_names:
            features["Z_{}".format(col)] = (features[col] - features[col].rolling(window=21).mean()) / features[col].rolling(
                21).std()
        # Z score 63 jours
        for col in features_names:
            features["Z63_{}".format(col)] = (features[col].rolling(window=5).mean() - features[col].rolling(
                window=63).mean()) \
                                           / features[col].rolling(63).std()
        # MM 126 jours
        for col in features_names:
            features["MM126_{}".format(col)] = features[col] / features[col].shift(1).rolling(window=126).mean() - 1
        # MM 252 jours
        for col in features_names:
            features["MM252_{}".format(col)] = features[col] / features[col].shift(1).rolling(window=252).mean() - 1
        # MM 20 jours
        for col in features_names:
            features["MM20_{}".format(col)] = features[col] / features[col].shift(1).rolling(window=20).mean() - 1
        # ecart type:
        for col in features_names:
            features["vol20_{}".format(col)] = features[col].rolling(window=20).std() / features[col].rolling(
                window=20).mean()

        for col in features_names:
            features[col] = features[col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.)

        for col in features_names:
            features[col] = (features[col] - features[col].mean()) / features[col].std(ddof=0)

        for col in features_names:
            features["quantile21_0_{}".format(col)] = features[col].rolling(window=20).quantile(0.)
            features["quantile21_25_{}".format(col)] = features[col].rolling(window=20).quantile(0.25)
            features["quantile21_50_{}".format(col)] = features[col].rolling(window=20).quantile(0.5)
            features["quantile21_55_{}".format(col)] = features[col].rolling(window=20).quantile(0.75)
            features["quantile21_75_{}".format(col)] = features[col].rolling(window=20).quantile(1.)

            features["quantile63_0_{}".format(col)] = features[col].rolling(window=63).quantile(0.)
            features["quantile63_25_{}".format(col)] = features[col].rolling(window=63).quantile(0.25)
            features["quantile63_50_{}".format(col)] = features[col].rolling(window=63).quantile(0.5)
            features["quantile63_55_{}".format(col)] = features[col].rolling(window=63).quantile(0.75)
            features["quantile63_75_{}".format(col)] = features[col].rolling(window=63).quantile(1.)

            features["quantile126_0_{}".format(col)] = features[col].rolling(window=126).quantile(0.)
            features["quantile126_25_{}".format(col)] = features[col].rolling(window=126).quantile(0.25)
            features["quantile126_50_{}".format(col)] = features[col].rolling(window=126).quantile(0.5)
            features["quantile126_55_{}".format(col)] = features[col].rolling(window=126).quantile(0.75)
            features["quantile126_75_{}".format(col)] = features[col].rolling(window=126).quantile(1.)

            features["quantile252_0_{}".format(col)] = features[col].rolling(window=252).quantile(0.)
            features["quantile252_25_{}".format(col)] = features[col].rolling(window=252).quantile(0.25)
            features["quantile252_50_{}".format(col)] = features[col].rolling(window=252).quantile(0.5)
            features["quantile252_55_{}".format(col)] = features[col].rolling(window=252).quantile(0.75)
            features["quantile252_75_{}".format(col)] = features[col].rolling(window=252).quantile(1.)

            features["mean_21_{}".format(col)] = features[col].rolling(window=21).mean()
            features["mean_63_{}".format(col)] = features[col].rolling(window=63).mean()
            features["mean_126_{}".format(col)] = features[col].rolling(window=126).mean()
            features["mean_252_{}".format(col)] = features[col].rolling(window=252).mean()

        features_names = [col for col in list(features.columns) if col!='Date']
        predictor_names = None
        features['Date'] = pd.to_datetime(features['Date'])

        return features , features_names

    def preprocessReturns(self):

        df = pd.read_pickle(self.root + self.returns_path)
        df['Date'] = pd.to_datetime(df['Date'])
        strats = [col for col in list(df.columns) if col != 'Date']

        #@todo later
        #corr_strats = [col1+'_'+col2 for col1 in strats for col2 in strats]

        df = df.set_index('Date')


        df = df.fillna(method='ffill').fillna(method='bfill')


        # MM 3 mois
        for col in strats:
            df["MM63_{}".format(col)] = df[col] / df[col].shift(1).rolling(window=63).mean() - 1
        # Z score 21 jours
        for col in strats:
            df["Z_{}".format(col)] = (df[col] - df[col].rolling(window=21).mean()) / df[col].rolling(
                21).std()
        # Z score 63 jours
        for col in strats:
            df["Z63_{}".format(col)] = (df[col].rolling(window=5).mean() - df[col].rolling(
                window=63).mean()) \
                                           / df[col].rolling(63).std()
        # MM 126 jours
        for col in strats:
            df["MM126_{}".format(col)] = df[col] / df[col].shift(1).rolling(window=126).mean() - 1
        # MM 252 jours
        for col in strats:
            df["MM252_{}".format(col)] = df[col] / df[col].shift(1).rolling(window=252).mean() - 1
        # MM 20 jours
        for col in strats:
            df["MM20_{}".format(col)] = df[col] / df[col].shift(1).rolling(window=20).mean() - 1
        # ecart type:
        for col in strats:
            df["vol20_{}".format(col)] = df[col].rolling(window=20).std() / df[col].rolling(
                window=20).mean()

        print('computing returns')
        for col in strats:
            df[col] = df[col].pct_change().fillna(0.)

        for col in strats:
            df["quantile21_0_{}".format(col)] = df[col].rolling(window=20).quantile(0.)
            df["quantile21_25_{}".format(col)] = df[col].rolling(window=20).quantile(0.25)
            df["quantile21_50_{}".format(col)] = df[col].rolling(window=20).quantile(0.5)
            df["quantile21_55_{}".format(col)] = df[col].rolling(window=20).quantile(0.75)
            df["quantile21_75_{}".format(col)] = df[col].rolling(window=20).quantile(1.)

            df["quantile63_0_{}".format(col)] = df[col].rolling(window=63).quantile(0.)
            df["quantile63_25_{}".format(col)] = df[col].rolling(window=63).quantile(0.25)
            df["quantile63_50_{}".format(col)] = df[col].rolling(window=63).quantile(0.5)
            df["quantile63_55_{}".format(col)] = df[col].rolling(window=63).quantile(0.75)
            df["quantile63_75_{}".format(col)] = df[col].rolling(window=63).quantile(1.)

            df["quantile126_0_{}".format(col)] = df[col].rolling(window=126).quantile(0.)
            df["quantile126_25_{}".format(col)] = df[col].rolling(window=126).quantile(0.25)
            df["quantile126_50_{}".format(col)] = df[col].rolling(window=126).quantile(0.5)
            df["quantile126_55_{}".format(col)] = df[col].rolling(window=126).quantile(0.75)
            df["quantile126_75_{}".format(col)] = df[col].rolling(window=126).quantile(1.)

            df["quantile252_0_{}".format(col)] = df[col].rolling(window=252).quantile(0.)
            df["quantile252_25_{}".format(col)] = df[col].rolling(window=252).quantile(0.25)
            df["quantile252_50_{}".format(col)] = df[col].rolling(window=252).quantile(0.5)
            df["quantile252_55_{}".format(col)] = df[col].rolling(window=252).quantile(0.75)
            df["quantile252_75_{}".format(col)] = df[col].rolling(window=252).quantile(1.)

            df["mean_21_{}".format(col)] = df[col].rolling(window=21).mean()
            df["mean_63_{}".format(col)] = df[col].rolling(window=63).mean()
            df["mean_126_{}".format(col)] = df[col].rolling(window=126).mean()
            df["mean_252_{}".format(col)] = df[col].rolling(window=252).mean()

        strat_features = [col for col in list(df.columns) if col!='Date']
        return df , strats, strat_features


    def assembleFeature(self, feature_type, n_past_features):

        if (feature_type is features_type.FeaturesType.HISTORY or feature_type is features_type.FeaturesType.HISTORY_ADVANCED) and n_past_features is None:
            return

        if (feature_type is features_type.FeaturesType.STANDARD or feature_type is features_type.FeaturesType.STANDARD_ADVANCED) and n_past_features is not None:
            return

        print('feature_type' + str(feature_type))
        print('n_past_features' + str(n_past_features))


        np.random.seed(0)
        torch.manual_seed(0)

        advanced_features, features_names = self.preprocessFeature()
        df, strats, strat_features_names = self.preprocessReturns()

        # quotes_df=quotes_df.sort_values(by='date', ascending=True)
        # quotes_df.head()

        print(df.columns)
        print(df.shape)

        print(advanced_features.columns)
        print(advanced_features.shape)

        # Computationnal period (default 1 year)


        ##===================##
        ##  Setting targets  ##
        ##===================##


        print('merging')
        df_ret = pd.merge(df, advanced_features, how='left', on=['Date'])

        df_ret = df_ret.replace([np.inf, -np.inf], np.nan)
        df_ret=df_ret.fillna(0.)

        print('number of infs in advanced features')
        print(np.isinf(df_ret[features_names]).sum(axis=0))

        print('number of infs in returns')
        print(np.isinf(df_ret[strats]).sum(axis=0))

        print('number of nans in advanced features')
        print(np.isnan(df_ret[features_names]).sum(axis=0))

        print('number of nans in returns')
        print(np.isnan(df_ret[strats]).sum(axis=0))


        ret_df = df_ret[strats]
        ret = ret_df.values

        feat = df_ret[strat_features_names + features_names].values

        T = df.index.size
        N = len(strats)

        all_features_length = len(features_names) + len(strat_features_names)

        if feature_type is features_type.FeaturesType.HISTORY_ADVANCED:
            features = np.zeros([T, n_past_features, N + all_features_length], np.float32)
            predictor_names = strats  + features_names + strat_features_names

        if feature_type is features_type.FeaturesType.HISTORY:
            features = np.zeros([T, n_past_features, N], np.float32)


        if feature_type is features_type.FeaturesType.HISTORY or feature_type is features_type.FeaturesType.HISTORY_ADVANCED:
            for t in range(n_past_features, T):
                np.random.seed(0)
                torch.manual_seed(0)
                # the output to predict cannot be computed in the future
                # we still assemble the predictors
                # Set input data
                t_n = min(max(t - n_past_features, 0), T)
                F = feat[t_n: t, :]
                X_back = ret[t_n: t, :]

                if feature_type is features_type.FeaturesType.HISTORY_ADVANCED or feature_type is features_type.FeaturesType.HISTORY:
                    if feature_type is features_type.FeaturesType.HISTORY_ADVANCED:
                        X_back = np.concatenate((X_back, F), axis=1)
                        predictor_names = features_names + strat_features_names
                    if feature_type is features_type.FeaturesType.HISTORY:
                        predictor_names = strats


                    features[t: t + 1] = X_back

                if t % 500 == 0:
                    print('{:.2%}'.format(t / T))
                # we compute the utility output to predict only if not in future
        else :
            if feature_type is features_type.FeaturesType.STANDARD_ADVANCED:

                #features = np.zeros([T, N + all_features_length], np.float32)
                features = feat
                predictor_names =  features_names + strat_features_names

            if feature_type is features_type.FeaturesType.STANDARD:
                #features = np.zeros([T, N], np.float32)
                features = ret
                predictor_names = strats

        print('saved files')
        print('number of nan/infinity features')
        print(np.isnan(features).sum(axis=0).sum())
        print(np.isinf(features).sum(axis=0).sum())
        if np.isnan(features).sum(axis=0).sum() > 0:
            raise Exception('nan values for assembled features')
        if np.isinf(features).sum(axis=0).sum() > 0:
            raise Exception('inf values for assembled features')

        print('saving file')
        features_saving_path = self.root  + self.user + '_' + feature_type.name + '_' + str(n_past_features) + self.features_saving_path
        print(features_saving_path)
        np.save(features_saving_path, features)

        features_names_saving_path = self.root  + self.user + '_' + feature_type.name + '_' + str(n_past_features) + self.features_names_saving_path
        print(features_names_saving_path)
        np.save(features_names_saving_path, predictor_names)







