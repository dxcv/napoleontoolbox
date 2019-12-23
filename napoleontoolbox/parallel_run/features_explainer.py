#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from napoleontoolbox.neural_net import roll_multi_layer_lstm

from napoleontoolbox.neural_net import roll_multi_layer_perceptron
from napoleontoolbox.boosted_trees import roll_lightgbm


from napoleontoolbox.utility import weights

import torch
import torch.nn as nn


class AbstractRunner(ABC):
    def __init__(self, supervised_utility_path, features_path, features_names_path, returns_path, root='../data/', user = 'napoleon', lr=0.001):
        super().__init__()
        self.root =  root
        self.user =  user
        self.supervised_utility_path = supervised_utility_path
        self.features_path = features_path
        self.features_names_path = features_names_path
        self.returns_path = returns_path
        self.lr = lr

    @abstractmethod
    def runTrial(self,saver, seed, sup, layers, epochs, n_past_features, n, s, whole_history, advance_feature, advance_signal,
                 normalize, activation_string, convolution):
        pass


class SimpleExplainer(AbstractRunner):
    def runTrial(self, saver, seed, sup, layers, epochs, n_past_features, n, s, whole_history, advance_feature,
                 advance_signal,
                 normalize, activation_string, convolution):
        param = '_' + str(seed) + '_' + str(n_past_features) + '_' + str(n) + '_' + str(layers) + '_' + str(
            whole_history) + '_' + str(advance_feature) + '_' + str(advance_signal) + '_' + str(normalize) + '_' + str(
            epochs) + '_' + str(s) + '_' + activation_string + '_' + str(convolution)
        param = param.replace(' ', '')
        param = param.replace(',', '_')
        param = param.strip()

        print('Launching')
        print(param)

        meArg = (
            seed, sup, param, layers, epochs, n_past_features, n, s, whole_history, advance_feature, advance_signal,
            normalize,
            activation_string, convolution)

        supervisors = {}
        supervisors['f_minVar'] = 0
        supervisors['f_maxMean'] = 1
        supervisors['f_sharpe'] = 2
        supervisors['f_MeanVar'] = 3
        supervisors['f_calmar'] = 4
        supervisors['f_drawdown'] = 5

        df = pd.read_pickle(self.root + self.returns_path)
        print(df.columns)

        print(df.columns)
        dates = pd.to_datetime(df['Date'])
        df['Date'] = dates
        df = df.set_index('Date')
        df = df.fillna(method='ffill')
        T = df.index.size

        result = np.load(self.root + self.user + '_' + str(s) + self.supervised_utility_path )

        np.random.seed(seed)
        torch.manual_seed(seed)
        # Set data
        features = np.load(
            self.root + self.user + '_' + str(normalize) + '_' + str(whole_history) + '_' + str(
                advance_feature) + '_' + str(n_past_features) + self.features_path)

        features_names = np.load(
            self.root + self.user + '_' + str(normalize) + '_' + str(whole_history) + '_' + str(
                advance_feature) + '_' + str(n_past_features) + self.features_names_path)

        # X = features[s:-s]
        # y = result[s:-s, :, supervisors[sup]]
        X = features[s:]
        y = result[s:, :, supervisors[sup]]
        df = df.iloc[s:]
        print('predictors')
        print(X.shape)
        print('utility')
        print(y.shape)
        print('prices')
        print(df.shape)

        sup = sup + param

        # convolution 0 : perceptron
        # convolution 1 : LSTM
        # convolution 2 : xgboost

        if whole_history:
            if convolution == 2:
                print('no whole time history with ensembling method')
                return
            if convolution == 0:
                print('flattening predictor time series for perceptron')
                _X = np.empty((X.shape[0], X.shape[1] * X.shape[2]), dtype=np.float32)
                for l in range(X.shape[0]):
                    temp = np.transpose(X[l, :, :])
                    _X[l, :] = temp.flatten()
                X = _X

        if not whole_history:
            if convolution == 1:
                print('only whole time history with lstm')
                # print('adding one virtual time stamp')
                # X = X[..., np.newaxis, :]
                return

        print('number of nan/infinity features')
        print(np.isnan(X).sum(axis=0).sum())
        print(np.isinf(X).sum(axis=0).sum())
        print('number of nan/infinity output')
        print(np.isnan(y).sum(axis=0).sum())
        print(np.isinf(y).sum(axis=0).sum())

        print(np.count_nonzero(~np.isnan(X)))

        activation_function = None
        if activation_string == 'sigmoid':
            activation_function = torch.sigmoid
        if activation_string == 'relu':
            activation_function = F.relu

        if convolution == 1:
            # the number of figures
            input_size = X.shape[2]
            hidden_size = 32
            num_layers = 1
            num_classes = y.shape[1]
            tm = roll_multi_layer_lstm.RollMultiLayerLSTM(
                X=X,
                y=y,
                num_classes=num_classes,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                # nn.Softmax/nn.Softmin can be good activations for this problem
                x_type=torch.float32,
                y_type=torch.float32
                # activation_kwargs={'dim':1} # Parameter needed for nn.Softmax/nn.Softmin
            )
            tm.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=self.lr, betas=(0.9, 0.999), amsgrad=True)
            tm = tm.set_roll_period(n, s, epochs=epochs)
        elif convolution == 0:
            tm = roll_multi_layer_perceptron.RollMultiLayerPerceptron(
                X=X,
                y=y,
                layers=layers,
                activation=activation_function,  # nn.Softmax/nn.Softmin can be good activations for this problem
                x_type=torch.float32,
                y_type=torch.float32,
                # activation_kwargs={'dim':1} # Parameter needed for nn.Softmax/nn.Softmin
            )
            tm.set_optimizer(nn.MSELoss, torch.optim.Adam, lr=self.lr, betas=(0.9, 0.999), amsgrad=True)
            tm = tm.set_roll_period(n, s, epochs=epochs)
        elif convolution == 2:
            tm = roll_lightgbm.RollLightGbm(
                X=X,
                y=y
            )
            tm = tm.set_roll_period(n, s)
        features_df = tm.unroll_features(dates, features_names)
        return features_df


