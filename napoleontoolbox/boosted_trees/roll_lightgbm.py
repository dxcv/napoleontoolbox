#!/usr/bin/env python3
# coding: utf-8

""" Basis of rolling models.
Examples
--------
# >>> roll_xgb = RollingXGB(X, y)
# >>> for pred_eval, pred_test in roll_xgb(256, 64):
# >>>     plot(pred_eval, pred_test)
"""

# Built-in packages

# External packages
import numpy as np
from matplotlib import pyplot as plt
import torch

# Local packages
# from fynance.models.xgb import XGBData

from napoleontoolbox.backtest.dynamic_plot_backtest import DynaPlotBackTest

import lightgbm as lgb

import shap

# Set plot style
plt.style.use('seaborn')

__all__ = ['_RollingBasis', 'RollLightGbm']

#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-05-06 20:16:31
# @Last modified by: ArthurBernard
# @Last modified time: 2019-09-26 17:10:55

""" Basis of neural networks models. """

# Built-in packages

# External packages
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch
import torch.nn as nn
from torch.autograd import Variable

# Local packages

__all__ = ['RollLightGbm']





class _RollingBasis:
    """ Base object to roll a neural network model.
    Rolling over a time axis with a train period from `t - n` to `t` and a
    testing period from `t` to `t + s`.
    Parameters
    ----------
    X, y : array_like
        Respectively input and output data.
    f : callable, optional
        Function to transform target, e.g. ``torch.sign`` function.
    index : array_like, optional
        Time index of data.
    Methods
    -------
    __call__
    Attributes
    ----------
    n, s, r : int
        Respectively size of training, testing and rolling period.
    b, e, T : int
        Respectively batch size, number of epochs and size of entire dataset.
    t : int
        The current time period.
    y_eval, y_test : np.ndarray[ndim=1 or 2, dtype=np.float64]
        Respectively evaluating (or training) and testing predictions.
    """

    # TODO : other methods
    def __init__(self, X, y, f=None, index=None):
        """ Initialize shape of target. """
        self.T = X.shape[0]
        self.y_shape = y.shape

        if f is None:
            self.f = lambda x: x

        else:
            self.f = f

        if index is None:
            self.idx = np.arange(self.T)

        else:
            self.idx = index

    # TODO : fix callable method to overwritten problem with torch.nn.Module
    def __call__(self, train_period, test_period, start=0, end=None,
                 roll_period=None, eval_period=None):
        """ Callable method to set target features data, and model.
        Parameters
        ----------
        train_period, test_period : int
            Size of respectively training and testing sub-periods.
        start : int, optional
            Starting observation, default is first observation.
        end : int, optional
            Ending observation, default is last observation.
        roll_period : int, optional
            Size of the rolling period, default is the same size of the
            testing sub-period.
        eval_period : int, optional
            Size of the evaluating period, default is the same size of the
            testing sub-period if training sub-period is large enough.
        batch_size : int, optional
            Size of a training batch, default is 64.
        epochs : int, optional
            Number of epochs on the same subperiod, default is 1.
        Returns
        -------
        _RollingBasis
            The rolling basis model.
        """
        # Set size of subperiods
        self.n = train_period
        self.s = test_period
        self.r = test_period if roll_period is None else roll_period


        # Set boundary of period
        self.T = self.T if end is None else min(self.T, end)
        self.t = max(self.n - self.r, min(start, self.T - self.n - self.s))
        self.b = int((self.n - self.s) / 2)

        return self

    def __iter__(self):
        """ Set iterative method. """
        self.y_eval = np.zeros(self.y_shape)
        self.y_test = np.zeros(self.y_shape)
        self.loss_train = []
        self.loss_eval = []
        self.loss_test = []

        return self

    def __next__(self):
        """ Incrementing method. """
        # TODO : to finish
        # Time forward incrementation
        self.t += self.r

        if self.t > (self.T-1):
            raise StopIteration

        if self.t + self.s > self.T:
            # output to train would need the future : we do not retrain the networl
            return slice(self.t - self.r, self.t), slice(self.t, self.T)


        training_range = range(self.t - self.n, self.t-self.s)
        loss_batch = self._train(
            X=self.X[training_range],
            y=self.f(self.y[training_range]),
        )
        self.loss_train += [loss_batch]

        # for t in range(self.t - self.n, self.t-self.s, self.b):
        #     # Set new train periods
        #     s = min(t + self.b, self.t)
        #     if s >= self.t - 1:
        #         continue
        #     train_slice = slice(t, s)
        #     #print('train_slice '+str(train_slice))
        #     # Train model
        #     loss_epoch += lo.item()
        #
        #     self.loss_train += [loss_epoch]

        # Set eval and test periods
        return slice(self.t - self.r, self.t), slice(self.t, self.t + self.s)

    def run(self, backtest_plot=True, backtest_kpi=True, figsize=(9, 6)):
        """ Run neural network model.
        Parameters
        ----------
        backtest_plot : bool, optional
            If True, display plot of backtest performances.
        backtest_kpi : bool, optional
            If True, display kpi of backtest performances.
        """
        perf_eval = 100. * np.ones(self.y.shape)
        perf_test = 100. * np.ones(self.y.shape)
        # Set dynamic plot object
        f, (ax_1, ax_2) = plt.subplots(2, 1, figsize=figsize)
        plt.ion()
        ax_loss = DynaPlotBackTest(
            f, ax_1, title='Model loss', ylabel='Loss', xlabel='Epochs',
            yscale='log', tick_params={'axis': 'x', 'labelsize': 10}
        )
        ax_perf = DynaPlotBackTest(
            f, ax_2, title='Model perf.', ylabel='Perf.',
            xlabel='Date', yscale='log',
            tick_params={'axis': 'x', 'rotation': 30, 'labelsize': 10}
        )

        # TODO : get stats, loss, etc.
        # TODO : plot loss, perf, etc.
        for eval_slice, test_slice in self:
            # Predict on training and testing period
            self.y_eval[eval_slice] = self.sub_predict(self.X[eval_slice])
            self.y_test[test_slice] = self.sub_predict(self.X[test_slice])
            # Compute losses
            self.loss_eval += [self.criterion(
                torch.from_numpy(self.y_eval[eval_slice]),
                self.y[eval_slice]
            ).item()]
            self.loss_test += [self.criterion(
                torch.from_numpy(self.y_test[test_slice]),
                self.y[test_slice]
            ).item()]

            if backtest_kpi:
                # Display %
                pct = self.t - self.n - self.s
                pct = pct / (self.T - self.n - self.T % self.s)
                txt = '{:5.2%} is done | '.format(pct)
                txt += 'Eval loss is {:5.2} | '.format(self.loss_eval[-1])
                txt += 'Test loss is {:5.2} | '.format(self.loss_test[-1])
                print(txt, end='\r')

            if backtest_plot:
                # Set performances of training period
                returns = np.sign(self.y_eval[eval_slice]) * self.y[eval_slice].numpy()
                cumret = np.exp(np.cumsum(returns, axis=0))
                perf_eval[eval_slice] = perf_eval[self.t - self.r - 1] * cumret

                # Set performances of estimated period
                returns = np.sign(self.y_test[test_slice]) * self.y[test_slice].numpy()
                cumret = np.exp(np.cumsum(returns, axis=0))
                perf_test[test_slice] = perf_test[self.t - 1] * cumret

                ax_loss.ax.clear()
                ax_perf.ax.clear()
                # Plot loss
                ax_loss.plot(np.array([self.loss_test]).T, names='Test',
                             col='BuGn', lw=2.)
                ax_loss.plot(
                    np.array([self.loss_eval]).T, names='Eval', col='YlOrBr',
                    loc='upper right', ncol=2, fontsize=10, handlelength=0.8,
                    columnspacing=0.5, frameon=True, lw=1.,
                )

                # Plot perf
                ax_perf.plot(
                    perf_test[: self.t + self.s],
                    x=self.idx[: self.t + self.s],
                    names='Test set', col='GnBu', lw=1.7, unit='perf',
                )
                ax_perf.plot(
                    perf_eval[: self.t], x=self.idx[: self.t],
                    names='Eval set', col='OrRd', lw=1.2, unit='perf'
                )
                ax_perf.ax.legend(loc='upper left', fontsize=10, frameon=True,
                                  handlelength=0.8, ncol=2, columnspacing=0.5)
                f.canvas.draw()
                # plt.draw()

        return self





class RollLightGbm(_RollingBasis):
    """ Rolling version of the vanilla neural network model.
    TODO:
    - fix train and predict methods
    - finish docstring
    - finish methods
    """

    def __init__(self, X, y, num_boost_round = 50, early_stopping_rounds = 50, x_type = None, y_type = None):
        """ Initialize rolling multi-layer perceptron model. """
        _RollingBasis.__init__(self, X, y)
        self.set_data(X=X, y=y, x_type=x_type, y_type=y_type)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.models = [None] * y.shape[1]

    def set_roll_period(self, train_period, test_period, start=0, end=None,
                        roll_period=None, eval_period=None):
        """ Callable method to set target features data, and model.
        Parameters
        ----------
        train_period, test_period : int
            Size of respectively training and testing sub-periods.
        start : int, optional
            Starting observation, default is first observation.
        end : int, optional
            Ending observation, default is last observation.
        roll_period : int, optional
            Size of the rolling period, default is the same size of the
            testing sub-period.
        eval_period : int, optional
            Size of the evaluating period, default is the same size of the
            testing sub-period if training sub-period is large enough.
        batch_size : int, optional
            Size of a training batch, default is 64.
        epochs : int, optional
            Number of epochs, default is 1.
        Returns
        -------
        _RollingBasis
            The rolling basis model.
        """
        return _RollingBasis.__call__(
            self, train_period=train_period, test_period=test_period,
            start=start, end=end, roll_period=roll_period,
            eval_period=eval_period
        )

    def _train(self, X, y):

        params = {
            "objective": "regression",
            "learning_rate": 0.05,
            "metric": "rmse",
#            "min_data": 1,
            'verbose': -1
        }

        for output in range(y.shape[1]):
            # lgb training

            y_train = y[:,output]
            prev_trained_model = self.models[output]


            # X_half_1 = X[:int(X.shape[0] / 2)]
            # X_half_2 = X[int(X.shape[0] / 2):]
            #
            # y_half_1 = y_train[:int(X.shape[0] / 2)]
            # y_half_2 = y_train[int(X.shape[0] / 2):]
            #
            # d_half_1 = lgb.Dataset(X_half_1, label=y_half_1)
            # d_half_2 = lgb.Dataset(X_half_2, label=y_half_2)
            #
            # watchlist_1 = [d_half_1, d_half_2]
            #
            #
            # if prev_trained_model is None:
            #     model = lgb.train(params, train_set=d_half_1, num_boost_round=self.num_boost_round, valid_sets=watchlist_1)
            # else :
            #     model = lgb.train(params, train_set=d_half_1, num_boost_round=self.num_boost_round, valid_sets=watchlist_1,
            #                       init_model=prev_trained_model)


            lgb_dataset = lgb.Dataset(X, label=y_train)#, categorical_feature=categorical_features, free_raw_data=False)
                # to watch if
            watchlist = [lgb_dataset]
            prev_trained_model = self.models[output]
            if prev_trained_model is None:
                model = lgb.train(params, train_set=lgb_dataset, num_boost_round=self.num_boost_round, valid_sets=watchlist)
            else:
                model = lgb.train(params, train_set=lgb_dataset, num_boost_round=self.num_boost_round, valid_sets=watchlist,
                                  init_model=prev_trained_model)

            self.models[output] = model




    def sub_predict(self, X):
        """ Predict. """
        if len(self.models) != self.y.shape[1]:
            raise RuntimeError('light gbm have not been properly trained')
        # lgb prediction
        predictions = []
        for model in self.models :
            y_pred =  model.predict(X)# num_iteration=self.model.best_iteration)
            predictions.append(y_pred)
        preds = np.array(predictions).T
        #print('averaged predictions '+str(preds.mean().mean()))
        return preds

    def set_data(self, X, y, x_type=None, y_type=None):
        """ Set data inputs and outputs.
        Parameters
        ----------
        X, y : array-like
            Respectively input and output data.
        x_type, y_type : torch.dtype
            Respectively input and ouput data types. Default is `None`.
        """
        if hasattr(self, 'N') and self.N != X.size(1):
            raise ValueError('X must have {} input columns'.foramt(self.N))

        if hasattr(self, 'M') and self.M != y.size(1):
            raise ValueError('y must have {} output columns'.format(self.M))

        self.X = self._set_data(X, dtype=x_type)
        self.y = self._set_data(y, dtype=y_type)
        self.T, self.N = self.X.shape
        T_veri, self.M = self.y.shape

        if self.T != T_veri:
            raise ValueError('{} time periods in X differents of {} time \
                             periods in y'.format(self.T, T_veri))

        return self

    def _set_data(self, X, dtype=None):
        """ Convert array-like data to tensor. """
        # TODO : Verify dtype of data torch tensor
        if isinstance(X, np.ndarray):

            return X

        elif isinstance(X, pd.DataFrame):
            # TODO : Verify memory efficiancy
            return X.values


        else:
            raise ValueError('Unkwnown data type: {}'.format(type(X)))

def eval_predictor_importance(self, features, features_names):
        explainer_shap = shap.GradientExplainer(model=self,
                                            data=features)
        # Fit the explainer on a subset of the data (you can try all but then gets slower)
        shap_values = explainer_shap.shap_values(X=features,
                                                 ranked_outputs=True)

        predictors_shap_values = shap_values[0]
        predictors_feature_order = np.argsort(np.sum(np.mean(np.abs(predictors_shap_values), axis=0), axis=0))

        predictors_left_pos = np.zeros(len(predictors_feature_order))

        predictors_class_inds = np.argsort([-np.abs(predictors_shap_values[i]).mean() for i in range(len(predictors_shap_values))])
        for i, ind in enumerate(predictors_class_inds):
            predictors_global_shap_values = np.abs(predictors_shap_values[ind]).mean(0)
            predictors_left_pos += predictors_global_shap_values[predictors_feature_order]

        predictors_ds = {}
        predictors_ds['features'] = np.asarray(features_names)[predictors_feature_order]
        predictors_ds['values'] = predictors_left_pos
        predictors_features_df = pd.DataFrame.from_dict(predictors_ds)
        values = {}
        for index, row in predictors_features_df.iterrows():
            values[row['features']]=row['values']

        return values



