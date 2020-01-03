#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd

def process_weights(w, df, s, n):
    """ Process weight to respect constraints.

    Parameters
    ----------
    w : array_like
        Matrix of weights.
    df : pd.DataFrame
        Data of returns or prices.
    n, s : int
        Training and testing periods.

    Returns
    -------
    pd.DataFrame
        Dataframe of weight s.t. sum(w) = 1 and 0 <= w_i <= 1.

    """
    T, N = w.shape
    weight_mat = pd.DataFrame(index=df.index, columns=df.columns)
    idx = weight_mat.index

    def process(series):
        # True if less than 50% of obs. are constant
        return series.value_counts(dropna=False).max() < 0.5 * s

    previousValidWeights = None
    for t in range(n, T, s):
        t_s = min(T, t + s)
        weight_vect = np.zeros([N, 1])
        # check if the past data are constant
        # if asset i is constant set w_i = 0
        sub_X = df.iloc[t:t_s, :].copy()
        assets = sub_X.apply(process).values
        weight_vect[assets, 0] = w[t][assets]
        # Verify if sum(w) = 0
        if weight_vect.min()<0:
            weight_vect=weight_vect - weight_vect.min()

        if weight_vect.sum() == 0.:
            if previousValidWeights is None:
                weight_mat.iloc[t: t_s] = np.ones([t_s - t, N]) / N
            else :
                weight_mat.iloc[t: t_s] = previousValidWeights
        elif weight_vect.sum() != 1.:
            weight_mat.iloc[t: t_s] = np.ones([t_s - t, 1]) @ weight_vect.T / weight_vect.sum()
            previousValidWeights =  np.ones([t_s - t, 1]) @ weight_vect.T / weight_vect.sum()
        else:
            weight_mat.iloc[t: t_s] = np.ones([t_s - t, 1]) @ weight_vect.T
            previousValidWeights =  np.ones([t_s - t, 1]) @ weight_vect.T

        if abs(weight_mat.iloc[t:t_s].sum().sum()) <= 1e-6:
            print('null prediction, investigate')

    return weight_mat


