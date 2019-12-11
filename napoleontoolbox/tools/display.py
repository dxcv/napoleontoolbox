#!/usr/bin/env python3
# coding: utf-8

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from napoleontoolbox.analyzer import market

def rand_color_palette(N):
    col = []
    colors = sns.mpl_palette('Set1', 9)
    for j in range(N):
        if j == 9:
            colors = sns.mpl_palette('Set3', 12)

        elif j == 21:
            colors = sns.mpl_palette('Set2', 8)

        elif j == 29:
            colors = list(sns.crayons.keys())

        i = np.random.randint(0, high=len(colors))
        if j >= 29:
            col += [sns.crayon_palette(colors.pop(i))]

        else:
            col += [colors.pop(i)]

    return col


palette = rand_color_palette(17)


def display_results(df, *args, w_mat=None, title='', palette=None, figsize=(16, 16),strats=None):
    N = df.columns.size - len(args)
    if palette is None:
        palette = rand_color_palette(N)

    if w_mat is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)

    else:
        n = sum(w_mat.sum(axis=1) == 0.)
        f, (ax, ax2) = plt.subplots(2, 1, figsize=figsize)
        ax4 = f.add_subplot(2, 2, 4)
        ax3 = f.add_subplot(2, 2, 3, sharey=ax2)
        w_mat.iloc[n:].mean(axis=0).plot(
            kind='pie',
            ax=ax4,
            title='Mean of weights allocation',
            colors=palette
        )
        ax4.set_ylabel('weights in %', fontsize=12, y=0.5, x=-0.5, rotation=0)
        ax2.set_xticks([])
        w_mat.iloc[n:].plot(
            ax=ax3,
            kind='area',
            stacked=True,
            title='Weights allocation',
            color=palette
        )

    ma = market.MarketAnalyzer(df)
    ma.display_kpi()
    ma.plot_perf(logy=True, title=title, y=strats, ax=ax, color=palette, show=False)
    ma.plot_perf(logy=True, title=title, y=list(args), ax=ax, c='k', lw=2)
