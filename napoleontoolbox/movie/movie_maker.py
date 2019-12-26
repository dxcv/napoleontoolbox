#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.animation as manimation

def createCorrelationBarMovie(rolled_sharpe, saving_path):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    ax = fig.gca()

    print('saving file')
    with writer.saving(fig, saving_path, 100):
        for row in rolled_sharpe.iterrows():
            ax.clear()
            plt.ylim(-1, 1)
            histo_values = row[1].to_frame()
            histo_values.columns = ['CORR']
            histo_values['STRAT'] = histo_values.index
            histo_values.plot.bar(x='STRAT', y='CORR', rot=90, ax = ax, title = str(row[0]), legend = None)
            writer.grab_frame()
