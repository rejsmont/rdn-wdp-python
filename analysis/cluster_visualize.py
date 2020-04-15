#!/usr/bin/env python3

import argparse
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

from CellModels.IO import CellReader
from CellModels.Figures import Figure, LogScaleGenePlot


class SamplePlot(Figure):

    GX_MIN = 0
    GX_MAX = 80
    GY_MIN = -10
    GY_MAX = 45
    GY_CMAX = 25

    def __init__(self, data, sample=None, field='Venus', method='ward'):
        super().__init__(data)
        self.sample = sample
        self.field = field
        self.method = method

    def plot(self):

        self.fig = plt.figure(figsize=(10, 6))
        column = 'Cluster_' + self.method
        cells = self.data.cells

        if self.sample is not None:
            filter = cells['Sample'] == self.sample
            s_cells = cells.loc[filter].sort_values(by=[self.field])
        else:
            s_cells = cells.sort_values(by=[self.field])
        n_clusters = np.round(s_cells[column].max()).astype(int)

        ax = self.ax([0.075, 0.4, 0.4, 0.5])
        ax.scatter(s_cells['cx'], s_cells['cy'], c=s_cells[column], cmap='rainbow', s=5)
        ax.set_xlim(self.GX_MIN, self.GX_MAX)
        ax.set_ylim(self.GY_MAX, self.GY_MIN)

        ax = self.ax([0.075, 0.2, 0.4, 0.125])
        ax.set_axis_off()
        cmap = plt.cm.get_cmap('rainbow', n_clusters)
        handles = []
        for i in range(n_clusters):
            handles.append(Line2D([0], [0], marker='o', color=cmap(i), markersize=5, lw=0))
        labels = [('Cluster ' + str(i+1) + ' (' + str(s_cells.loc[s_cells[column] == i+1]['Gene'].count()) + ')') for i in range(n_clusters)]
        ax.legend(handles, labels, frameon=False, ncol=2, loc='upper center')

        ax = self.ax([0.575, 0.4, 0.4, 0.5])
        sc = ax.scatter(s_cells['cx'], s_cells['cy'], c=s_cells[self.field],
                        norm=colors.LogNorm(*LogScaleGenePlot.v_lim()), cmap='plasma', s=5)
        ax.set_xlim(self.GX_MIN, self.GX_MAX)
        ax.set_ylim(self.GY_MAX, self.GY_MIN)

        ax = self.ax([0.575, 0.25, 0.4, 0.025])
        cb = self.fig.colorbar(sc, cax=ax, orientation='horizontal', ticks=LogScaleGenePlot.v_ticks(),
                               format=LogScaleGenePlot.major_formatter_log)
        cb.set_label(label=str(self.field) + ' expression')

        super().plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize clustering result')
    parser.add_argument('input')
    parser.add_argument('outdir')
    args = parser.parse_args()

    data_file = args.input
    out_dir = args.outdir

    data = CellReader.read(data_file)
    samples = data.cells['Sample'].unique()
    figure = SamplePlot(data)
    out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(data_file))[0] + '.pdf')
    figure.save(out_file)
    for sample in samples:
        figure = SamplePlot(data, sample)
        out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(data_file))[0] + '_' + sample + '_' + '.pdf')
        figure.save(out_file)
        plt.close(figure.fig)
