#!/usr/bin/env python3

import argparse
import logging
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from scipy.stats import linregress, zscore
from statsmodels.stats.weightstats import CompareMeans
from images import Qimage, Thumbnails
from data import DiscData, OriginalData
from stats import CellStats
from chip import ChIP
from clustering import Clustering, ClusteredData
from figures_ng import Figure, MultiCellPlot, LogScaleGenePlot, SmoothProfilePlot, APProfilePlot, LabeledPlot,\
                       DVProfilePlot, DiscThumb, LogScaleExtPlot, MFProfilePlot

e = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/Figures'
d = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/samples'
f = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/processing/samples_complete.csv'
cl = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/processing/clustering/bigc100k6n20r1000_metadata.yml'
ch = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/contrib/eye-ChIP/chip_peaks_eye_genes.tsv'


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


class Figure_1(Figure):
    """
    # Figure 1 - Image segmentation and quantification
    *a*
    """

    SAMPLE = 'K21OU5'
    SLICE = 40
    BONDS = [(0, 32), (0, 4000), (0, 16)]

    def __init__(self, stats, image, thumbs):
        super().__init__(None)
        self.stats = stats
        self.h5 = image
        self.thumbs = thumbs

    def plot(self):
        self.fig = plt.figure(figsize=mm2inch(183, 145))

        ax = self.ax([0.05, 0.65, 0.425, 0.3])
        self.h5.plot_img(self.SLICE, ax=ax, bonds=self.BONDS)
        self.h5.scalebar(ax, linewidth=3)
        self.h5.label(ax, 'a', fontsize=12)
        ax = self.ax([0.525, 0.65, 0.425, 0.3])
        self.h5.plot_simg(self.SLICE, ax=ax, bonds=self.BONDS, reference='auto')
        self.h5.scalebar(ax, linewidth=3)
        self.h5.label(ax, 'b', fontsize=12)

        ax = self.ax([0.1, 0.40, 0.3, 0.2])
        self.thumbs.plot_normalized(ax)
        self.thumbs.label(ax, 'c', fontsize=12)
        ax = self.ax([0.1, 0.05, 0.3, 0.3])
        self.thumbs.plot_aligned(ax)
        self.thumbs.label(ax, 'd', fontsize=12)

        ax = self.ax([0.5125, 0.425, 0.1625, 0.175])
        self.stats.size_hist(ax)
        self.stats.label(ax, 'e', fontsize=12)

        ax = self.ax([0.7875, 0.425, 0.1625, 0.175])
        self.stats.sample_size_hist(ax)
        self.stats.label(ax, 'f', fontsize=12)

        ax = self.ax([0.5125, 0.1125, 0.1625, 0.175])
        self.fig.add_axes(ax)
        self.stats.sample_count_hist(ax)
        self.stats.label(ax, 'g', fontsize=12)

        ax = plt.Axes(self.fig, [0.7875, 0.1125, 0.1625, 0.175])
        self.fig.add_axes(ax)
        self.stats.mf_ato_hist(ax)
        self.stats.label(ax, 'h', fontsize=12)


class Figure_2(Figure):

    class GeneProfilePlot(LogScaleGenePlot, SmoothProfilePlot, APProfilePlot):
        def format_axis(self):
            super().format_axis()
            self.ax.set_ylabel(r'Mean expression')
            self.ax.set_xlabel(r'A-P position')

        @staticmethod
        def v_lim():
            return [0.2, 5]

        @staticmethod
        def x_lim():
            return [-10, 40]

        @staticmethod
        def v_ticks(): return [0.2, 0.5, 1, 2, 5]

    class DVGeneProfilePlot(LogScaleGenePlot, SmoothProfilePlot, DVProfilePlot):
        def format_axis(self):
            super().format_axis()
            self.ax.set_ylabel(r'Mean expression')
            self.ax.set_xlabel(r'D-V position')

        @staticmethod
        def v_lim():
            return [0.2, 5]

        @staticmethod
        def v_ticks(): return [0.2, 0.5, 1, 2, 5]

    class GeneDiscThumb(LogScaleGenePlot, DiscThumb):
        @staticmethod
        def v_ticks(): return [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]

        @staticmethod
        def v_minor_ticks(): return False

    class ExtDiscThumb(LogScaleExtPlot, DiscThumb):
        @staticmethod
        def v_ticks(): return [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]

        @staticmethod
        def v_minor_ticks(): return False

    def __init__(self, data):
        super().__init__(data)

    def plot(self):
        self.fig = plt.figure(figsize=mm2inch(136, 170))
        matrices = self.data.matrices()

        thumbs = []
        for row, gene in enumerate(['AtoClean', 'ato']):
            matrix = matrices.loc[gene]
            thumbs = [self.GeneDiscThumb(self.fig, matrix['mean'], r'Mean expression'),
                      self.ExtDiscThumb(self.fig, matrix['ext'], r'Max eccentricity')]

            ax = self.ax([0.075, 1 - ((row + 1) * 0.22 + row * 0.03), 0.4, 0.17])
            thumbs[0].plot(ax)
            letter = chr(ord('a') + 2 * row)
            ax.text(-0.15, 1.15, letter, color='black', transform=ax.transAxes, va='top', fontsize=12)
            ax = self.ax([0.575, 1 - ((row + 1) * 0.22 + row * 0.03), 0.4, 0.17])
            thumbs[1].plot(ax)
            letter = chr(ord('b') + 2 * row)
            ax.text(-0.15, 1.15, letter, color='black', transform=ax.transAxes, va='top', fontsize=12)

        for col, thumb in enumerate(thumbs):
            ax = self.ax([col * 0.5 + 0.075, 0.45, 0.4, 0.02])
            thumb.legend(ax, fontsize=8)

        profiles = self.data.profiles()
        plots = []

        styles = {
            'Reporter mean': {},
            'Protein mean': {'color': '#2ca02c'},
        }

        ato_ap = pd.DataFrame()
        ato_ap['Reporter mean'] = profiles.loc['ato']['mean']
        ato_ap['Protein mean'] = profiles.loc['AtoClean']['mean']
        plots.append(self.GeneProfilePlot(self.fig, ato_ap, styles=styles))

        dv_profiles = self.data.dv_profiles()
        ato_dv = pd.DataFrame()
        ato_dv['Reporter mean'] = dv_profiles.loc['ato']['mean']
        ato_dv['Protein mean'] = dv_profiles.loc['AtoClean']['mean']
        plots.append(self.DVGeneProfilePlot(self.fig, ato_dv, styles=styles))

        def build_handles_labels(h, l, legend):
            new_h, new_l = legend
            for i, label in enumerate(new_l):
                if label not in l:
                    h.append(new_h[i])
                    l.append(new_l[i])

        handles = []
        labels = []

        ax = self.ax([0.125, 0.125, 0.275, 0.2])
        plots[0].plot(ax, firstrow=False, lastrow=True)
        ax.text(-0.385, 1.2, 'e', color='black', transform=ax.transAxes, va='top', fontsize=12)
        build_handles_labels(handles, labels, plots[0].ax.get_legend_handles_labels())

        ax = self.ax([0.575, 0.125, 0.4, 0.2])
        plots[1].plot(ax, firstrow=False, lastrow=True)
        ax.text(-0.25, 1.2, 'f', color='black', transform=ax.transAxes, va='top', fontsize=12)
        build_handles_labels(handles, labels, plots[1].ax.get_legend_handles_labels())

        ax = self.ax([0.05, 0, 0.9, 0.075])
        ax.set_axis_off()
        ax.legend(handles, labels, ncol=2, loc='center', frameon=False)
        super().plot()


class Figure_3(Figure):

    GX_MIN = 0
    GX_MAX = 80
    GY_MIN = -10
    GY_MAX = 45
    GY_CMAX = 25

    #SAMPLE = 'K21OU5'
    SAMPLE = 'O4UW6B'

    def plot(self):

        def bars(v):
            q25, q50, q75 = np.percentile(v, [25, 50, 75])

            upper_adjacent_value = q75 + (q75 - q25) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q75, v.max())

            lower_adjacent_value = q25 - (q75 - q25) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, v.min(), q25)

            return v, q50, q25, q75, lower_adjacent_value, upper_adjacent_value

        self.fig = plt.figure(figsize=mm2inch(136, 136))
        column = 'Cluster_' + self.data.clustering.method
        cells = self.data.cells().loc[self.data.acceptable_mask()]
        g_centroids = cells.groupby(column, as_index=False)[Clustering.C_FEATURES].mean()

        filter = cells['Sample'] == self.SAMPLE
        s_cells = cells.loc[filter].sort_values(by=['mCherry'])

        gene = s_cells['Gene'].unique()[0]
        s_centroids = s_cells.groupby(column, as_index=False)[Clustering.C_FEATURES].mean()

        ax = self.ax([0.075, 0.725, 0.4, 0.2125])
        ax.scatter(s_cells['cx'], s_cells['cy'], c=s_cells[column], cmap='rainbow', s=5)
        ax.set_xlim(self.GX_MIN, self.GX_MAX)
        ax.set_ylim(self.GY_MAX, self.GY_MIN)
        ax.text(-0.15, 1.20, 'a', color='black', transform=ax.transAxes, va='top', fontsize=12)

        ax = self.ax([0.075, 0.5375, 0.4, 0.125])
        ax.set_axis_off()
        cmap = plt.cm.get_cmap('rainbow', 6)
        handles = [Line2D([0], [0], marker='o', color=cmap(0), markersize=5, lw=0),
                   Line2D([0], [0], marker='o', color=cmap(1), markersize=5, lw=0),
                   Line2D([0], [0], marker='o', color=cmap(5), markersize=5, lw=0),
                   Line2D([0], [0], marker='o', color=cmap(2), markersize=5, lw=0),
                   Line2D([0], [0], marker='o', color=cmap(4), markersize=5, lw=0),
                   Line2D([0], [0], marker='o', color=cmap(3), markersize=5, lw=0)]
        labels = ['R8', 'MF High', 'MF Medium', 'MF Low', 'Pre-MF', 'Post-MF']
        ax.legend(handles, labels, frameon=False, ncol=2, loc='upper center')

        ax = self.ax([0.575, 0.725, 0.4, 0.2125])
        sc = ax.scatter(s_cells['cx'], s_cells['cy'], c=s_cells['mCherry'], norm=colors.LogNorm(*LogScaleGenePlot.v_lim()),
                        cmap='plasma', s=5)
        ax.set_xlim(self.GX_MIN, self.GX_MAX)
        ax.set_ylim(self.GY_MAX, self.GY_MIN)
        ax.text(-0.15, 1.20, 'b', color='black', transform=ax.transAxes, va='top', fontsize=12)

        ax = self.ax([0.575, 0.625, 0.4, 0.025])
        cb = self.fig.colorbar(sc, cax=ax, orientation='horizontal', ticks=LogScaleGenePlot.v_ticks(),
                               format=LogScaleGenePlot.major_formatter_log)
        cb.set_label(label='Ato expression')

        s_cells = s_cells.sort_values(by=['ext_mCherry'])
        ax = self.ax([0.1, 0.235, 0.25, 0.25])
        ax.scatter(s_cells['cy'], s_cells['mCherry'], c=s_cells[column], s=40, cmap='rainbow')
        ax.scatter(s_cells['cy'], s_cells['mCherry'], c=s_cells['ext_mCherry'], s=5)
        ax.scatter(g_centroids['cy'], g_centroids['mCherry'], c=g_centroids[column], s=20, marker='D', cmap='rainbow',
                   linewidth=1, edgecolor='black')
        ax.scatter(s_centroids['cy'], s_centroids['mCherry'], c=s_centroids[column], s=20, marker='s', cmap='rainbow',
                   linewidth=1, edgecolor='black')
        ax.set_xlim(self.GY_MIN, self.GY_CMAX)
        ax.set_xlabel('A-P position')
        ax.set_yscale('log')
        ax.set_ylim(0.1, 5)
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
        ax.yaxis.set_major_formatter(LogScaleGenePlot.major_formatter_log)
        ax.set_ylabel('Ato expression')
        ax.text(-0.35, 1.175, 'c', color='black', transform=ax.transAxes, va='top', fontsize=12)

        s_centroids = cells.groupby(['Sample', column], as_index=False)[Clustering.C_FEATURES].mean().sort_values(by=['ext_mCherry'])
        g_centroids = cells.groupby([column], as_index=False)[Clustering.C_FEATURES].mean()

        ax = self.ax([0.375, 0.235, 0.25, 0.25])
        ax.scatter(s_centroids['cy'], s_centroids['mCherry'], c=s_centroids[column], s=40, cmap='rainbow')
        sc = ax.scatter(s_centroids['cy'], s_centroids['mCherry'], c=s_centroids['ext_mCherry'], s=5)
        # ax.scatter(g_centroids['cy'], g_centroids['mCherry'], c=g_centroids[column], s=20, marker='D', cmap='rainbow',
        #            linewidth=1, edgecolor='black')
        ax.set_xlim(self.GY_MIN, self.GY_CMAX)
        ax.set_xlabel('A-P position')
        ax.set_ylim(0.1, 5)
        ax.set_yscale('log')
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
        ax.set_yticklabels([])
        ax.text(-0.1, 1.175, 'd', color='black', transform=ax.transAxes, va='top', fontsize=12)

        ax = self.ax([0.1, 0.1, 0.525, 0.025])
        cb = self.fig.colorbar(sc, cax=ax, orientation='horizontal')
        cb.set_label(label='Ato eccentricity')

        # Plot fraction of cells that belong to a cluster

        counts = cells.groupby(['Sample', column])['mCherry'].count()
        totals = cells.groupby(['Sample'])['mCherry'].count()
        order = [1, 2, 6, 3, 5, 4]
        inds = [1, 2]
        data = [(counts.xs(cluster, level=1) / totals * 100) for cluster in order]
        values = [bars(v) for v in data]
        d, m, q25, q75, w1, w2 = zip(*values)

        ax = self.ax([0.70, 0.4, 0.275, 0.085])
        parts = ax.violinplot(data[4:], showmeans=False, showmedians=False, showextrema=False, vert=False)
        ax.scatter(m[4:], inds, marker='o', color='white', s=10, zorder=3)
        ax.hlines(inds, q25[4:], q75[4:], color='k', linestyle='-', lw=2)
        ax.hlines(inds, w1[4:], w2[4:], color='k', linestyle='-', lw=0.5)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(cmap(order[i + 4] - 1))
            pc.set_edgecolor('black')
            pc.set_alpha(1)
            pc.set_linewidth(0.5)
        ax.add_line(Line2D([0, 0], [0.65, -0.55], clip_on=False, linewidth=0.5, color='k', alpha=0.5))
        ax.add_line(Line2D([15, 100], [0.65, -0.55], clip_on=False, linewidth=0.5, color='k', alpha=0.5))
        ax.yaxis.set_visible(False)
        ax.set_xlim(0, 100)
        ax.text(-0.1, 1.525, 'e', color='black', transform=ax.transAxes, va='top', fontsize=12)

        ax = self.ax([0.70, 0.25, 0.275, 0.085])
        parts = ax.violinplot(data[2:4], showmeans=False, showmedians=False, showextrema=False, vert=False)
        ax.scatter(m[2:4], inds, marker='o', color='white', s=10, zorder=3)
        ax.hlines(inds, q25[2:4], q75[2:4], color='k', linestyle='-', lw=2)
        ax.hlines(inds, w1[2:4], w2[2:4], color='k', linestyle='-', lw=0.5)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(cmap(order[i + 2] - 1))
            pc.set_edgecolor('black')
            pc.set_alpha(1)
            pc.set_linewidth(0.5)
        ax.yaxis.set_visible(False)
        ax.add_line(Line2D([0, 0], [0.65, -0.55], clip_on=False, linewidth=0.5, color='k', alpha=0.5))
        ax.add_line(Line2D([5, 15], [0.65, -0.55], clip_on=False, linewidth=0.5, color='k', alpha=0.5))
        ax.set_xlim(0, 15)

        ax = self.ax([0.70, 0.1, 0.275, 0.085])
        parts = ax.violinplot(data[:2], showmeans=False, showmedians=False, showextrema=False, vert=False)
        ax.scatter(m[:2], inds, marker='o', color='white', s=10, zorder=3)
        ax.hlines(inds, q25[:2], q75[:2], color='k', linestyle='-', lw=2)
        ax.hlines(inds, w1[:2], w2[:2], color='k', linestyle='-', lw=0.5)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(cmap(order[i] - 1))
            pc.set_edgecolor('black')
            pc.set_alpha(1)
            pc.set_linewidth(0.5)
        ax.yaxis.set_visible(False)
        ax.set_xlim(0, 5)
        ax.set_xlabel('Fraction of cells [%]')

        super().plot()


class Figure_4(Figure):

    class GeneProfilePlot(MultiCellPlot, LogScaleGenePlot, MFProfilePlot, SmoothProfilePlot):
        @staticmethod
        def v_lim():
            return [0.1, 10]

        @staticmethod
        def v_ticks():
            return [0.1, 0.2, 0.5, 1, 2, 5, 10]

    def __init__(self, cells, chip):
        super().__init__(cells)
        self.chip = chip
        self.ratios = None

    def compute(self):
        cells = self.data.cells()[self.data.furrow_mask() & ~self.data.bad_gene_mask()]

        high_ato = self.data.AGGREGATE_NAMES.inverse['high-ato'][0]
        no_ato = self.data.AGGREGATE_NAMES.inverse['no-ato'][0]
        fields = ['Venus', 'Gene']

        cells_high = cells['Cluster_agg'] == high_ato
        cells_no = cells['Cluster_agg'] == no_ato

        group_cells_high = cells.loc[cells_high, fields].groupby('Gene')
        group_cells_no = cells.loc[cells_no, fields].groupby('Gene')

        t = {}
        for group in group_cells_high.groups.keys():
            cm = CompareMeans.from_data(group_cells_high.get_group(group)['Venus'],
                                        group_cells_no.get_group(group)['Venus'])
            s, p = cm.ztest_ind(usevar='unequal')

            # Apply Bonferroni correction to t-test thresholds
            if p <= 0.0001 / len(group_cells_high.groups):
                a = ''
            elif p <= 0.001 / len(group_cells_high.groups):
                a = ''
            elif p <= 0.01 / len(group_cells_high.groups):
                a = ''
            elif p <= 0.05 / len(group_cells_high.groups):
                a = ''
            else:
                a = 'ns'

            t[group] = (s, p, a)

        t_stats = pd.DataFrame(t).T
        t_stats.columns = ['s', 'p', 'a']

        cell_ratio = group_cells_high['Venus'].mean() / group_cells_no['Venus'].mean()
        cell_ratio = cell_ratio.rename('Expression ratio (mean)')

        cell_ratio_sem = np.sqrt(np.square(group_cells_high['Venus'].sem() / group_cells_high['Venus'].mean()) +
                                 np.square(group_cells_no['Venus'].sem() / group_cells_no['Venus'].mean())) * cell_ratio
        cell_ratio_sem = cell_ratio_sem.rename('Expression ratio (SEM)')

        peaks = self.chip.peaks().groupby('Gene').agg(['sum', 'count'])
        ato_p_area = peaks.loc[:, ('p_area', 'sum')].rename('Peak area')
        ato_p_num = peaks.loc[:, ('p_area', 'count')].rename('Peak count')

        self.ratios = pd.DataFrame([cell_ratio, cell_ratio_sem, ato_p_area, ato_p_num]).transpose()
        self.ratios = self.ratios.join(t_stats)

    def plot(self):
        if self.ratios is None:
            self.compute()

        self.fig = plt.figure(figsize=mm2inch(183, 200))

        column = 'Cluster_' + self.data.clustering.method
        cells = self.data.cells().loc[
            self.data.acceptable_mask() &
            self.data.furrow_mask() &
            ~self.data.bad_gene_mask()
            ]
        profiles = self.data.profiles()
        plist = self.data.CLUSTER_NAMES

        order = [5, 3, 6, 2, 1, 4]
        cmap = plt.cm.get_cmap('rainbow', 6)
        styles = {
            'Target mean R8': {'color': cmap(0)},
            'Target mean MF-high': {'color': cmap(1)},
            'Target mean MF': {'color': cmap(2)},
            'Target mean post-MF': {'color': cmap(3)},
            'Target mean pre-MF': {'color': cmap(4)},
            'Target mean MF-ato': {'color': cmap(5)}
        }

        template = pd.DataFrame(index=pd.Index(range(int(DiscData.FURROW_MIN), int(DiscData.FURROW_MAX) + 1)))
        for c in plist.keys():
            template['Target mean ' + plist[c]] = np.nan

        def trim(v, min=None, max=None, zc=None):
            s = v.copy()
            if min is not None:
                s.loc[s < min] = min
            if max is not None:
                s.loc[s > max] = max
            if zc is not None:
                z = zscore(s)
                s.loc[z > zc] = s[z <= zc].max()
                s.loc[z < -zc] = s[z >= -zc].min()
            return s.values

        def bars(v):
            d = trim(v, 0.1, 20, 3)
            q25, q50, q75 = np.percentile(v, [25, 50, 75])

            upper_adjacent_value = q75 + (q75 - q25) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q75, d.max())

            lower_adjacent_value = q25 - (q75 - q25) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, d.min(), q25)

            return d, q50, q25, q75, lower_adjacent_value, upper_adjacent_value

        def gene_violin(ax, data, gene, start=1):

            end = start + len(data)
            inds = np.arange(start, end)

            values = [bars(v) for v in data]
            d, m, q25, q75, w1, w2 = zip(*values)
            parts = ax.violinplot(d, inds, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(cmap(order[i] - 1))
                pc.set_edgecolor('black')
                pc.set_alpha(1)
                pc.set_linewidth(0.25)

            ax.scatter(inds, m, marker='o', color='white', s=10, zorder=3)
            ax.vlines(inds, q25, q75, color='k', linestyle='-', lw=2)
            ax.vlines(inds, w1, w2, color='k', linestyle='-', lw=0.5)

            ax.set_xticks([])
            ax.set_yscale('log')
            ax.set_ylim(*self.GeneProfilePlot.v_lim())
            ax.set_yticks(self.GeneProfilePlot.v_ticks())
            labels = [LogScaleGenePlot.major_formatter_log(s, None) for s in self.GeneProfilePlot.v_ticks()]
            labels[0] = '<.1'
            labels[-1] = '10'
            ax.set_yticklabels(labels)
            ax.set_ylabel('Expression level')
            ax.text(0.5, -0.25, gene, horizontalalignment='center', verticalalignment='center',
                    fontsize=12, transform=ax.transAxes)

        def gene_profile(ax, data):
            plot = self.GeneProfilePlot(self.fig, data, styles)
            plot.plot(ax, text=None, label='left',
                      firstcol=False, firstrow=False,
                      lastcol=False, lastrow=True)
            ax.set_xlabel('A-P position')
            ax.set_xticks([-8, -4, 0, 4, 8])

        def gene_data(gene):
            if gene == 'Ato':
                dists = [cells.loc[cells[column] == cluster, 'mCherry'] for cluster in order]
            else:
                dists = [cells.loc[(cells[column] == cluster) & (cells['Gene'] == gene), 'Venus'] for cluster in order]
            profile = profiles.loc[gene]
            gene_profiles = template.copy()
            for c in plist.keys():
                gene_profiles['Target mean ' + plist[c]] = profile.loc[c]['mean']

            return dists, gene_profiles

        w = 0.175
        h = 0.12
        wr = 1.125
        hr = 1.5

        row = 0
        dists, profs = gene_data('ato')
        ax = self.ax([0.1, 0.975 - h - row * h * hr, w, h])
        ax.text(-0.525, 1.15, 'a', color='black', transform=ax.transAxes, va='top', fontsize=12)
        gene_violin(ax, dists, 'ato')
        ax = self.ax([0.1 + w * wr, 0.975 - h - row * h * hr, w, h])
        gene_profile(ax, profs)

        dists, profs = gene_data('sca')
        ax = self.ax([0.6, 0.975 - h - row * h * hr, w, h])
        gene_violin(ax, dists, 'sca')
        ax = self.ax([0.6 + w * wr, 0.975 - h - row * h * hr, w, h])
        gene_profile(ax, profs)

        row = 1
        dists, profs = gene_data('dap')
        ax = self.ax([0.1, 0.975 - h - row * h * hr, w, h])
        gene_violin(ax, dists, 'dap')
        ax = self.ax([0.1 + w * wr, 0.975 - h - row * h * hr, w, h])
        gene_profile(ax, profs)

        dists, profs = gene_data('nSyb')
        ax = self.ax([0.6, 0.975 - h - row * h * hr, w, h])
        gene_violin(ax, dists, 'nSyb')
        ax = self.ax([0.6 + w * wr, 0.975 - h - row * h * hr, w, h])
        gene_profile(ax, profs)


        # (Ato ChIP peak area) vs (Target expression fold change)
        ax = self.ax([0.1, 0.40, 0.375, 0.2])
        ratios = self.ratios.dropna().sort_values('Peak area')
        ratiosA = self.ratios.loc[self.ratios.index != 'nvy'].dropna().sort_values('Peak area')
        ratiosN = self.ratios.loc[['nvy']].dropna().sort_values('Peak area')
        x = ratiosA['Peak area']
        y = ratiosA['Expression ratio (mean)']
        ax.scatter(x, y)
        gradient, intercept, r_value, p_value, std_err = linregress(x, y)
        ry = gradient * x + intercept
        ax.plot(x, ry)
        ax.text(0.075, 0.9, 'ρ=' + '{:.2f}'.format(np.corrcoef(x, y)[0, 1]) + ', p=' + '{:.2f}'.format(p_value),
                fontsize=8, transform=ax.transAxes, color='C0')
        x = ratiosN['Peak area']
        y = ratiosN['Expression ratio (mean)']
        ax.scatter(x, y)
        x = ratios['Peak area']
        y = ratios['Expression ratio (mean)']
        gradient, intercept, r_value, p_value, std_err = linregress(x, y)
        ry = gradient * x + intercept
        ax.plot(x, ry)
        ax.text(0.075, 0.8, 'ρ=' + '{:.2f}'.format(np.corrcoef(x, y)[0, 1]) + ', p=' + '{:.2f}'.format(p_value),
                fontsize=8, transform=ax.transAxes, color='C1')
        ax.set_ylim(0.1, 3.5)
        ax.set_xlabel('Ato ChIP peak area')
        ax.set_ylabel('Expression fold change')
        ax.text(-0.25, 1.1, 'b', color='black', transform=ax.transAxes, va='top', fontsize=12)

        # (Ato ChIP peak area)
        ax = self.ax([0.6, 0.40, 0.375, 0.2])
        ratios = self.ratios.dropna().sort_values('Peak area')
        y = ratios['Peak area']
        x = np.arange(y.count())
        labels_chip = list(y.index.values)
        barlist = ax.bar(x, y)
        barlist[10].set_color('C1')
        ax.set_ylabel('Ato ChIP peak area')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_chip, rotation=45, ha='right')
        ax.text(-0.25, 1.1, 'c', color='black', transform=ax.transAxes, va='top', fontsize=12)

        # (Target expression fold change)
        ax = self.ax([0.1, 0.12, 0.875, 0.2])
        ratios = self.ratios.sort_values('Expression ratio (mean)')
        y = ratios['Expression ratio (mean)'].dropna()
        e = ratios['Expression ratio (SEM)'].dropna()
        x = np.arange(y.count())
        a = ratios['a'].dropna()
        labels = list(y.index.values)
        ax.axhline(y=1, color='C3')
        barlist = ax.bar(x, y)
        ax.errorbar(x, y, fmt='none', yerr=e, ecolor='black', capsize=5, elinewidth=0.5, capthick=0.5)
        for i, v in enumerate(x):
            ax.text(x[i], y[i] + e[i] + 0.025, a[i], ha='center', fontsize=8)
        for i, bar in enumerate(barlist):
            if labels[i] not in labels_chip:
                bar.set_color('C2')
            elif labels[i] == 'nvy':
                bar.set_color('C1')
        ax.set_ylim(0.1, 3.5)
        ax.set_ylabel('Expression fold change')
        ax.set_xticks([labels.index(i) for i in labels_chip])
        ax.set_xticklabels(labels_chip, {'weight': 'bold'}, rotation=45, ha='right')
        ax.set_xticks([labels.index(l) for l in labels if l not in labels_chip], minor=True)
        ax.set_xticklabels([l for l in labels if l not in labels_chip], rotation=45, ha='right', minor=True)
        ax.text(-0.107, 1.1, 'd', color='black', transform=ax.transAxes, va='top', fontsize=12)


if __name__ == "__main__":
    plt.rc('font', size=8)
    o_data = OriginalData(f)
    data = DiscData(o_data.cells)

    h5 = Qimage(d, Figure_1.SAMPLE)
    thumbs = Thumbnails(d + '/thumbs', Figure_1.SAMPLE)
    stats = CellStats(o_data)
    fig_1 = Figure_1(stats, h5, thumbs)
    fig_1.show()
    fig_1.save(e + '/fig_1.pdf')
    fig_2 = Figure_2(data)
    fig_2.show()
    fig_2.save(e + '/fig_2.pdf')

    clustering = Clustering(cl, disc_data=data)
    clustered = ClusteredData(clustering)

    fig_3 = Figure_3(clustered)
    fig_3.show()
    fig_3.save(e + '/fig_3.pdf')

    chip = ChIP(ch, data.genes())
    fig_4 = Figure_4(clustered, chip)
    fig_4.show()
    fig_4.save(e + '/fig_4.pdf')
