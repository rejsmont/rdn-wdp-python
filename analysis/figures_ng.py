#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import savgol_filter
from data import DiscData
from clustering import Clustering

GX_MIN = 0
GX_MAX = 80
GY_MIN = -10
GY_MAX = 45
MF_MIN = -10
MF_MAX = 10


class Figure:

    def __init__(self, data):
        self.data = data
        self.fig = None
        self.gs = None
        self.plotted = False

    def show(self):
        if not self.plotted:
            self.plot()
        self.fig.show()

    def save(self, path):
        if not self.plotted:
            self.plot()
        self.fig.savefig(path)

    def plot(self, outdir):
        self.plotted = True
        pass


class Plot:

    def __init__(self, fig, data):
        self.fig = fig
        self.data = data
        self.ax = None

    def plot(self, position, *args, **kwargs):
        self.ax = self.fig.add_subplot(position)

    def legend(self, position, *args, **kwargs):
        pass

    @staticmethod
    def x_lim(): return False

    @staticmethod
    def y_lim(): return False

    @staticmethod
    def v_lim(): return False

    @staticmethod
    def x_scale(): return False

    @staticmethod
    def y_scale(): return False

    @staticmethod
    def v_scale(): return False

    @staticmethod
    def x_ticks(): return False

    @staticmethod
    def y_ticks(): return False

    @staticmethod
    def v_ticks(): return False

    @staticmethod
    def v_minor_ticks(): return False

    def v_axis_formatter(self): return False


class LogScaleGenePlot:

    @staticmethod
    def cmap(): return 'plasma'

    @staticmethod
    def v_lim(): return [0.1, 30]

    @staticmethod
    def v_scale(): return 'log'

    @staticmethod
    def v_ticks(): return [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]

    @staticmethod
    def v_minor_ticks(): return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]

    @ticker.FuncFormatter
    def major_formatter_log(x, pos):
        return "%g" % (round(x * 10) / 10)

    def v_axis_formatter(self): return self.major_formatter_log


class LogScaleExtPlot(LogScaleGenePlot):

    @staticmethod
    def cmap(): return 'viridis'

    @staticmethod
    def v_lim(): return [0.1, 20]

    @staticmethod
    def v_minor_ticks(): return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]


class ProfilePlot(Plot):
    """
    Plot gene expression profile
    """

    def __init__(self, fig, data, styles=None):
        super().__init__(fig, data)
        self._styles = styles if styles is not None else False

    def plot(self, position, *args, **kwargs):
        super().plot(position, *args, **kwargs)
        self.plot_profiles()
        self.format_axis()

    def legend(self, position, ncol=1, loc='upper center', *args, **kwargs):
        ax = self.fig.add_subplot(position)
        ax.set_axis_off()
        handles, labels = self.ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=ncol, loc=loc, frameon=False, fontsize=18)

    def plot_profile(self, profile, style=None):
        data = self.data[profile]
        x = data.index
        y = self.preprocessor(data.values)
        style = style if style is not None else {}
        self.ax.plot(x, y, label=profile, **style)

    def plot_profiles(self):
        styles = self.styles()
        for profile in self.data:
            style = styles[profile] if styles and styles[profile] else None
            self.plot_profile(profile, style=style)
        return self.ax.get_legend_handles_labels()

    def format_axis(self):
        if self.x_lim():
            self.ax.set_xlim(self.x_lim())
        if self.v_lim():
            self.ax.set_ylim(self.v_lim())
        if self.v_scale():
            self.ax.set_yscale(self.v_scale())
        if self.v_ticks():
            self.ax.set_yticks(self.v_ticks())
        if self.v_axis_formatter():
            self.ax.yaxis.set_major_formatter(self.v_axis_formatter())

    @staticmethod
    def preprocessor(x):
        return x

    def styles(self):
        return self._styles


class APProfilePlot(ProfilePlot):
    @staticmethod
    def x_lim():
        return [GY_MIN, GY_MAX]


class MFProfilePlot(ProfilePlot):
    @staticmethod
    def x_lim():
        return [MF_MIN, MF_MAX]


class DVProfilePlot(ProfilePlot):
    @staticmethod
    def x_lim():
        return [GX_MIN, GX_MAX]


class SmoothProfilePlot(ProfilePlot):
    @staticmethod
    def preprocessor(x):
        return savgol_filter(x, 9, 3, mode='nearest')


class DiscThumb(Plot):
    """
    Plot a disc thumbnail
    """
    def __init__(self, fig, data, title):
        super().__init__(fig, data)
        self.img = None
        self.index = self.data.index.to_frame()
        self.extent = [self.index['cx'].min(), self.index['cx'].max(),
                       self.index['cy'].max(), self.index['cy'].min()]
        self.title = title

    def plot(self, position, *args, **kwargs):
        super().plot(position, color='white', *args, **kwargs)
        matrix = self.disc_matrix()
        self.img = self.ax.imshow(matrix, extent=self.extent, norm=self.norm(), cmap=self.cmap(), aspect='auto')
        self.ax.set_facecolor('black')
        self.ax.set_xlim(self.x_lim())
        self.ax.set_ylim(self.y_lim())

    def legend(self, position, *args, **kwargs):
        ax = self.fig.add_subplot(position)
        cb = self.fig.colorbar(self.img, cax=ax, orientation='horizontal', ticks=self.v_ticks(),
                               format=self.v_axis_formatter())
        cb.set_label(label=self.title, fontsize=18)
        if self.v_scale() == 'log' and self.v_minor_ticks():
            ticks = self.img.norm(self.v_minor_ticks())
            cb.ax.xaxis.set_ticks(ticks, minor=True)

    def disc_matrix(self):
        x = self.index['cx']
        y = self.index['cy'] - self.extent[3]
        v = self.data
        matrix = np.full([self.extent[2] - self.extent[3] + 1, self.extent[1] - self.extent[0] + 1], np.NaN)
        matrix[y, x] = v
        raw = matrix.copy()
        nan = np.argwhere(np.isnan(matrix))
        x_max, y_max = matrix.shape
        for x, y in nan:
            xs = max([0, x-1])
            xe = min([x+1, x_max-1])
            ys = max([0, y - 1])
            ye = min([y + 1, y_max-1])
            kernel = raw[xs:xe, ys:ye]
            kernel = kernel[~np.isnan(kernel)]
            if kernel.size > 0:
                matrix[x, y] = np.mean(kernel)
            else:
                matrix[x, y] = 0
        return matrix

    def norm(self):
        v_scale = self.v_scale()
        v_lim = self.v_lim()
        if v_lim is not False:
            v_min, v_max = v_lim
        else:
            v_min = None
            v_max = None
        norm = None
        if v_scale is False or v_scale == 'linear':
            norm = colors.Normalize(vmin=v_min, vmax=v_max)
        elif v_scale == 'log':
            norm = colors.LogNorm(vmin=v_min, vmax=v_max)
        else:
            raise NotImplementedError("Override `norm` method to implement %s v_scale", v_scale)
        return norm

    @staticmethod
    def x_lim(): return [GX_MIN, GX_MAX]

    @staticmethod
    def y_lim(): return [GY_MAX, GY_MIN]

    @staticmethod
    def cmap(): return None


class LabeledPlot(Plot):

    def plot(self, position, text=None, color='black', *args, **kwargs):
        super().plot(position, *args, **kwargs)
        self.ax.text(0.025, 0.95, text, horizontalalignment='left', verticalalignment='top', fontsize=24,
                     color=color, transform=self.ax.transAxes)


class MultiCellPlot(Plot):

    def plot(self, position, firstrow=False, firstcol=True, lastrow=False, lastcol=False, controw=False,
             label='left', *args, **kwargs):
        super().plot(position, *args, **kwargs)
        self.ax.tick_params(bottom=(lastrow or controw),
                            top=((firstrow and not lastrow) or controw),
                            labelbottom=lastrow,
                            labeltop=(firstrow and not lastrow),
                            left=(label == 'left'),
                            right=(label == 'right'),
                            labelleft=(firstcol and (label == 'left')),
                            labelright=(lastcol and (label == 'right')))
        if self.v_scale() == 'log':
            self.ax.tick_params(axis='y', which='minor', left=(label == 'left'), right=(label == 'right'),
                                labelleft=False, labelright=False)


class Figure_3d51(Figure):

    class GeneProfilePlot(MultiCellPlot, LogScaleGenePlot, SmoothProfilePlot, APProfilePlot, LabeledPlot):
        pass

    class DVGeneProfilePlot(MultiCellPlot, LogScaleGenePlot, SmoothProfilePlot, DVProfilePlot, LabeledPlot):
        pass

    class GeneDiscThumb(MultiCellPlot, LogScaleGenePlot, DiscThumb, LabeledPlot):
        pass

    class ExtDiscThumb(MultiCellPlot, LogScaleExtPlot, DiscThumb, LabeledPlot):
        pass

    def __init__(self, data):
        super().__init__(data)

    def plot(self, outdir):
        self.fig = plt.figure(figsize=(15, 5 * 2.67))
        self.gs = gridspec.GridSpec(2, 1, height_ratios=[25, 9])
        #self.gs = gridspec.GridSpec(6, 3, height_ratios=[2, 2, 2, 1, 2, 1])
        matrices = self.data.matrices()

        rows = 3
        columns = 3
        height_ratios = [item for sub in [[8] * rows, [1]] for item in sub]
        ogs = gridspec.GridSpecFromSubplotSpec(rows, columns, subplot_spec=self.gs[0], height_ratios=height_ratios)
        thumbs = []
        pos = 0
        for row, gene in enumerate(['Ato', 'AtoClean', 'ato']):
            matrix = matrices.loc[gene]
            thumbs = [self.GeneDiscThumb(self.fig, matrix['mean'], 'Mean expression'),
                      self.GeneDiscThumb(self.fig, matrix['max'], 'Max expression'),
                      self.ExtDiscThumb(self.fig, matrix['ext'], 'Max eccentricity')]
            for col, thumb in enumerate(thumbs):
                pos = (row * 3) + col
                text = chr(ord('A') + pos)
                thumb.plot(ogs[pos], text=text, controw=True,
                           firstcol=(col == 0), firstrow=(row == 0),
                           lastcol=(col == columns-1), lastrow=(row == rows-1))

        for col, thumb in enumerate(thumbs):
            thumb.legend(ogs[pos + col + 1])

        profiles = self.data.profiles()
        plots = []
        ato_protein = pd.DataFrame()
        ato_protein['Protein Mean'] = profiles.loc['Ato']['mean']
        ato_protein['Protein Q99'] = profiles.loc['Ato']['q99']
        ato_protein['Protein (clean) mean'] = profiles.loc['AtoClean']['mean']
        ato_protein['Protein (clean) Q99'] = profiles.loc['AtoClean']['q99']
        styles = {
            'Protein Mean': {'linestyle': 'dotted', 'color': '#2ca02c'},
            'Protein Q99': {'linestyle': 'dotted', 'color': '#d62728'},
            'Protein (clean) mean': {'color': '#2ca02c'},
            'Protein (clean) Q99': {'color': '#d62728'}
        }
        plots.append(self.GeneProfilePlot(self.fig, ato_protein, styles=styles))
        ato_reporter = pd.DataFrame()
        ato_reporter['Reporter mean'] = profiles.loc['ato']['mean']
        ato_reporter['Reporter Q99'] = profiles.loc['ato']['q99']
        ato_reporter['Protein (clean) mean'] = profiles.loc['AtoClean']['mean']
        plots.append(self.GeneProfilePlot(self.fig, ato_reporter))

        dv_profiles = self.data.dv_profiles()
        ato_dv = pd.DataFrame()
        ato_dv['Reporter mean'] = dv_profiles.loc['ato']['mean']
        ato_dv['Reporter Q99'] = dv_profiles.loc['ato']['q99']
        ato_dv['Protein (clean) mean'] = dv_profiles.loc['AtoClean']['mean']
        plots.append(self.DVGeneProfilePlot(self.fig, ato_dv))

        ogs = gridspec.GridSpecFromSubplotSpec(1, columns, subplot_spec=self.gs[2])

        def build_handles_labels(h, l, legend):
            new_h, new_l = legend
            for i, label in enumerate(new_l):
                if label not in l:
                    h.append(new_h[i])
                    l.append(new_l[i])

        handles = []
        labels = []
        for col, plot in enumerate(plots):
            text = chr(ord('A') + pos + col + 1)
            plot.plot(ogs[col], text=text, firstrow=False, lastrow=True)
            build_handles_labels(handles, labels, plot.ax.get_legend_handles_labels())

        ax = self.fig.add_subplot(self.gs[3])
        ax.set_axis_off()
        ax.legend(handles, labels, ncol=3, loc='center', frameon=False, fontsize=18)
        super().plot()


class Figure_79eb(Figure_3d51):

    def __init__(self, data, columns=2):
        super().__init__(data)
        self.columns = columns
        self.genes = self.data.genes_sorted()
        self.genes.remove('ato')
        self.rows = math.ceil(len(self.genes) / self.columns)

    def plot(self, outdir):
        e = 1 if (len(self.genes) % self.columns == 0) else 0
        rows = self.rows + e
        self.fig = plt.figure(figsize=(16 * self.columns, rows * 2.67))
        self.gs = gridspec.GridSpec(rows, self.columns)

        profiles = self.data.profiles()
        matrices = self.data.matrices()
        template = pd.DataFrame()
        template['Target mean'] = np.NaN
        template['Target Q99'] = np.NaN
        template['Ato protein'] = profiles.loc['AtoClean']['mean']
        n_genes = len(self.genes)

        def symbol(i):
            if n_genes <= 26:
                return chr(ord('A') + i)
            else:
                m = math.floor(index / 26)
                return chr(ord('A') + m) + chr(ord('A') + (i - (26 * m)))

        plots = []
        index = 0
        for index, gene in enumerate(self.genes):
            profile = profiles.loc[gene]
            matrix = matrices.loc[gene]
            gene_profiles = template.copy()
            gene_profiles['Target mean'] = profile['mean']
            gene_profiles['Target Q99'] = profile['q99']
            ogs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs[index], width_ratios=[1, 20])
            igs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=ogs[1])
            plots = [self.GeneDiscThumb(self.fig, matrix['mean'], 'Mean target expression'),
                     self.GeneDiscThumb(self.fig, matrix['max'], 'Max target expression'),
                     self.ExtDiscThumb(self.fig, matrix['ext'], 'Max target eccentricity'),
                     self.GeneProfilePlot(self.fig, gene_profiles)]

            ax = self.fig.add_subplot(ogs[0])
            ax.set_axis_off()
            ax.text(0.5, 0.5, gene, horizontalalignment='center', verticalalignment='center', fontsize=24, rotation=90)
            letter = symbol(index)
            for pid, plot in enumerate(plots):
                text = letter + '\'' * pid
                plot.plot(igs[pid], text=text, label=('right' if pid == 3 else 'left'),
                          firstcol=(pid == 0), firstrow=(index < self.columns),
                          lastcol=(pid == 3), lastrow=(index >= n_genes-self.columns), controw=True)

        ogs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs[index + 1], width_ratios=[1, 20])
        igs = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=ogs[1], height_ratios=[1, 4])
        for pid, plot in enumerate(plots):
            pos = igs[pid] if pid < 3 else igs[:, -1]
            plot.legend(pos)
        Figure.plot(self)


class Figure9d28(Figure):

    def plot(self, outdir):

        Figure.plot(self)

    # def plot(self):
    #
    #     def plot_centroids(s_centroids, g_centroids, name):
    #         fig = plt.figure(figsize=[5, 5])
    #         ax = fig.add_subplot(1, 1, 1)
    #         c_centroids = s_centroids.loc[s_centroids['Cluster'] != 0]
    #         ax.scatter(c_centroids['cy'], c_centroids['mCherry'], c=c_centroids['Cluster'], s=160, cmap='rainbow')
    #         ax.scatter(s_centroids['cy'], s_centroids['mCherry'], c=s_centroids['ext_mCherry'])
    #         ax.scatter(g_centroids['cy'], g_centroids['mCherry'], c='green', s=80, marker='D')
    #         if self.outdir is None:
    #             fig.show()
    #         else:
    #             fig.savefig(os.path.join(self.outdir, self.basename + '_' + name + '.png'))
    #
    #     def plot_samples(method):
    #         samples = self.cells.sort_values('Gene')['Sample'].unique()
    #         n_samples = len(samples)
    #         fig = plt.figure(figsize=[3 * 3, 3 * n_samples])
    #         gs = gridspec.GridSpec(n_samples, 1)
    #         g_centroids = self.cells.groupby('Cluster_' + method)[self.C_FEATURES].mean()
    #         for index, sample in enumerate(samples):
    #             cells = self.cells.loc[self.cells['Sample'] == sample].sort_values(by=['mCherry'])
    #             gene = cells['Gene'].unique()[0]
    #             s_centroids = cells.groupby('Cluster_' + method)[self.C_FEATURES].mean()
    #             sgs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[index])
    #             ax = fig.add_subplot(sgs[0])
    #             ax.scatter(cells['cx'], cells['cy'], c=cells['Cluster_' + method], cmap='rainbow')
    #             ax.set_xlim(0, 80)
    #             ax.set_ylim(10, -10)
    #             ax.text(0.025, 0.95, sample + ' ' + gene, horizontalalignment='left', verticalalignment='top',
    #                     fontsize=18, color='black', transform=ax.transAxes)
    #             ax = fig.add_subplot(sgs[1])
    #             ax.scatter(cells['cx'], cells['cy'], c=cells['mCherry'])
    #             ax.set_xlim(0, 80)
    #             ax.set_ylim(10, -10)
    #             ax = fig.add_subplot(sgs[2])
    #             ax.scatter(cells['cy'], cells['mCherry'], c=cells['Cluster_' + method], s=160, cmap='rainbow')
    #             ax.scatter(cells['cy'], cells['mCherry'], c=cells['ext_mCherry'])
    #             ax.scatter(g_centroids['cy'], g_centroids['mCherry'], c='magenta', s=80, marker='D')
    #             ax.scatter(s_centroids['cy'], s_centroids['mCherry'], c='yellow', s=80, marker='D')
    #             ax.set_xlim(-10, 10)
    #             ax.set_ylim(0, 4)
    #         if self.outdir is None:
    #             fig.show()
    #         else:
    #             fig.savefig(os.path.join(self.outdir, self.basename + '_' + method + '_samples.png'))
    #
    #     def sorted_legend(handles, labels=None):
    #         if labels is None:
    #             handles, labels = handles
    #         hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
    #         handles2, labels2 = zip(*hl)
    #         labels3 = [l.replace(' 0', ' ') for l in labels2]
    #         return handles2, labels3
    #
    #     def h_cluster_dendrogram(z, ax, c_colors):
    #         dd = dendrogram(
    #             z,
    #             truncate_mode='lastp',  # show only the last p merged clusters
    #             p=self.k,               # show only the last p merged clusters
    #             leaf_rotation=90.,      # rotates the x axis labels
    #             leaf_font_size=8,       # font size for the x axis labels
    #             show_contracted=True,   # to get a distribution impression in truncated branches
    #             ax=ax,
    #             link_color_func=lambda c: 'black'
    #         )
    #         ax.set_xticklabels(['Cluster %d' % c for c in range(1, len(ax.get_xmajorticklabels()) + 1)])
    #         x_lbls = ax.get_xmajorticklabels()
    #         num = -1
    #         for lbl in x_lbls:
    #             num += 1
    #             lbl.set_color(c_colors(num))
    #         return dd
    #
    #     plot_centroids(self.centroids.loc[idx[:, method], :], centroids, method + '_centroids')
    #     plot_centroids(
    #         self.cells.groupby(['Sample', 'Cluster_' + method], as_index=False)[self.C_FEATURES].mean()
    #             .rename(columns={'Cluster_' + method: 'Cluster'}),
    #         self.cells.groupby(['Cluster_' + method], as_index=False)[self.C_FEATURES].mean()
    #             .rename(columns={'Cluster_' + method: 'Cluster'}),
    #             method + '_sample_centroids')
    #
    #     # fig = plt.figure(figsize=[10, 10])
    #     # ax = fig.add_subplot(2, 2, 1)
    #     # ax_xi = fig.add_subplot(2, 2, 2)
    #     # ax_xy = fig.add_subplot(2, 2, 4)
    #
    #     # cells.loc[cells.groupby('Cluster')['Cluster'].transform('count').sort_values(ascending=False).index]
    #
    #     # clusters = cells.groupby('Cluster')['cx'].count().sort_values(ascending=False).index.tolist()
    #     # c_colors = plt.cm.get_cmap("gist_rainbow", len(clusters))
    #     # h_cluster_dendrogram(z, ax, c_colors)
    #     #
    #     # for cluster in clusters:
    #     #     if cluster == 0:
    #     #         continue
    #     #     c_cells = cells[cells['Cluster'] == cluster]
    #     #     c_count = c_cells['Cluster'].count()
    #     #     label = "Cluster " + '%02d' % cluster + " (" + str(c_count) + " cells)"
    #     #     ax_xi.scatter(c_cells['cy'], c_cells['mCherry'], c=[c_colors(cluster - 1)], label=label)
    #     #     ax_xy.scatter(c_cells['cx'], c_cells['cy'], c=[c_colors(cluster - 1)], label=label)
    #     #
    #     # handles, labels = sorted_legend(ax_xy.get_legend_handles_labels())
    #     # ax = fig.add_subplot(2, 2, 3)
    #     # ax.set_axis_off()
    #     # ax.legend(handles, labels, frameon=False, fontsize=15, loc='center')
    #     #
    #     # filename = id + '_' + method + '_' + str(self.k) + '.png'
    #     # fig.savefig(os.path.join(outdir, filename))
    #     # plt.close(fig)
    #     #
    #     # samples = cells['Sample'].unique().tolist()
    #     # for sample in samples:
    #     #     fig = plt.figure(figsize=[10, 10])
    #     #     ax_xyi = fig.add_subplot(2, 2, 1)
    #     #     ax_xi = fig.add_subplot(2, 2, 4)
    #     #     ax_xy = fig.add_subplot(2, 2, 2)
    #     #     s_cells = cells[cells['Sample'] == sample].sort_values('mCherry')
    #     #     ax_xyi.scatter(s_cells['cx'], s_cells['cy'], c=s_cells['mCherry'])
    #     #     for cluster in clusters:
    #     #         c_cells = s_cells[s_cells['Cluster'] == cluster]
    #     #         c_count = c_cells['Cluster'].count()
    #     #         label = "Cluster " + '%02d' % cluster + " (" + str(c_count) + " cells)"
    #     #         ax_xi.scatter(c_cells['cy'], c_cells['mCherry'], c=[c_colors(cluster - 1)], label=label)
    #     #         ax_xy.scatter(c_cells['cx'], c_cells['cy'], c=[c_colors(cluster - 1)], label=label)
    #     #     handles, labels = sorted_legend(ax_xy.get_legend_handles_labels())
    #     #     ax = fig.add_subplot(2, 2, 3)
    #     #     ax.set_axis_off()
    #     #     ax.legend(handles, labels, frameon=False, fontsize=15, loc='center')
    #     #     filename = id + '_' + method + '_' + str(self.k) + '_' + sample + '.png'
    #     #     fig.savefig(os.path.join(outdir, filename))
    #     #     plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot all data.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--log')
    parser.add_argument('--outdir')
    parser.add_argument('-k', '--clusters', dest='k', type=int, default=6)
    parser.add_argument('-n', '--samples', dest='n', type=int, default=20)
    parser.add_argument('-r', '--repeats', dest='r', type=int, default=100)
    parser.add_argument('--cutoff', type=float, default=1.0)
    parser.add_argument('--reproducible', dest='reproducible', action='store_true')
    parser.add_argument('--not-reproducible', dest='reproducible', action='store_false')
    parser.set_defaults(reproducible=False)
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=args.log.upper())
        logging.getLogger('PIL.Image').setLevel(logging.INFO)
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger('joblib').setLevel(logging.INFO)
        logging.getLogger('cloudpickle').setLevel(logging.INFO)

    if args.reproducible:
        np.random.seed(0)
    else:
        np.random.seed()

    data = DiscData(args.data)
    clustering = Clustering(args.data, disc_data=data, args=args)
