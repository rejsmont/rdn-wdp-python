#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import savgol_filter


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


class DiscData:

    def __init__(self, args):
        self._cells = pd.read_csv(args.data)
        self._cells_clean = None
        self._cells_background = None
        self._cells_ato = None
        self._cells_no_ato = None
        self._genes = None
        self._genes_sorted = None
        self._profiles = None
        self._dv_profiles = None
        self._matrices = None
        self.clean_up()
        self._clean_mask = None
        self._background_mask = None
        self._furrow_mask = None

    def clean_up(self):
        # Remove artifact from sample ZBO7IH
        artifact = self._cells[(self._cells['Sample'] == 'ZBO7IH') &
                               (self._cells['cy'] > 35) &
                               (self._cells['cx'] > 20) &
                               (self._cells['cx'] < 30)].index
        self._cells = self._cells.drop(artifact)

        # Mark and remove bad CG9801 samples
        bad_samples = ['J0RYWJ', '3SKX4V', '7AMINR', '4EAAEF', 'VH2DCR', 'WJ8F8M', 'ZNVOPe', 'APKoAe', 'zfroDh',
                              'lgxpL6', 'pcTNzE', '80IkVQ', 'UQZJ3K']
        self._cells.loc[self._cells['Sample'].isin(bad_samples), 'Gene'] = 'CG9801-B'
        bad_cells = self._cells[self._cells['Gene'] == 'CG9801-B'].index
        self._cells = self._cells.drop(bad_cells)

    # Binary masks for identifying cell types #

    def clean_mask(self):
        if self._clean_mask is None:
            self._clean_mask = self._cells['Gene'].isin(self.genes_clean())
        return self._clean_mask

    def furrow_mask(self):
        if self._furrow_mask is None:
            self._furrow_mask = (self._cells['cy'] >= -10) & (self._cells['cy'] <= 10)
        return self._furrow_mask

    def background_mask(self):
        if self._background_mask is None:
            self._background_mask = (self._cells['cy'] >= -10) & (self._cells['cy'] <= -5)
        return self._background_mask


    # Shortcuts to get cells (filtered) #

    def cells(self):
        return self._cells

    def cells_clean(self):
        if self._cells_clean is None:
            self._cells_clean = self._cells[self._cells['Gene'].isin(self.genes_clean())]
        return self._cells_clean

    def cells_background(self):
        if self._cells_background in None:
            self._cells_background = self._cells[(self._cells['cy'] >= -10) & (self._cells['cy'] <= -5)]
        return self._cells_background

    def cells_ato(self):
        if self._cells_ato is None:
            cells = self._cells()
            background = self.cells_background()
            self._cells_ato = cells[(cells['mCherry'] > background['mCherry'].quantile(0.90))]
        return self._cells_ato

    def cells_no_ato(self):
        if self._cells_no_ato is None:
            cells = self._cells()
            background = self.cells_background()
            self._cells_no_ato = cells[(cells['mCherry'] < background['mCherry'].quantile(0.50))]
        return self._cells_no_ato

    def genes(self):
        if not self._genes:
            self._genes = self._cells['Gene'].unique().tolist()
        return self.genes_sorted

    def genes_sorted(self):
        if self._genes_sorted is None:
            before = self._cells[self._cells['cy'] < 0].groupby(['Gene'])['Venus'].quantile(0.99)
            after = self._cells[(self._cells['cy'] > 0) & (self._cells['cy'] < 20)].groupby(['Gene'])['Venus'].quantile(0.99)
            ratio = after / before
            self._genes_sorted = ratio.sort_values(ascending=False).index.tolist()
        return self._genes_sorted

    @staticmethod
    def genes_clean():
        return ['CG31176', 'beat-IIIc', 'king-tubby', 'lola-P', 'nmo', 'sNPF', 'Vn', 'Fas2', 'siz']

    def profiles(self):
        if self._profiles is None:
            self._profiles_matrices()
        return self._profiles

    def dv_profiles(self):
        if self._dv_profiles is None:
            self._profiles_matrices()
        return self._dv_profiles

    def matrices(self):
        if self._matrices is None:
            self._profiles_matrices()
        return self._matrices

    @staticmethod
    def q99(x): return np.percentile(x, 99)

    def _profiles_matrices(self):
        profiles = []
        dv_profiles = []
        matrices = []

        cells = self.cells()
        cells_mf = cells[(cells['cy'] >= -3) & (cells['cy'] <= 3)]
        cells_clean = self.cells_clean()
        cells_mf_clean = cells_clean[(cells_clean['cy'] >= -3) & (cells_clean['cy'] <= 3)]

        cx = cells_clean['cx'].round().astype('int')
        cy = cells_clean['cy'].round().astype('int')
        profile = cells_clean.groupby(cy)['mCherry'].agg([np.mean, self.q99])
        profiles.append(pd.concat([profile], keys=['AtoClean'], names=['Gene']))
        matrix = cells_clean.groupby([cx, cy])['mCherry', 'ext_mCherry'].agg(
            {'mCherry': [np.mean, np.max], 'ext_mCherry': np.max})
        matrix.columns = ['mean', 'max', 'ext']
        matrices.append(pd.concat([matrix], keys=['AtoClean'], names=['Gene']))
        cx = cells_mf_clean['cx'].round().astype('int')
        profile = cells_mf_clean.groupby(cx)['mCherry'].agg([np.mean, self.q99])
        dv_profiles.append(pd.concat([profile], keys=['AtoClean'], names=['Gene']))

        cx = cells['cx'].round().astype('int')
        cy = cells['cy'].round().astype('int')
        profile = cells.groupby(cy)['mCherry'].agg([np.mean, self.q99])
        profiles.append(pd.concat([profile], keys=['Ato'], names=['Gene']))
        matrix = cells.groupby([cx, cy])['mCherry', 'ext_mCherry'].agg(
            {'mCherry': [np.mean, np.max], 'ext_mCherry': np.max})
        matrix.columns = ['mean', 'max', 'ext']
        matrices.append(pd.concat([matrix], keys=['Ato'], names=['Gene']))

        profile = cells.groupby(['Gene', cy])['Venus'].agg([np.mean, self.q99])
        profiles.append(profile)

        matrix = cells.groupby(['Gene', cx, cy])['Venus', 'ext_Venus'].agg(
            {'Venus': [np.mean, np.max], 'ext_Venus': np.max})
        matrix.columns = ['mean', 'max', 'ext']
        matrices.append(matrix)

        cx = cells_mf['cx'].round().astype('int')
        profile = cells_mf.groupby(cx)['mCherry'].agg([np.mean, self.q99])
        dv_profiles.append(pd.concat([profile], keys=['Ato'], names=['Gene']))
        profile = cells_mf.groupby(['Gene', cx])['Venus'].agg([np.mean, self.q99])
        dv_profiles.append(profile)

        self._profiles = pd.concat(profiles)
        self._dv_profiles = pd.concat(dv_profiles)
        self._matrices = pd.concat(matrices)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot all data.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--log')
    parser.add_argument('--outdir')
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=args.log.upper())
        logging.getLogger('PIL.Image').setLevel(logging.INFO)
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger('joblib').setLevel(logging.INFO)

    data = DiscData(args)
    #fig_3d51 = Figure_3d51(data)
    #fig_3d51.show()
    #fig_76eb = Figure_79eb(data)
    #fig_76eb.show()

