#!/usr/bin/env python3

import argparse
import logging
import os
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import cpu_count, Pool
import random
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.signal import savgol_filter
from scipy.spatial.distance import squareform
import scipy.optimize
from shapely.geometry import Polygon


GX_MIN = 0
GX_MAX = 80
GY_MIN = -10
GY_MAX = 45
MF_MIN = -10
MF_MAX = 10


class Figure:

    def __init__(self, data):
        self.data = data


class Plot:


    @staticmethod
    def x_lim():
        return False

    @staticmethod
    def y_lim():
        return False

    @staticmethod
    def v_lim():
        return False

    @staticmethod
    def x_scale():
        return False

    @staticmethod
    def y_scale():
        return False

    @staticmethod
    def v_scale():
        return False

    @staticmethod
    def x_ticks():
        return False

    @staticmethod
    def y_ticks():
        return False

    @staticmethod
    def v_ticks():
        return False


class DiscThumb(Plot):
    """
    Plot a disc thumbnail
    """
    def __init__(self, fig, gs, data):
        self.fig = fig
        self.gs = gs
        self.data = data
        self.ax = None
        self.img = None
        self.index = self.data.index.to_frame()
        self.extent = [self.index['cx'].min(), self.index['cx'].max(),
                       self.index['cy'].min(), self.index['cy'].max()]

    def plot(self, position, cmap=None):
        self.ax = self.fig.add_subplot(self.gs[position])
        self.img = self.ax.imshow(self.disc_matrix(), extent=self.extent, norm=self.norm(), cmap=cmap, aspect='auto')
        self.ax.set_facecolor('black')
        self.ax.set_xlim(self.x_lim())
        self.ax.set_ylim(self.y_lim())
        self.ax.xaxis.tick_top()

    def disc_matrix(self):
        x = self.index['cx']
        y = self.index['cy'] - self.extent[2]
        v = self.data
        matrix = np.full([self.extent[3] - self.extent[2] + 1, self.extent[1] - self.extent[0] + 1], np.NaN)
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
    def x_lim():
        return [GX_MIN, GX_MAX]

    @staticmethod
    def y_lim():
        return [GY_MIN, GY_MAX]


class GeneDiscThumb(DiscThumb):

    @staticmethod
    def v_lim(): return [0.1, 30]

    @staticmethod
    def v_scale(): return 'log'

    @staticmethod
    def v_ticks(): return [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]


class ExtDiscThumb(GeneDiscThumb):

    @staticmethod
    def v_lim(): return [0.1, 20]


class ProfilePlot(Plot):

    def __init__(self, fig, gs, data):
        self.fig = fig
        self.gs = gs
        self.data = data
        self.ax = None

    def plot(self, position):
        self.ax = self.fig.add_subplot(self.gs[position])
        self.plot_profiles()
        self.format_axis()

    def plot_profile(self, data, preprocessor=None, style=None):
        x = data.index
        y = preprocessor(data.values) if preprocessor else data.values
        self.ax.plot(x, y, label=data, **style)

    def plot_profiles(self):
        preprocessors = self.preprocessors()
        styles = self.styles()
        for profile in self.data:
            preprocessor = preprocessors[profile] if preprocessors and preprocessors[profile] else None
            style = styles[profile] if styles and styles[profile] else None
            self.plot_profile(profile, preprocessor=preprocessor, style=style)
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

    @staticmethod
    def preprocessors():
        return False

    @staticmethod
    def styles():
        return False


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
        self._matrices = None
        self.clean_up()

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

    def cells(self):
        return self._cells

    def cells_clean(self):
        if self._cells_clean is None:
            self._cells_clean = self._cells[self._cells['Gene'].isin(self.genes_clean())]
        return self._cells_clean

    def cells_background(self):
        if self._cells_background in None:
            self._cells_background = data[(data['cy'] >= -10) & (data['cy'] <= -5)]
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

    def matrices(self):
        if self._matrices is None:
            self._profiles_matrices()
        return self._matrices

    @staticmethod
    def q99(x): return np.percentile(x, 0.99)

    def _profiles_matrices(self):
        profiles = []
        matrices = []

        cells = self.cells()
        cells_clean = self.cells_clean()

        cx = cells_clean['cx'].round().astype('int')
        cy = cells_clean['cy'].round().astype('int')
        profile = cells_clean.groupby(cy)['mCherry'].agg([np.mean, self.q99])
        profiles.append(pd.concat([profile], keys=['AtoClean'], names=['Gene']))
        matrix = cells_clean.groupby([cx, cy])['mCherry', 'ext_mCherry'].agg(
            {'mCherry': [np.mean, np.max], 'ext_mCherry': np.max})
        matrix.columns = ['mean', 'max', 'ext']
        matrices.append(pd.concat([matrix], keys=['AtoClean'], names=['Gene']))

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

        self._profiles = pd.concat(profiles)
        self._matrices = pd.concat(matrices)


class Figure_79eb:

    def __init__(self, data, columns=2):
        self.data = data
        self.fig = None
        self.gs = None
        self.columns = columns
        self.genes = self.data.genes_sorted()
        self.genes.remove('ato')
        self.rows = math.ceil(len(self.genes) / self.columns)

    def geom(self):
        self.fig = plt.figure(figsize=(32, self.rows * 2.65))
        width_ratios = [item for sub in [[1], [16] * 4] * self.columns for item in sub]
        height_ratios = [item for sub in [[8] * self.rows, [1]] for item in sub]
        self.gs = gridspec.GridSpec(self.rows + 1, 5 * self.columns,
                                    width_ratios=width_ratios, height_ratios=height_ratios)

    def plot(self):
        self.geom()
        profiles = self.data.profiles()
        matrices = self.data.matrices()
        template = pd.DataFrame()
        template['Target mean'] = np.NaN
        template['Target Q99'] = np.NaN
        template['Ato protein'] = profiles.loc['Ato']['mean']
        for gene in self.genes:
            profile = profiles.loc[gene]
            matrix = matrices.loc[gene]
            gene_profiles = template.copy()
            gene_profiles['Target mean'] = profile['mean']
            gene_profiles['Target Q99'] = profile['q99']
            plots = [GeneDiscThumb(self.fig, self.gs, matrix['mean']), GeneDiscThumb(self.fig, self.gs, matrix['max']),
                     ExtDiscThumb(self.fig, self.gs, matrix['ext']), ProfilePlot(self.fig, self.gs, gene_profiles)]


parser = argparse.ArgumentParser(description='Plot all data.')
parser.add_argument('--data', required=True)
parser.add_argument('--log')
parser.add_argument('--outdir')
args = parser.parse_args()

if args.log:
    logging.basicConfig(level=args.log.upper())
    logging.getLogger('PIL.Image').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)

print(pd.__version__)

data = DiscData(args)
fig_76eb = Figure_79eb(data)
fig_76eb.plot()
