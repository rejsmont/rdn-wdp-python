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

# Figure labels:
# ceb2
# 9a76
# 5bd1
# 37a5
# ede8
# 422e
# 0d67
# 6313
# 1d46
# a2a8
# 7e0b
# 0ac7
# de30
# e170
# 1718
# 7895
# 9d28


def disc_matrix(disc, field, method='max'):
    """
    :param disc:        Data to process
    :param field:       Field to read intensities from
    :param method:      Method used to project data (default: max)
    :return:            2D Numpy array with intensity values
    """
    data = pd.DataFrame()
    shift = disc['cy'].min()
    if shift >= 0:
        shift = 0
    # Compute rounded coordinates
    data['ux'] = round(disc['cx'])
    data['uy'] = round(disc['cy'] - shift)
    data['uz'] = round(disc['cz'])
    data['int'] = disc[field]
    # Compute bounds
    x_min = int(data['ux'].min())
    x_max = int(data['ux'].max()) + 1
    y_min = int(data['uy'].min())
    y_max = int(data['uy'].max()) + 1
    # Initialize empty array
    matrix = np.zeros([x_max, y_max])
    # Compute matrix values
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            if method == 'max':
                z = data[(data['ux'] == x) & (data['uy'] == y)]['int'].max()
            if method == 'mean':
                z = data[(data['ux'] == x) & (data['uy'] == y)]['int'].mean()
            elif method == 'sum':
                z = data[(data['ux'] == x) & (data['uy'] == y)]['int'].sum()
            elif method == 'count':
                z = data[(data['ux'] == x) & (data['uy'] == y)]['int'].count()
            matrix[x, y] = z
    # Compute smoothened matrix
    smooth = np.zeros([x_max, y_max])
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            if math.isnan(matrix[x, y]):
                i = 0
                z = 0
                for ix in range(x - 1, x + 2):
                    if x_min <= ix < x_max:
                        for iy in range(y - 1, y + 2):
                            if y_min <= iy < y_max:
                                v = matrix[ix, iy]
                                if not math.isnan(v):
                                    z = z + v
                                    i = i + 1
                if i:
                    smooth[x, y] = z / i
                else:
                    smooth[x, y] = 0
            else:
                smooth[x, y] = matrix[x, y]
    return np.transpose(smooth)


def display_normalize(data, vmin=None, vmax=None, log=False):
    if log:
        data[data == 0] = vmin
        data = np.log10(data)
        vmin = math.log10(vmin)
        vmax = math.log10(vmax)
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = (data.mean() * 2)
    if data.max() == 1:
        return (data * 255).clip(0, 255).astype('uint8')
    else:
        return (((data-vmin) / (vmax-vmin)) * 255).clip(0, 255).astype('uint8')


parser = argparse.ArgumentParser(description='Plot all data.')
parser.add_argument('--data', required=True)
parser.add_argument('--log')
parser.add_argument('--outdir')
args = parser.parse_args()

if args.log:
    logging.basicConfig(level=args.log.upper())
    logging.getLogger('PIL.Image').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)

input = pd.read_csv(args.data)

gxmin = 0
gxmax = 80
gymin = -10
gymax = 45

cherry_min = input['mCherry'].min()
cherry_max = input['mCherry'].mean() * 5

dapi_min = input['DAPI'].min()
dapi_max = input['DAPI'].mean() * 2

venus_min = input['Venus'].min()
venus_max = input['Venus'].mean()

vmin = 0.1
vmax = 15

clean = ['CG31176', 'beat-IIIc', 'king-tubby', 'lola-P', 'nmo', 'sNPF', 'Vn', 'Fas2', 'siz']

# Remove artifact from sample ZBO7IH
artifact = input[(input['Sample'] == 'ZBO7IH') & (input['cy'] > 35) & (input['cx'] > 20) & (input['cx'] < 30)].index
input = input.drop(artifact)

# Mark and remove bad CG9801 samples
CG9801_bad_samples = ['J0RYWJ', '3SKX4V', '7AMINR', '4EAAEF', 'VH2DCR', 'WJ8F8M', 'ZNVOPe', 'APKoAe', 'zfroDh', 'lgxpL6',
              'pcTNzE', '80IkVQ', 'UQZJ3K']
input.loc[input['Sample'].isin(CG9801_bad_samples), 'Gene'] = 'CG9801-B'
CG9801_bad_cells = input[input['Gene'] == 'CG9801-B'].index
input = input.drop(CG9801_bad_cells)


def plot_image(ax, data, channel, projection='mean', norm=None, cmap=None):
    xmin = round(data['cx'].min())
    xmax = round(data['cx'].max())
    ymin = round(data['cy'].min())
    ymax = round(data['cy'].max())
    img = ax.imshow(disc_matrix(data, channel, method=projection),
                    extent=[xmin, xmax, ymax, ymin], norm=norm, cmap=cmap, aspect='auto')
    ax.set_facecolor('black')
    ax.set_xlim(gxmin, gxmax)
    ax.set_ylim(gymax, gymin)
    ax.xaxis.tick_top()
    return img, ax


@ticker.FuncFormatter
def major_formatter_log(x, pos):
    return "%g" % (round(x * 10) / 10)


def genes_sorted(data):
    before = data[data['cy'] < 0].groupby(['Gene'])['Venus'].quantile(0.99)
    after = data[(data['cy'] > 0) & (data['cy'] < 20)].groupby(['Gene'])['Venus'].quantile(0.99)
    ratio = after / before
    return ratio.sort_values(ascending=False).index.tolist()


# TODO: make this generic
def e_series(min = 0, max = 100, res=3):
    return [0.1, 0.22, 0.47, 1, 2.2, 4.7, 10, 22]


def y_profile(data, channel):
    return data.groupby([round(data['cy'])])[channel]


def x_profile(data, channel):
    return data[(data['cy'] >= -3) & (data['cy'] <= 3)].groupby([round(data['cx'])])[channel]


def plot_profile(ax, data, label, linestyle=None, color=None):
    x = data.index
    y = savgol_filter(data.values, 9, 3, mode='nearest')
    ax.plot(x, y, label=label, linestyle=linestyle, color=color)


def format_axis(ax, ticks, axis='y'):
    if axis == 'y':
        ax.set_xlim(gymin, gymax)
    else:
        ax.set_xlim(gxmin, gxmax)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 30)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(major_formatter_log)


def plot_profiles(ax, profiles, styles, ticks, axis='y'):
    for index, profile in enumerate(profiles):
        plot_profile(ax, profile, **styles[index])
    format_axis(ax, ticks, axis)
    return ax.get_legend_handles_labels()


def plot_profile_2(ax, data, preprocessor=None, style=None):
    x = data.index
    y = preprocessor(data.values) if preprocessor else data.values
    ax.plot(x, y, label=data, **style)


def plot_profiles_2(ax, profiles, ticks, styles=None, preprocessors=None, axis='y'):
    for profile in profiles:
        preprocessor = preprocessors[profile] if preprocessors and preprocessors[profile] else None
        style = styles[profile] if styles and styles[profile] else None
        plot_profile_2(ax, profile, preprocessor=preprocessor, style=style)
    format_axis(ax, ticks, axis)
    return ax.get_legend_handles_labels()


def fig_79eb(data, columns):
    genes = genes_sorted(data)
    genes.remove('ato')
    rows = math.ceil(len(genes) / columns)
    fig = plt.figure(figsize=(32, rows * 2.65))
    width_ratios = [item for sub in [[1], [16] * 4] * columns for item in sub]
    height_ratios = [item for sub in [[8] * rows, [1]] for item in sub]
    gs = gridspec.GridSpec(rows + 1, 5 * columns, width_ratios=width_ratios, height_ratios=height_ratios)
    legend_data = ()
    ato = y_profile(data[data['Gene'].isin(clean)], 'mCherry')
    ato_mean = ato.mean()

    def fig_79eb_row(gene, index, ticks):
        row = math.ceil(index / columns)
        pos = (index - 1) * 5
        data = input[input['Gene'] == gene]
        if rows <= 13:
            symbol = chr(ord('A') + (index - 1))
        else:
            major = math.floor((index - 1) / 26)
            symbol = chr(ord('A') + major) + chr(ord('A') + (index - (26 * major) - 1))

        def fig_79eb_image(position, channel, projection, cmap, norm):
            text = symbol + '\'' * (position - 1)
            img, ax = plot_image(fig.add_subplot(gs[pos + position]), data, channel, projection=projection,
                                 norm=norm, cmap=cmap)
            ax.text(0.025, 0.95, text, horizontalalignment='left', verticalalignment='top', fontsize=24,
                    color='white', transform=ax.transAxes)
            ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                           left=True, right=False, labelleft=(position == 1), labelright=False)
            return img, ax

        def fig_79eb_profile(position):
            text = symbol + '\'' * (position - 1)
            target = y_profile(data, 'Venus')
            ax = fig.add_subplot(gs[pos + position])
            profiles = pd.DataFrame()
            profiles['Target mean'] = target.mean()
            profiles['Target Q99'] = target.quantile(0.99)
            profiles['Ato protein'] = ato_mean
            plot_profiles_2(ax, profiles, ticks=ticks)
            format_axis(ax, ticks)
            ax.text(0.025, 0.95, text, horizontalalignment='left', verticalalignment='top', fontsize=24,
                    transform=ax.transAxes)
            ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                           left=False, right=True, labelleft=False, labelright=True)
            ax.tick_params(axis='y', which='minor', left=False, right=False, labelleft=False, labelright=False)
            handles, labels = ax.get_legend_handles_labels()
            ax.yaxis.set_major_formatter(major_formatter_log)
            return ax, handles, labels

        ax = fig.add_subplot(gs[pos])
        ax.set_axis_off()
        ax.text(0.5, 0.5, gene, horizontalalignment='center', verticalalignment='center', fontsize=24, rotation=90)

        norm = colors.LogNorm(vmin=0.1, vmax=30)
        img0, ax = fig_79eb_image(1, 'Venus', 'mean', 'plasma', norm)
        img1, ax = fig_79eb_image(2, 'Venus', 'max', 'plasma', norm)
        norm = colors.LogNorm(vmin=0.1, vmax=25)
        img2, ax = fig_79eb_image(3, 'ext_Venus', 'max', 'viridis', norm)
        ax, handles, labels = fig_79eb_profile(4)

        return img0, img1, img2, handles, labels

    def fig_79eb_legends(img0, img1, img2, handles, labels, ticks):
        origin = rows * columns * 5
        cax = fig.add_subplot(gs[origin + 1])
        fig.colorbar(img0, cax=cax, orientation='horizontal', ticks=ticks,
                     format=major_formatter_log, label='Mean target expression')
        cax = fig.add_subplot(gs[origin + 2])
        fig.colorbar(img1, cax=cax, orientation='horizontal', ticks=ticks,
                     format=major_formatter_log, label='Max target expression')
        cax = fig.add_subplot(gs[origin + 3])
        fig.colorbar(img2, cax=cax, orientation='horizontal', ticks=ticks,
                     format=major_formatter_log, label='Max target eccentricity')
        cax = fig.add_subplot(gs[origin + 4])
        cax.set_axis_off()
        cax.legend(handles, labels, ncol=2, loc='center', frameon=False)

    for index, gene in enumerate(genes):
        print(index, gene)
        legend_data = fig_79eb_row(gene, index + 1, e_series())
    fig_79eb_legends(*legend_data, e_series())

    return fig


def fig_3d51(data):
    rows = 4
    fig = plt.figure(figsize=(12, rows * 3))
    width_ratios = [item for item in [4] * 3]
    height_ratios = [item for sub in [[8] * 3, [1], [8], [4]] for item in sub]
    gs = gridspec.GridSpec(rows + 2, 3, width_ratios=width_ratios, height_ratios=height_ratios)
    data_ato = data[data['Gene'] == 'ato']
    data_clean = data[data['Gene'].isin(clean)]

    def fig_3d51_image(position, data, channel, projection, cmap, norm):
        text = chr(ord('A') + (position - 1))
        img, ax = plot_image(fig.add_subplot(gs[position - 1]), data, channel, projection=projection,
                             norm=norm, cmap=cmap)
        ax.text(0.025, 0.95, text, horizontalalignment='left', verticalalignment='top', fontsize=24,
                color='white', transform=ax.transAxes)
        ax.tick_params(bottom=True, top=True, labelbottom=(math.ceil(position / 3) == 3),
                       labeltop=(math.ceil(position / 3) == 1), left=True, right=False,
                       labelleft=(position % 3 == 1), labelright=False)
        return img, ax

    def fig_3d51_img_row(data, protein, row):
        norm = colors.LogNorm(vmin=0.1, vmax=30)
        img0, ax = fig_3d51_image((row - 1) * 3 + 1, data, protein, 'mean', 'plasma', norm)
        img1, ax = fig_3d51_image((row - 1) * 3 + 2, data, protein, 'max', 'plasma', norm)
        norm = colors.LogNorm(vmin=0.1, vmax=25)
        img2, ax = fig_3d51_image((row - 1) * 3 + 3, data, 'ext_' + protein, 'max', 'viridis', norm)

        return img0, img1, img2

    def fig_3d51_colorbar(position, img, label):
        cax = fig.add_subplot(gs[position])
        fig.colorbar(img, cax=cax, orientation='horizontal', ticks=e_series(),
                            format=major_formatter_log, label=label)

    def fig_3d51_profile_row(ticks):

        def plot_profiles(position, profiles, styles, axis='y'):
            ax = fig.add_subplot(gs[position])
            text = chr(ord('A') + (position - 3))
            ax.text(0.025, 0.95, text, horizontalalignment='left', verticalalignment='top', fontsize=24,
                    color='black', transform=ax.transAxes)
            for index, profile in enumerate(profiles):
                plot_profile(ax, profile, **styles[index])
            format_axis(ax, ticks, axis)
            ax.tick_params(bottom=True, top=False, labelbottom=True, labeltop=False,
                           left=True, right=False, labelleft=(position % 3 == 0), labelright=False)
            ax.tick_params(axis='y', which='minor', left=False, right=False, labelleft=False, labelright=False)
            return ax.get_legend_handles_labels()

        ato_protein = y_profile(data, 'mCherry')
        ato_clean = y_profile(data_clean, 'mCherry')
        ato_reporter = y_profile(data_ato, 'Venus')
        ato_clean_x = x_profile(data_clean, 'mCherry')
        ato_reporter_x = x_profile(data[data['Gene'] == 'ato'], 'Venus')

        profiles = [ato_protein.mean(), ato_protein.quantile(0.99), ato_clean.mean(), ato_clean.quantile(0.99)]
        styles = [
            {'label': 'Protein mean', 'linestyle': 'dotted', 'color': '#2ca02c'},
            {'label': 'Protein Q99', 'linestyle': 'dotted', 'color': '#d62728'},
            {'label': 'Protein (clean) mean', 'color': '#2ca02c'},
            {'label': 'Protein (clean) Q99', 'color': '#d62728'}
        ]
        handles, labels = plot_profiles(12, profiles, styles)

        profiles = [ato_reporter.mean(), ato_reporter.quantile(0.99), ato_clean.mean()]
        styles = [
            {'label': 'Reporter mean'},
            {'label': 'Reporter Q99'},
            {'label': 'Protein (clean) mean'}
        ]
        handles1, labels1 = plot_profiles(13, profiles, styles)

        profiles = [ato_reporter_x.mean(), ato_reporter_x.quantile(0.99), ato_clean_x.mean()]
        plot_profiles(14, profiles, styles, 'x')

        handles.append(handles1[0])
        handles.append(handles1[1])
        labels.append(labels1[0])
        labels.append(labels1[1])

        ax = fig.add_subplot(gs[5, 0:])
        ax.set_axis_off()
        ax.legend(handles, labels, ncol=3, loc='center', frameon=False)

    fig_3d51_img_row(data, 'mCherry', 1)
    fig_3d51_img_row(data_clean, 'mCherry', 2)
    img0, img1, img2 = fig_3d51_img_row(data_ato, 'Venus', 3)

    fig_3d51_colorbar(9, img0, 'Mean expression')
    fig_3d51_colorbar(10, img1, 'Max expression')
    fig_3d51_colorbar(11, img2, 'Max eccentricity')

    fig_3d51_profile_row(e_series())

    return fig


def fig_32b7(data, columns=5):
    genes = genes_sorted(data)
    cells = data[(data['cy'] >= -10) & (data['cy'] <= 10)]
    background = data[(data['cy'] >= -10) & (data['cy'] <= -5)]
    ato_cells = cells[(cells['mCherry'] > background['mCherry'].quantile(0.90))]
    no_ato_cells = cells[(cells['mCherry'] < background['mCherry'].quantile(0.50))]
    rows = math.ceil(len(genes) / columns)
    fig = plt.figure(figsize=(15, rows * 3))
    gs = gridspec.GridSpec(rows, columns)
    ato = y_profile(cells[cells['Gene'].isin(clean)], 'mCherry')

    for index, gene in enumerate(genes):
        row = math.ceil((index + 1) / columns)
        ato_gene_cells = ato_cells[ato_cells['Gene'] == gene]
        no_ato_gene_cells = no_ato_cells[no_ato_cells['Gene'] == gene]
        ato_target = y_profile(ato_gene_cells, 'Venus')
        no_ato_target = y_profile(no_ato_gene_cells, 'Venus')
        ax = fig.add_subplot(gs[index])
        profiles = [ato_target.mean(), no_ato_target.mean(), ato.mean()]
        styles = [{'label': 'Target mean (ato+)'}, {'label': 'Target mean (ato-)'}, {'label': 'Ato protein'}]
        plot_profiles(ax, profiles, styles, e_series())
        ax.set_xlim(-10, 10)
        ax.set_yscale('log')
        ax.set_ylim(0.05, 10)
        ax.set_yticks(e_series())
        ax.yaxis.set_major_formatter(major_formatter_log)
        ax.text(0.025, 0.95, gene, horizontalalignment='left', verticalalignment='top', fontsize=24,
                transform=ax.transAxes)
        ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                       left=True, right=False, labelleft=(index % 5 == 0), labelright=False)
        ax.tick_params(axis='y', which='minor', left=False, right=False, labelleft=False, labelright=False)
        handles, labels = ax.get_legend_handles_labels()
        ax.yaxis.set_major_formatter(major_formatter_log)

    ax = fig.add_subplot(gs[len(genes):])
    ax.set_axis_off()
    ax.legend(handles, labels, ncol=1, loc='center', frameon=False)

    return fig


def fig_01a8(data):
    """
    Cluster genes based on the amount of regulation by Atonal

    :param data: Input data
    :return: The dendrogram figure
    """
    def distance_matrix(data):
        matrix = np.full((data.size, data.size), float('inf'))

        for i in range(0, data.size):
            a = data[i]
            for j in range(0, i + 1):
                b = data[j]
                matrix[i, j] = abs(a - b)

        for j in range(0, data.size):
            for i in range(0, j + 1):
                matrix[i, j] = matrix[j, i]

        return matrix

    genes = genes_sorted(data)
    cells = data[(data['cy'] > -10) & (data['cy'] < 10)]
    background = data[(data['cy'] >= -10) & (data['cy'] <= -5)]
    ato_cells = cells[(cells['mCherry'] > background['mCherry'].quantile(0.90))]
    no_ato_cells = cells[(cells['mCherry'] < background['mCherry'].quantile(0.50))]
    areas = pd.DataFrame(data={'area': np.zeros(len(genes))}, index=genes)
    fig = plt.figure()

    for index, gene in enumerate(genes):
        ato_gene_cells = ato_cells[ato_cells['Gene'] == gene]
        no_ato_gene_cells = no_ato_cells[no_ato_cells['Gene'] == gene]
        profiles = pd.DataFrame()
        profiles['ato'] = y_profile(ato_gene_cells, 'Venus').mean()
        profiles['no_ato'] = y_profile(no_ato_gene_cells, 'Venus').mean()
        profiles = profiles.dropna()

        gradients = pd.DataFrame(index=profiles.index)
        gradients['ato'] = np.gradient(savgol_filter(profiles['ato'].values, 9, 3, mode='nearest'))
        gradients['no_ato'] = np.gradient(savgol_filter(profiles['no_ato'].values, 9, 3, mode='nearest'))

        ax = fig.subplots()
        gradients['ato_smooth'] = savgol_filter(gradients['ato'].values, 9, 3, mode='nearest')
        gradients['no_ato_smooth'] = savgol_filter(gradients['no_ato'].values, 9, 3, mode='nearest')
        polycol = ax.fill_between(gradients.index, gradients['ato_smooth'], gradients['no_ato_smooth'],
                                  where=gradients['ato_smooth'] >= gradients['no_ato_smooth'], color='red',
                                  interpolate=True)
        area = 0
        for path in polycol.get_paths():
            polygon = Polygon(path.vertices)
            area = area + polygon.area
        areas.loc[[gene], 'area'] = area
    fig.clear()

    matrix = distance_matrix(areas.values)
    distances = squareform(matrix)
    linkage_matrix = linkage(distances, "single")
    ax = fig.add_subplot(111)
    dendrogram(linkage_matrix, labels=genes, ax=ax, leaf_rotation=90, color_threshold=0.17)
    return fig


def fig_93ea(data, columns=5):
    genes = genes_sorted(data)
    rows = math.ceil(len(genes) / columns)
    fig = plt.figure(figsize=(15, rows * 3))
    gs = gridspec.GridSpec(rows, columns)

    for index, gene in enumerate(genes):
        row = math.ceil((index + 1) / columns)
        gene_data = input[input['Gene'] == gene]
        samples = gene_data['Sample'].unique()
        ax = fig.add_subplot(gs[index])
        profiles = []
        styles = []
        for sample in samples:
            profiles.append(y_profile(gene_data[gene_data['Sample'] == sample], 'Venus').mean())
            styles.append({'label': sample})
        plot_profiles(ax, profiles, styles, e_series())
        ax.text(0.025, 0.95, gene, horizontalalignment='left', verticalalignment='top', fontsize=24,
                transform=ax.transAxes)
        ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                       left=True, right=False, labelleft=(index % 5 == 0), labelright=False)
        ax.tick_params(axis='y', which='minor', left=False, right=False, labelleft=False, labelright=False)
    return fig


def fig_d3a8(data, columns=5):
    genes = genes_sorted(data)
    cells = data[(data['cy'] > -10) & (data['cy'] < 10)]
    background = data[(data['cy'] >=-10) & (data['cy'] <= -5)]
    ato_cells = cells[(cells['mCherry'] > background['mCherry'].quantile(0.90))]
    no_ato_cells = cells[(cells['mCherry'] < background['mCherry'].quantile(0.50))]
    rows = math.ceil(len(genes) / columns)
    fig = plt.figure(figsize=(15, rows * 3))
    gs = gridspec.GridSpec(rows, columns)
    ato = y_profile(cells[cells['Gene'].isin(clean)], 'mCherry')

    for index, gene in enumerate(genes):
        row = math.ceil((index + 1) / columns)
        ato_gene_cells = ato_cells[ato_cells['Gene'] == gene]
        no_ato_gene_cells = no_ato_cells[no_ato_cells['Gene'] == gene]
        profiles = pd.DataFrame()
        profiles['ato'] = y_profile(ato_gene_cells, 'Venus').mean()
        profiles['no_ato'] = y_profile(no_ato_gene_cells, 'Venus').mean()
        profiles = profiles.dropna()

        gradients = pd.DataFrame(index=profiles.index)
        gradients['ato'] = np.gradient(savgol_filter(profiles['ato'].values, 9, 3, mode='nearest'))
        gradients['no_ato'] = np.gradient(savgol_filter(profiles['no_ato'].values, 9, 3, mode='nearest'))

        ax = fig.add_subplot(gs[index])
        profiles = [gradients['ato'], gradients['no_ato'], ato.mean()]
        styles = [{'label': 'Target gradient (ato+)'}, {'label': 'Target gradient (ato-)'}, {'label': 'Ato protein'}]
        plot_profiles(ax, profiles, styles, e_series())

        gradients['ato_smooth'] = savgol_filter(gradients['ato'].values, 9, 3, mode='nearest')
        gradients['no_ato_smooth'] = savgol_filter(gradients['no_ato'].values, 9, 3, mode='nearest')
        ax.fill_between(gradients.index, gradients['ato_smooth'], gradients['no_ato_smooth'],
                        where=gradients['ato_smooth'] >= gradients['no_ato_smooth'], color='red', interpolate=True)
        ax.set_xlim(-10, 10)
        ax.set_yscale('linear')
        ax.set_ylim(-0.2, 1)
        ax.set_yticks(np.linspace(-0.2, 1, 7))
        ax.text(0.025, 0.95, gene, horizontalalignment='left', verticalalignment='top', fontsize=24,
                transform=ax.transAxes)
        ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                       left=True, right=False, labelleft=(index % 5 == 0), labelright=False)
        ax.tick_params(axis='y', which='minor', left=False, right=False, labelleft=False, labelright=False)
        handles, labels = ax.get_legend_handles_labels()
        ax.yaxis.set_major_formatter(major_formatter_log)

    ax = fig.add_subplot(gs[len(genes):])
    ax.set_axis_off()
    ax.legend(handles, labels, ncol=1, loc='center', frameon=False)

    return fig


def fig_ceb2(data, columns=5):

    def pd_distance(d1, d2, factor=1, normalize=False):
        s1 = d1 / d1.mean() if normalize else d1
        s2 = d2 / d2.mean() if normalize else d2

        distances = s1.subtract(s2).abs().dropna()
        indices = distances.index.values
        factors = np.power(factor, np.abs(indices))
        distance = np.sum(distances.values * factors) / len(indices)

        return distance

    def distance_matrix(data):
        matrix = np.full((len(data), len(data)), float('inf'))

        for i in range(0, len(data)):
            a = data[i]
            for j in range(0, i + 1):
                b = data[j]
                matrix[i, j] = pd_distance(a, b)

        for j in range(0, len(data)):
            for i in range(0, j + 1):
                matrix[i, j] = matrix[j, i]

        return matrix

    genes = genes_sorted(data)
    cells = data[(data['cy'] > -10) & (data['cy'] < 10)]
    background = data[(data['cy'] >= -10) & (data['cy'] <= -5)]
    ato_cells = cells[(cells['mCherry'] > background['mCherry'].quantile(0.90))]
    no_ato_cells = cells[(cells['mCherry'] < background['mCherry'].quantile(0.50))]
    gene_profiles = []

    # rows = math.ceil(len(genes) / columns)
    # fig = plt.figure(figsize=(15, rows * 3))
    # gs = gridspec.GridSpec(rows, columns)
    for index, gene in enumerate(genes):
        ato_gene_cells = ato_cells[ato_cells['Gene'] == gene]
        no_ato_gene_cells = no_ato_cells[no_ato_cells['Gene'] == gene]
        profiles = pd.DataFrame()
        profiles['ato'] = y_profile(ato_gene_cells, 'Venus').mean()
        profiles['no_ato'] = y_profile(no_ato_gene_cells, 'Venus').mean()
        profiles = profiles.dropna()

        gradients = pd.DataFrame(index=profiles.index)
        gradients['ato'] = np.gradient(savgol_filter(profiles['ato'].values, 9, 3, mode='nearest'))
        gradients['no_ato'] = np.gradient(savgol_filter(profiles['no_ato'].values, 9, 3, mode='nearest'))

        gradients['ato_smooth'] = savgol_filter(gradients['ato'].values, 9, 3, mode='nearest')
        gradients['no_ato_smooth'] = savgol_filter(gradients['no_ato'].values, 9, 3, mode='nearest')
        gradients['diff'] = gradients['ato_smooth'] - gradients['no_ato_smooth']
        gradients.loc[gradients['diff'] < 0, 'diff'] = 0
        gene_profiles.append(gradients['diff'])

        # row = math.ceil((index + 1) / columns)
        # ax = fig.add_subplot(gs[index])
        # ax.plot(gradients['diff'].index, gradients['diff'].values)
        # ax.set_xlim(-10, 10)
        # ax.set_yscale('linear')
        # ax.set_ylim(0, 0.6)
        # ax.text(0.025, 0.95, gene, horizontalalignment='left', verticalalignment='top', fontsize=24,
        #         transform=ax.transAxes)
        # ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
        #                left=True, right=False, labelleft=(index % 5 == 0), labelright=False)
        # ax.tick_params(axis='y', which='minor', left=False, right=False, labelleft=False, labelright=False)
    # fig.show()

    matrix = distance_matrix(gene_profiles)
    distances = squareform(matrix)
    linkage_matrix = linkage(distances, "single")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dendrogram(linkage_matrix, labels=genes, ax=ax, leaf_rotation=90)
    return fig


# fig = fig_3d51(input)
# fig.show()
# if args.outdir:
#     fig.savefig(os.path.join(args.outdir, 'figure_3d51.png'))
#
# fig = fig_79eb(input, 2)
# fig.show()
# if args.outdir:
#     fig.savefig(os.path.join(args.outdir, 'figure_79eb.png'))
#
# fig = fig_32b7(input)
# fig.show()
# if args.outdir:
#     fig.savefig(os.path.join(args.outdir, 'figure_32b7.png'))
#
# fig = fig_93ea(input)
# fig.show()
# if args.outdir:
#     fig.savefig(os.path.join(args.outdir, 'figure_93ea.png'))
#
#
# fig = fig_01a8(input)
# fig.show()
# if args.outdir:
#     fig.savefig(os.path.join(args.outdir, 'figure_01a8.png'))
#
# fig = fig_d3a8(input)
# fig.show()
# if args.outdir:
#     fig.savefig(os.path.join(args.outdir, 'figure_d3a8.png'))

fig = fig_ceb2(input)
fig.show()
if args.outdir:
     fig.savefig(os.path.join(args.outdir, 'figure_ceb2.png'))
