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
from scipy.signal import savgol_filter


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
    y = savgol_filter(data.values, 5, 3, mode='nearest')
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
            profiles = [target.mean(), target.quantile(0.99), ato.mean()]
            styles = [{'label': 'Target mean'}, {'label': 'Target Q99'}, {'label': 'Ato protein'}]
            plot_profiles(ax, profiles, styles, ticks)
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
        cbar = fig.colorbar(img, cax=cax, orientation='horizontal', ticks=e_series(),
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


fig = fig_3d51(input)
fig.show()
if args.outdir:
    fig.savefig(os.path.join(args.outdir, 'figure_3d51.png'))

fig = fig_79eb(input, 2)
fig.show()
if args.outdir:
    fig.savefig(os.path.join(args.outdir, 'figure_79eb.png'))
