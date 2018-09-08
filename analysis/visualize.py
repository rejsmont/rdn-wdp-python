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
parser.add_argument('--target')
parser.add_argument('--targets', action='store_true')
parser.add_argument('--sample')
parser.add_argument('--samples', action='store_true')
parser.add_argument('--summary', action='store_true')
parser.add_argument('--log')
parser.add_argument('--dir')
args = parser.parse_args()

if args.log:
    logging.basicConfig(level=args.log.upper())
    logging.getLogger('PIL.Image').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)

input = pd.read_csv(args.data)
input.loc[input['cy'] > 9, 'mCherry'] = input['mCherry'].min()

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


def plot_profiles(ax, data, ticks):
    target_mean = data.groupby([round(data['cy'])])['Venus'].mean()
    x = target_mean.index
    y = savgol_filter(target_mean.values, 11, 3)
    ax.plot(x, y, label='Target mean')
    target_max = data.groupby([round(data['cy'])])['Venus'].quantile(0.99)
    x = target_max.index
    y = savgol_filter(target_max.values, 11, 3)
    ax.plot(x, y, label='Target Q99')
    ato = input.groupby([round(input['cy'])])['mCherry'].mean()
    x = ato.index
    y = savgol_filter(ato.values, 11, 3)
    ax.plot(x, y, label='Ato protein')
    ax.set_xlim(gymin, gymax)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 30)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(fig_3c_major_formatter)
    return ax


def plot_legends(fig, gs, rows, columns, img0, img1, img2, handles, labels, ticks):
    origin = rows * columns * 5
    cax = fig.add_subplot(gs[origin + 1])
    cbar = fig.colorbar(img0, cax=cax, orientation='horizontal', ticks=ticks,
                        format=fig_3c_major_formatter, label='Mean target expression')
    cbar.ax.margins(1.2, 0.2)

    cax = fig.add_subplot(gs[origin + 2])
    cbar = fig.colorbar(img1, cax=cax, orientation='horizontal', ticks=ticks,
                        format=fig_3c_major_formatter, label='Max target expression')
    cbar.ax.margins(0.2, 0.2)

    cax = fig.add_subplot(gs[origin + 3])
    cbar = fig.colorbar(img2, cax=cax, orientation='horizontal', ticks=ticks,
                        format=fig_3c_major_formatter, label='Max target eccentricity')
    cbar.ax.margins(0.2, 0.2)

    cax = fig.add_subplot(gs[origin + 4])
    cax.set_axis_off()
    cax.legend(handles, labels, ncol=2, loc='center', frameon=False)


def genes_sorted(data):
    before = data[data['cy'] < 0].groupby(['Gene'])['Venus'].quantile(0.99)
    after = data[(data['cy'] > 0) & (data['cy'] < 20)].groupby(['Gene'])['Venus'].quantile(0.99)
    ratio = after / before
    return ratio.sort_values(ascending=False).index.tolist()


@ticker.FuncFormatter
def fig_3c_major_formatter(x, pos):
    return "%g" % (round(x * 10) / 10)


def fig_3_row(fig, gs, gene, index, rows, columns, ticks):
    row = math.ceil(index / columns)
    pos = (index - 1) * 5
    data = input[input['Gene'] == gene]

    ax = fig.add_subplot(gs[pos])
    ax.set_axis_off()
    ax.text(0.5, 0.5, gene, horizontalalignment='center', verticalalignment='center', fontsize=24, rotation=90)

    norm = colors.LogNorm(vmin=0.1, vmax=30)
    img0, ax = plot_image(fig.add_subplot(gs[pos + 1]), data, 'Venus', projection='mean', norm=norm, cmap='plasma')
    ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                   left=True, right=False, labelleft=True, labelright=False)

    img1, ax = plot_image(fig.add_subplot(gs[pos + 2]), data, 'Venus', projection='max', norm=norm, cmap='plasma')
    ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                   left=True, right=False, labelleft=False, labelright=False)

    norm = colors.LogNorm(vmin=0.1, vmax=25)
    img2, ax = plot_image(fig.add_subplot(gs[pos + 3]), data, 'ext_Venus', projection='max', norm=norm, cmap='viridis')
    ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                   left=True, right=False, labelleft=False, labelright=False)

    ax = plot_profiles(fig.add_subplot(gs[pos + 4]), data, ticks)
    ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                   left=False, right=True, labelleft=False, labelright=True)
    handles, labels = ax.get_legend_handles_labels()

    return img0, img1, img2, handles, labels


def fig_3(data, columns):
    genes = genes_sorted(data)
    e_series = [0.1, 0.22, 0.47, 1, 2.2, 4.7, 10, 22]
    rows = math.ceil(len(genes) / columns)
    fig = plt.figure(figsize=(32, rows * 2.65))
    width_ratios = [item for sub in [[1], [16] * 4] * columns for item in sub]
    height_ratios = [item for sub in [[8] * rows, [1]] for item in sub]
    gs = gridspec.GridSpec(rows + 1, 5 * columns, width_ratios=width_ratios, height_ratios=height_ratios)
    legend_data = ()
    for index, gene in enumerate(genes):
        print(index, gene)
        legend_data = fig_3_row(fig, gs, gene, index + 1, rows, columns, e_series)
    plot_legends(fig, gs, rows, columns, *legend_data, e_series)
    plt.show()


fig_3(input, 2)
