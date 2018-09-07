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
# gxmin = round(input['cx'].min())
# gxmax = round(input['cx'].max())
# gymin = round(input['cy'].min())
# gymax = round(input['cy'].max())

cherry_min = input['mCherry'].min()
cherry_max = input['mCherry'].mean() * 5

dapi_min = input['DAPI'].min()
dapi_max = input['DAPI'].mean() * 2

venus_min = input['Venus'].min()
venus_max = input['Venus'].mean()

vmin = 0.1
vmax = 15


def plot_summary():
    target_mean = input.groupby(['Gene', round(input['cy'])])['Venus'].mean()
    target_max = input.groupby(['Gene', round(input['cy'])])['Venus'].quantile(0.99)
    ato = input.groupby([round(input['cy'])])['mCherry'].mean()

    genes = target_mean.index.get_level_values(0).unique()
    fig = plt.figure(figsize=(30, 30))
    count = len(genes)
    sx = round(math.sqrt(count))
    sy = math.ceil(count / sx)
    sn = 1
    for gene in genes:
        ax = fig.add_subplot(sx, sy, sn)
        ax.set_title(gene + ' expression profile')
        div = make_axes_locatable(ax)
        x = target_mean[gene].index
        y = savgol_filter(target_mean[gene].values, 11, 3)
        ax.plot(x, y, label='Target mean')
        x = target_max[gene].index
        y = savgol_filter(target_max[gene].values, 11, 3)
        ax.plot(x, y, label='Target Q99')
        x = ato.index
        y = savgol_filter(ato.values, 11, 3)
        ax.plot(x, y, label='Ato protein')
        ax.set_xlim(gymin, gymax)
        ax.set_ylim(0, 16)
        sn = sn + 1
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    plt.show()

def plot_sample(sample):
    print(sample)
    data = input[input['Sample'] == sample]
    plot_data(data, sample)


def plot_target(target):
    print(target)
    data = input[input['Gene'] == target]
    plot_data(data)


def plot_image(ax, data, channel, projection='mean', title=None, norm=None, cmap=None):

    xmin = round(data['cx'].min())
    xmax = round(data['cx'].max())
    ymin = round(data['cy'].min())
    ymax = round(data['cy'].max())

    if title:
        ax.set_title(title)

    img = ax.imshow(disc_matrix(data, channel, method=projection),
                    extent=[xmin, xmax, ymax, ymin], norm=norm, cmap=cmap, aspect='auto')

    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xlim(gxmin, gxmax)
    ax.set_ylim(gymax, gymin)
    ax.xaxis.tick_top()

    return img, ax


def plot_data(data, sample=None):

    target = data['Gene'].unique()[0]
    cherry = disc_matrix(data, 'mCherry', method='mean')
    venus = disc_matrix(data, 'Venus', method='mean')
    venmax = disc_matrix(data, 'Venus', method='max')

    xmin = round(data['cx'].min())
    xmax = round(data['cx'].max())
    ymin = round(data['cy'].min())
    ymax = round(data['cy'].max())

    image = np.stack((
        display_normalize(cherry, vmin, 1.5, log=False),
        display_normalize(venus, vmin, 1.5, log=False),
        display_normalize(venmax, vmin, vmax, log=True)), axis=2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)

    if sample:
        ax.set_title(target + ' expression (' + sample + ')')
        filename = target + '_' + sample + '.png'
    else:
        ax.set_title(target + ' expression')
        filename = target + '.png'

    img = ax.imshow(image, extent=[xmin, xmax, ymax, ymin])

    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xlim(gxmin, gxmax)
    ax.set_ylim(gymax, gymin)

    if args.dir:
        plt.savefig(os.path.join(dir, filename))
    else:
        plt.savefig(filename)


def genes_sorted(data):
    before = data[data['cy'] < 0].groupby(['Gene'])['Venus'].quantile(0.99)
    after = data[(data['cy'] > 0) & (data['cy'] < 20)].groupby(['Gene'])['Venus'].quantile(0.99)
    ratio = after / before
    return ratio.sort_values(ascending=False).index.tolist()


@ticker.FuncFormatter
def fig_3c_major_formatter(x, pos):
    return "%g" % (round(x * 10) / 10)


def fig_3_row(fig, gs, gene, row, rows, ticks):
    pos = (row - 1) * 5
    data = input[input['Gene'] == gene]

    ax = fig.add_subplot(gs[pos])
    ax.set_axis_off()
    ax.text(0.5, 0.5, gene, horizontalalignment='center', verticalalignment='center', fontsize=24, rotation=90)

    norm = colors.LogNorm(vmin=0.1, vmax=30)
    img0, ax = plot_image(fig.add_subplot(gs[pos + 1]), data, 'Venus', projection='mean', title=False, norm=norm, cmap='plasma')
    ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                   left=True, right=False, labelleft=True, labelright=False)

    img1, ax = plot_image(fig.add_subplot(gs[pos + 2]), data, 'Venus', projection='max', title=False, norm=norm, cmap='plasma')
    ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                   left=True, right=False, labelleft=False, labelright=False)

    norm = colors.LogNorm(vmin=0.1, vmax=25)
    img2, ax = plot_image(fig.add_subplot(gs[pos + 3]), data, 'ext_Venus', projection='max', title=False, norm=norm, cmap='viridis')
    ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                   left=True, right=False, labelleft=False, labelright=False)

    ax = fig.add_subplot(gs[pos + 4])
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
    ax.tick_params(bottom=True, top=True, labelbottom=(row == rows), labeltop=(row == 1),
                   left=False, right=True, labelleft=False, labelright=True)

    handles, labels = ax.get_legend_handles_labels()
    return img0, img1, img2, handles, labels


def fig_3(data):
    genes = genes_sorted(data)
    #genes = ['ato', 'sens', 'Brd']
    e_series = [0.1, 0.22, 0.47, 1, 2.2, 4.7, 10, 22]
    rows = len(genes)
    fig = plt.figure(figsize=(32, math.ceil(rows / 2) * 2.65))
    width_ratios = [1]
    width_ratios.extend([16] * 4)
    width_ratios.extend([1])
    width_ratios.extend([16] * 4)
    height_ratios = ([8] * math.ceil(rows / 2))
    height_ratios.extend([1])
    gs = gridspec.GridSpec(math.ceil(rows / 2) + 1, 10, width_ratios=width_ratios, height_ratios=height_ratios)
    img0, img1, img2, handles, labels = None, None, None, None, None
    for index, gene in enumerate(genes):
        print(index, gene)
        img0, img1, img2, handles, labels = fig_3_row(fig, gs, gene, index + 1, rows, e_series)

    cax = fig.add_subplot(gs[rows * 5 + 1])
    cbar = fig.colorbar(img0, cax=cax, orientation='horizontal', ticks=e_series,
                        format=fig_3c_major_formatter, label='Mean target expression')
    cbar.ax.margins(1.2, 0.2)

    cax = fig.add_subplot(gs[rows * 5 + 2])
    cbar = fig.colorbar(img1, cax=cax, orientation='horizontal', ticks=e_series,
                        format=fig_3c_major_formatter, label='Max target expression')
    cbar.ax.margins(0.2, 0.2)

    cax = fig.add_subplot(gs[rows * 5 + 3])
    cbar = fig.colorbar(img2, cax=cax, orientation='horizontal', ticks=e_series,
                        format=fig_3c_major_formatter, label='Max target eccentricity')
    cbar.ax.margins(0.2, 0.2)

    cax = fig.add_subplot(gs[rows * 5 + 4])
    cax.set_axis_off()
    cax.legend(handles, labels, ncol=2, loc='center', frameon=False)
    plt.show()


fig_3(input)
