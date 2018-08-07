#!/usr/bin/env python3

import argparse
import logging
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import cpu_count, Pool


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
parser.add_argument('--log-scale', action='store_true')
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

vmin = 0.01
vmax = 1.5

log = args.log_scale


def plot_sample(sample):
    print(sample)
    data = input[input['Sample'] == sample]
    plot_data(data, sample)


def plot_target(target):
    print(target)
    data = input[input['Gene'] == target]
    plot_data(data)


def plot_data(data, sample=None):
    target = data['Gene'].unique()[0]
    cherry = disc_matrix(data, 'mCherry', method='mean')
    venus = disc_matrix(data, 'Venus', method='mean')
    # dapi = disc_matrix(data, 'DAPI', method='mean')

    xmin = round(data['cx'].min())
    xmax = round(data['cx'].max())
    ymin = round(data['cy'].min())
    ymax = round(data['cy'].max())

    image = np.stack((
        display_normalize(cherry, vmin, vmax, log=False),
        display_normalize(venus, vmin, vmax, log=True),
        display_normalize(np.zeros(venus.shape), vmin, vmax, log=True)), axis=2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)

    if sample:
        ax.set_title(target + ' expression (' + sample + ')')
        filename = target + '_' + sample + '.png'
    else:
        ax.set_title(target + ' expression')
        filename = target + '.png'

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cdict = {'red':   ((0.0, 0.0, 0.0),
                       (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
             'blue':  ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0))}
    cmap = colors.LinearSegmentedColormap('red', cdict)
    img = ax.imshow(image, extent=[xmin, xmax, ymax, ymin], norm=norm, cmap=cmap)
    cax = div.append_axes("right", size=0.3, pad=0.1)
    plt.colorbar(img, cax=cax)

    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0)),
             'green': ((0.0, 0.0, 0.0),
                       (1.0, 1.0, 1.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}
    cmap = colors.LinearSegmentedColormap('green', cdict)
    img = ax.imshow(image, extent=[xmin, xmax, ymax, ymin], norm=norm, cmap=cmap)
    cax = div.append_axes("right", size=0.3, pad=0.4)
    plt.colorbar(img, cax=cax)

    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xlim(gxmin, gxmax)
    ax.set_ylim(gymax, gymin)

    if args.dir:
        plt.savefig(os.path.join(dir, filename))
    else:
        plt.savefig(filename)


if args.target:
    plot_target(args.target)
elif args.sample:
    plot_sample(args.sample)
else:
    if args.samples:
        samples = input['Sample'].unique()
        if __name__ == '__main__':
            with Pool(cpu_count()) as p:
                p.map(plot_sample, samples)
        # for sample in samples:
        #     print(sample)
        #     plot_sample(input, sample)
    if args.targets or not args.samples:
        targets = input['Gene'].unique()
        if __name__ == '__main__':
            with Pool(cpu_count()) as p:
                p.map(plot_target, targets)
        # for target in targets:
        #     print(target)
        #     plot_target(input, target)
