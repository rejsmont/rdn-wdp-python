#!/usr/bin/env python3

import argparse
import logging
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


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


def display_normalize(data, min=None, max=None):
    if min is None:
        min = 0
    if max is None:
        max = (data.mean() * 2)
    if data.max() == 1:
        return (data * 255).clip(0, 255).astype('uint8')
    else:
        return (((data-min) / (max-min)) * 255).clip(0, 255).astype('uint8')


parser = argparse.ArgumentParser(description='Plot all data.')
parser.add_argument('--data', required=True)
parser.add_argument('--target')
parser.add_argument('--log')
args = parser.parse_args()

if args.log:
    logging.basicConfig(level=args.log.upper())
    logging.getLogger('PIL.Image').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)

input = pd.read_csv(args.data)
input.loc[input['cy'] > 9, 'mCherry'] = input['mCherry'].min()

targets = input['Gene'].unique()

print(targets)

gxmin = round(input['cx'].min())
#gxmax = round(input['cx'].max())
gxmax = 80
#gymin = round(input['cy'].min())
gymin = -10
gymax = round(input['cy'].max())

cherry_min = input['mCherry'].min()
cherry_max = input['mCherry'].mean() * 5

dapi_min = input['DAPI'].min()
dapi_max = input['DAPI'].mean() * 2

venus_min = input['Venus'].min()
venus_max = input['Venus'].mean()


def plot_target(input, target):

    data = input[input['Gene'] == target]

    cherry = disc_matrix(data, 'mCherry', method='mean')
    dapi = disc_matrix(data, 'DAPI', method='mean')
    venus = disc_matrix(data, 'Venus', method='mean')

    xmin = round(data['cx'].min())
    xmax = round(data['cx'].max())
    ymin = round(data['cy'].min())
    ymax = round(data['cy'].max())

    image = np.stack((
        display_normalize(cherry, cherry_min, cherry_max),
        display_normalize(venus, venus_min, venus_max),
        np.zeros(venus.shape).astype('uint8')), axis=2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(target + ' expression')
    plt.imshow(image, extent=[xmin, xmax, ymax, ymin])
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xlim(gxmin, gxmax)
    ax.set_ylim(gymax, gymin)
    plt.show()


args.target = None

if args.target:
    plot_target(input, args.target)
else:
    for target in targets:
        plot_target(input, target)
