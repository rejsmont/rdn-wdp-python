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
import os
from matplotlib import colors
from scipy import stats
from scipy.signal import savgol_filter
from scipy.cluster.hierarchy import dendrogram as dng, linkage
from data import DiscData
from clustering import Clustering, ClusteredData
from chip import ChIP
from figures_ng import Figure
from images import Qimage, Thumbnails, Furrows
from data import OriginalData
from stats import CellStats

d = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/samples'
f = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/processing/samples_complete.csv'

class Figure_1:

    SAMPLE = 'K21OU5'
    SLICE = 40
    BONDS = [(0, 32), (0, 4000), (0, 16)]

    def __init__(self, d, f):
        self.dir = d
        self.file = f

    def plot(self):
        fig = plt.figure(figsize=(10, 8))

        ax = plt.Axes(fig, [0.05, 0.65, 0.425, 0.3])
        fig.add_axes(ax)
        h5 = Qimage(self.dir, self.SAMPLE)
        h5.plot_img(self.SLICE, ax=ax, bonds=self.BONDS)
        h5.scalebar(ax, linewidth=3)
        h5.label(ax, 'a', fontsize=18)
        ax = plt.Axes(fig, [0.525, 0.65, 0.425, 0.3])
        fig.add_axes(ax)
        h5.plot_simg(self.SLICE, ax=ax, bonds=self.BONDS, reference='auto')
        h5.scalebar(ax, linewidth=3)
        h5.label(ax, 'b', fontsize=18)

        print("Loading thumbs...")
        thumbs = Thumbnails(self.dir + '/thumbs', self.SAMPLE)
        ax = plt.Axes(fig, [0.1, 0.40, 0.3, 0.2])
        fig.add_axes(ax)
        thumbs.plot_normalized(ax)
        thumbs.label(ax, 'c', fontsize=18)
        ax = plt.Axes(fig, [0.1, 0.05, 0.3, 0.3])
        fig.add_axes(ax)
        thumbs.plot_aligned(ax)
        thumbs.label(ax, 'd', fontsize=18)


        print("Loading cells...")
        vstats = CellStats(OriginalData(self.file))
        print("Plotting...")

        ax = plt.Axes(fig, [0.5125, 0.425, 0.1625, 0.175])
        fig.add_axes(ax)
        vstats.size_hist(ax)
        vstats.label(ax, 'e', fontsize=18)

        ax = plt.Axes(fig, [0.7875, 0.425, 0.1625, 0.175])
        fig.add_axes(ax)
        vstats.sample_size_hist(ax)
        vstats.label(ax, 'f', fontsize=18)

        ax = plt.Axes(fig, [0.5125, 0.1125, 0.1625, 0.175])
        fig.add_axes(ax)
        vstats.sample_count_hist(ax)
        vstats.label(ax, 'g', fontsize=18)

        ax = plt.Axes(fig, [0.7875, 0.1125, 0.1625, 0.175])
        fig.add_axes(ax)
        vstats.mf_ato_hist(ax)
        vstats.label(ax, 'h', fontsize=18)

        fig.show()


if __name__ == "__main__":
    fig_1 = Figure_1(d, f)
    fig_1.plot()


