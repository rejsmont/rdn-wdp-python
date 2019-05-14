#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import h5py
import yaml
import os
from PIL import Image
from images import Furrows
import matplotlib.ticker as tkr

dir = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/samples'
file = '58062_disc_1_K21OU5.h5'


class CellStats:

    VOL_SCALING = (0.15 * 0.151 * 0.151)

    def __init__(self, data):
        self.data = data

    @staticmethod
    def thousand(x, pos):
        s = "{:.0f}".format(x / 1000)
        return s

    def size_hist(self, ax):
        vol = self.data.cells['Volume'] * self.VOL_SCALING
        ax.hist(vol, bins=100)
        ax.axvline(vol.mean(), color='black')
        ax.axvline(vol.mean() * 0.5, linestyle='dashed', color='black')
        ax.axvline(vol.mean() * 1.5, linestyle='dashed', color='black')
        ax.set_ylabel(r'Cell count ($10^3$)')
        ax.yaxis.set_major_formatter(tkr.FuncFormatter(self.thousand))
        ax.set_xlabel(r'Nuclear volume [$\mu m^3$]')
        ax.set_xlim([0, 80])

    def sample_size_hist(self, ax):
        bins = self.data.cells.groupby(['Sample'])['Volume'].mean() * self.VOL_SCALING
        ax.hist(bins, bins=20, range=(20, 40))
        ax.axvline(self.data.cells['Volume'].mean() * self.VOL_SCALING, color='black')
        ax.set_ylabel(r'Disc count')
        ax.set_xlabel(r'Nuclear volume [$\mu m^3$]')
        ax.set_xlim([20, 40])

    def sample_count_hist(self, ax):
        bins = self.data.cells.groupby(['Sample'])['Volume'].count()
        ax.hist(bins, bins=20, range=(0, 20000))
        ax.axvline(bins.mean(), color='black')
        ax.set_ylabel(r'Disc count')
        ax.set_xlabel(r'Cell count ($10^3$)')
        ax.set_xlim([0, 20000])
        ax.xaxis.set_major_formatter(tkr.FuncFormatter(self.thousand))

    def mf_ato_hist(self, ax):
        mf = round(self.data.cells['cy']) == 0
        bins = self.data.cells[mf].groupby(['Sample'])['mCherry_orig'].mean()
        ax.hist(bins, bins=20, range=(0, 64))
        ax.axvline(self.data.cells[mf]['mCherry_orig'].mean(), color='black')
        ax.set_ylabel(r'Disc count')
        ax.set_xlabel(r'MF Ato mean [$au$]')

    @staticmethod
    def label(ax, label='a', **kwargs):
        ax.text(-0.4, 1.225, label, color='black', transform=ax.transAxes, va='top', **kwargs)
