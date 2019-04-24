#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import h5py
import yaml
import os
from PIL import Image


dir = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/samples'
file = '58062_disc_1_K21OU5.h5'


class Qimage:

    CHANNELS = ['scaled/mCherry', 'scaled/DAPI', 'scaled/Venus']

    def __init__(self, d, sample):
        for f in os.listdir(d):
            if sample in f and f.endswith('.h5'):
                self.h5 = h5py.File(os.path.join(d, f), 'r')
                with open(os.path.join(d, f.replace(".h5", ".yml"))) as yml:
                    self.meta = yaml.load(yml, Loader=yaml.FullLoader)
                    yml.close()
                self.cells = pd.read_csv(os.path.join(d, f.replace(".h5", ".csv")), index_col=0)

    def hierarchy(self):
        self.h5.visit(print)

    def image(self, dataset, img_slice=None):
        ds = self.h5[dataset]
        if img_slice is None:
            return ds
        else:
            return ds[img_slice, :, :]

    def scalebar(self, ax, size=30, reference=None, **kwargs):
        import matplotlib.lines as ml
        if reference is None:
            reference = self.CHANNELS[0]
        attrs = self.h5[reference].attrs
        shape = self.h5[reference].shape
        z, x, y = attrs['element_size_um']
        length = size / x
        xmax = 0.95 * shape[2]
        ymin = 0.9 * shape[1]
        ymax = 0.9 * shape[1]
        xmin = xmax - length
        l = ml.Line2D([xmin, xmax], [ymin, ymax], color='white', **kwargs)
        ax.add_line(l)

    @staticmethod
    def label(ax, label='a', **kwargs):
        ax.text(0.025, 0.95, label, color='white', transform=ax.transAxes, va='top', **kwargs)

    @staticmethod
    def format_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    def plot_img(self, img_slice, bonds=None, ax=None, datasets=None):

        if datasets is None:
            datasets = self.CHANNELS

        if isinstance(datasets, list):
            imgs = []
            for i, ds in enumerate(datasets):
                if bonds is not None:
                    min, max = bonds[i]
                    imgs.append(self.normalize(self.image(ds, img_slice), min, max))
                else:
                    imgs.append(self.normalize(self.image(ds, img_slice)))
            img = np.stack(imgs, axis=-1)
        else:
            img = self.image(datasets, img_slice)

        ax.imshow(img)
        self.format_ax(ax)

    def plot_simg(self, img_slice, bonds=None, ax=None, reference=None):

        import matplotlib.patches as mp

        def extract(cell, img_slice, bonds=None):
            x = cell['cx']
            y = cell['cy']
            z = cell['cz']
            v = cell['Volume']
            if bonds is None:
                rgb = cell['Mean 1'], cell['Mean 0'], cell['Mean 2']
            else:
                rgb = self.normalize(cell['Mean 1'], *bonds[0]), \
                      self.normalize(cell['Mean 0'], *bonds[1]), \
                      self.normalize(cell['Mean 2'], *bonds[2])

            cr = round(math.pow((v / math.pi) * (3/4), 1/3))
            r = round(math.sqrt(math.pow(cr, 2) - math.pow(img_slice - z, 2)))

            return x, y, z, r, rgb

        maxx = maxy = 0

        for i, cell in self.cells.iterrows():
            try:
                x, y, z, r, rgb = extract(cell, img_slice, bonds)
                if x > maxx:
                    maxx = x
                if y > maxy:
                    maxy = y
                c = mp.Circle((x, y), r, color=rgb)
                ax.add_patch(c)
            except ValueError:
                continue

        if reference is not None:
            if reference == 'auto':
                reference = self.CHANNELS[0]
            maxx = self.h5[reference].shape[2]
            maxy = self.h5[reference].shape[1]

        ax.set_xlim(0, maxx)
        ax.set_ylim(maxy, 0)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        self.format_ax(ax)

    @staticmethod
    def normalize(a, min, max):
        v = (a - min) / (max - min)
        v = np.clip(v, 0, 1)
        return v


class Thumbnails:

    NAMES = ['aligned', 'furrow_line', 'furrow_manual', 'hessian', 'labels', 'mask', 'normalized', 'raw', 'thresholded']

    def __init__(self, d, sample):
        self.sample = sample
        self.images = {}
        for f in os.listdir(d):
            if sample in f:
                for name in self.NAMES:
                    if name in f:
                        self.images[name] = Image.open(d + '/' + f)

    def furrow_line(self):
        return np.array(self.images['furrow_line']).argmax(axis=0)

    def plot_normalized(self, ax, furrow=True):
        ax.imshow(self.images['normalized'])
        if furrow:
            f_line = self.furrow_line()
            ax.plot(np.arange(0, len(f_line)), f_line, color='white', linewidth=3, linestyle='dotted')

    def plot_aligned(self, ax, furrow=True):
        f_shift = -np.max(self.furrow_line())
        f_line = self.furrow_line()
        extent = [-0.5, self.images['aligned'].width - 0.5,
                  self.images['aligned'].height + f_shift - 0.5, f_shift - 0.5]
        ax.imshow(self.images['aligned'], extent=extent)
        if furrow:
            ax.plot(np.arange(0, len(f_line)), np.zeros(f_line.shape),
                    color='white', linewidth=3, linestyle='dotted')

    @staticmethod
    def label(ax, label='a', **kwargs):
        ax.text(-0.15, 1.25, label, color='black', transform=ax.transAxes, va='top', **kwargs)


class Furrows:

    def __init__(self, d):
        self.furrows = {}
        for f in os.listdir(d):
            if str(f).endswith('furrow_line.tif'):
                sample = str(f).split('_')[-4]
                image = Image.open(d + '/' + f)
                furrow = np.array(image).argmax(axis=0)
                self.furrows[sample] = furrow

    def in_furrow(self, cell):
        return cell['Sample'] in self.furrows.keys() and \
               round(cell['cy_norm']) == self.furrows[cell['Sample']][round(cell['cx_norm'])]


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    #
    # h5 = Qimage(dir + '/' + file)
    # h5.hierarchy()
    #
    # sl = 40
    # b = [(0, 32), (0, 4000), (0, 16)]
    #
    # fig, ax = plt.subplots(1, 2)
    # h5.plot_img(sl, ax=ax[0], bonds=b)
    # h5.scalebar(ax[0], linewidth=3)
    # h5.label(ax[0], 'a', fontsize=18)
    # h5.plot_simg(sl, ax=ax[1], bonds=b, reference='auto')
    # h5.scalebar(ax[1], linewidth=3)
    # h5.label(ax[1], 'b', fontsize=18)
    # fig.show()

    thumbs = Thumbnails(dir + '/thumbs', 'K21OU5')

    fig, ax = plt.subplots(2, 1, figsize=(5, 7))
    thumbs.plot_normalized(ax[0])
    thumbs.plot_aligned(ax[1])
    fig.show()
