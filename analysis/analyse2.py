#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import collections
import numpy as np
import pandas as pd
import math
import scipy.spatial as spa
import matplotlib
from PIL import Image
from multiprocessing import cpu_count, Pool, Process
from skimage.color import label2rgb
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import threshold_triangle
from skimage.morphology import skeletonize
from skimage.measure import label


class SampleProcessor:

    class Thumbnail:
        def __init__(self, image, mode='RGB', title='Thumbnail', fn=None):
            self.image = image
            self.mode = mode
            self.title = title
            self.fn = fn

    class Options:
        show_thumbs = 'immediately'

    def __init__(self, directory, sample, options=None, furrow=None, flip=None):
        self.directory = directory
        self.sample = sample
        self.basename = os.path.splitext(self.sample)[0]
        split = self.basename.split("_")
        self.hash = split[len(split)-1]
        self.logger = logging.getLogger(self.hash)
        self.nuclei = pd.DataFrame()
        self.images = collections.OrderedDict()
        if options:
            self.options = options
        else:
            self.options = SampleProcessor.Options()
        self.furrow = np.poly1d([0, 0])
        self.furrow_manual = furrow
        self.flip = flip
        self.nuclei_p = pd.DataFrame()

    def read_nuclei(self):
        # Assemble sample path
        input_file = os.path.join(self.directory, self.sample)
        # Read nuclei from CSV file
        nuclei = pd.read_csv(input_file)
        # Filter discs by mean volume +/- 50%
        mean_volume = nuclei['Volume'].mean()
        # Rename intensity fields
        nuclei = nuclei.rename(index=str, columns={'Mean 1': "mCherry", "Mean 2": "Venus", "Mean 0": "DAPI"})
        self.nuclei = nuclei[(nuclei['Volume'] > mean_volume * 0.5) & (nuclei['Volume'] < mean_volume * 1.5)]

    def normalize_units(self):
        # Compute mean nucleus diameter
        unit = 2 * math.pow((3 * self.nuclei['Volume'].mean()) / (4 * math.pi), 1.0 / 3.0)
        # Scale units to mean diameter
        for field in ['cx', 'cy', 'cz']:
            self.nuclei[field] = self.nuclei[field] / unit

    def count_corners(self, radius=5):
        round_cx = round(self.nuclei['cx'])
        round_cy = round(self.nuclei['cy'])
        max_x = round(self.nuclei['cx'].max() - radius)
        max_y = round(self.nuclei['cy'].max() - radius)
        top_left = self.nuclei.loc[(round_cy < radius) & (round_cx < radius), 'DAPI'].count()
        top_right = self.nuclei.loc[(round_cy < radius) & (round_cx > max_x), 'DAPI'].count()
        btm_left = self.nuclei.loc[(round_cy > max_y) & (round_cx < radius), 'DAPI'].count()
        btm_right = self.nuclei.loc[(round_cy > max_y) & (round_cx > max_x), 'DAPI'].count()
        return [top_left, top_right, btm_left, btm_right]

    def fix_rotation(self, radius=15):
        if self.flip == 'auto' or self.flip is None:
            zeroes = 4
            i = 1
            while zeroes > 1 and i <= radius:
                corners = self.count_corners(i)
                zeroes = sum(c == 0 for c in corners)
                i = i + 1
            if zeroes == 0:
                self.logger.warning("Unable to detect origin: nuclei occupy the whole field of view")
                origin = 2
            elif zeroes > 1:
                self.logger.warning("Unable to detect origin: more than one corner is empty")
                origin = 2
            else:
                origin = corners.index(min(corners))
        else:
            origin = 2
        if origin < 2 or 'y' in self.flip:
            self.logger.debug("Flipping disc vertically")
            self.nuclei['cy'] = self.nuclei['cy'].max() - self.nuclei['cy']
        if origin % 2 != 0 or 'x' in self.flip:
            self.logger.debug("Flipping disc horizontally")
            self.nuclei['cx'] = self.nuclei['cx'].max() - self.nuclei['cx']

    def disc_matrix(self, disc, field, method='max'):
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

    def display_normalize(self, data):
        if data.max() == 1:
            return (data * 255).clip(0, 255).astype('uint8')
        else:
            return ((data / (data.mean() * 2)) * 255).clip(0, 255).astype('uint8')

    def detect_ridges(self, image, sigma=3.0):
        pad = round(sigma * 3)
        padded = np.pad(image, pad, 'edge')
        hessian = hessian_matrix(padded, sigma, order="rc")
        hessian[0] = np.zeros(hessian[0].shape)
        hessian[1] = np.zeros(hessian[1].shape)
        i1, i2 = hessian_matrix_eigvals(hessian)
        i1 = i1[pad:image.shape[0] + pad, pad:image.shape[1] + pad]
        i2 = i2[pad:image.shape[0] + pad, pad:image.shape[1] + pad]
        return i1, i2

    def rms_fit(self, cx, cy, deg=10, err_mean=0.5, err_max=2.0):
        for degree in range(0, deg):
            fn = np.polyfit(cx, cy, degree)
            fy = np.polyval(fn, cx)
            error_mean = ((cy - fy) ** 2).mean()
            error_max = ((cy - fy) ** 2).max()
            if error_mean < err_mean and error_max < err_max:
                return fn
        self.logger.warning("Unable to find good polynomial fit")
        return fn

    def auto_furrow(self, labels, mask):
        top_edge = mask.shape[0] - np.argmax(np.flip(mask, axis=0), axis=0) - 1
        bottom_edge = np.argmax(mask, axis=0)
        lines = []
        for line in range(1, labels.max() + 1):
            image = (labels == line)
            lines.append((line, image[image == True].size, image.argmax(axis=0).astype('float')))
        lines = sorted(lines, key=lambda l: l[1], reverse=True)
        for line in lines:
            line_label = line[0]
            positions = line[2]
            for i in range(0, len(positions)):
                if labels[int(positions[i]), i] != line_label:
                    positions[i] = np.NaN
        for i in range(0, len(lines)):
            if lines[i] is not None:
                positions = lines[i][2]
                top_distance = (top_edge - positions)
                bottom_distance = (positions - bottom_edge)
                middle_line = bottom_edge + (top_edge - bottom_edge) / 2
                total_count = positions[~np.isnan(positions)].size
                with np.errstate(invalid='ignore'):
                    top_count = top_distance[~np.isnan(top_distance) & (top_distance < 2)].size
                    bottom_count = bottom_distance[~np.isnan(bottom_distance) & (bottom_distance < 2)].size
                    half_count = positions[positions > middle_line].size
                if bottom_count / total_count > 0.5 or top_count / total_count > 0.25 or half_count / total_count > 0.5:
                    lines[i] = None
                    continue
                for j in range(i + 1, len(lines)):
                    if lines[j] is not None:
                        collisions = 0
                        positions_i = lines[i][2]
                        positions_j = lines[j][2]
                        if positions_j.size < 5:
                            lines[j] = None
                        for k in range(0, len(positions_i)):
                            if positions_i[k] >= 0 and positions_j[k] >= 0:
                                collisions = collisions + 1
                            if collisions >= 3:
                                lines[j] = None
                                break
        positions = None
        for i in range(0, len(lines)):
            if lines[i] is not None:
                positions = lines[i][2]
                break
        if positions is not None:
            for i in range(0, positions.size):
                if np.isnan(positions[i]):
                    for j in range(1, len(lines)):
                        if lines[j] is not None:
                            if not np.isnan(lines[j][2][i]):
                                positions[i] = lines[j][2][i]
                                break
        return positions

    def manual_furrow(self):
        image = Image.open(self.furrow_manual)
        if image:
            data = np.array(image)
            return data.argmax(axis=0).astype('float')
        else:
            return None

    def find_furrow(self, use_dapi=False):
        disc = pd.DataFrame()
        disc['cx'] = self.nuclei['cx']
        disc['cy'] = self.nuclei['cy']
        disc['cz'] = self.nuclei['cz']
        disc['mCherry'] = self.nuclei['mCherry']
        cherry_m = self.disc_matrix(disc, 'mCherry', 'mean')
        disc['DAPI'] = self.nuclei['DAPI']
        dapi_m = self.disc_matrix(disc, 'DAPI', 'mean')
        mask_thr = threshold_triangle(dapi_m)
        mask = dapi_m > mask_thr
        self.thumbnail('mask', mask, title="Disc mask")
        xhe_min, che_max = self.detect_ridges(cherry_m)
        if use_dapi:
            disc['iDAPI'] = 1 / self.nuclei['DAPI']
            idapi_m = self.disc_matrix(disc, 'iDAPI', 'mean')
            dhe_min, dhe_max = self.detect_ridges(idapi_m)
            he_max = dhe_max * che_max
        else:
            he_max = np.absolute(che_max)
        self.thumbnail('hessian', he_max, title="Max of Hessian Eigenvalue (MHE)")
        threshold = threshold_triangle(he_max)
        thresholded = he_max > threshold
        self.thumbnail('thresholded', thresholded, title="Thresholded MHE (triangle)")
        skeleton = skeletonize(thresholded)
        labels = label(skeleton)
        self.thumbnail('labels', label2rgb(labels, bg_label=0), title="Detected lines")
        if self.furrow_manual:
            furrow = self.manual_furrow()
        else:
            furrow = self.auto_furrow(labels, mask)
        mask = ~np.isnan(furrow)
        indices = np.arange(0, furrow.size)
        cx = indices[mask]
        cy = furrow[mask]
        furrow_img = np.zeros(cherry_m.shape, dtype=bool)
        if len(cx) < 2:
            self.logger.warning("Unable to reliably determine furrow position")
        fn = self.rms_fit(cx, cy)
        poly = np.poly1d(fn)
        for x in range(0, furrow.size):
            if np.isnan(furrow[x]):
                furrow[x] = round(poly(x))
            furrow_img[int(furrow[x]), x] = True
        self.thumbnail('furrow_line', furrow_img, title="Furrow line")
        self.furrow = furrow

    def normalize_intensities(self):
        d_cy = round(self.nuclei['cy'])
        f_cy = self.furrow[round(self.nuclei['cx']).astype('int')]
        unit = self.nuclei.loc[d_cy == f_cy, 'mCherry'].mean()
        self.logger.debug("Normalization factor: %f", unit)
        self.nuclei['mCherry'] = self.nuclei['mCherry'] / unit
        self.nuclei['Venus'] = self.nuclei['Venus'] / unit
        self.nuclei['DAPI'] = self.nuclei['DAPI'] / unit

    def align_nuclei(self):
        self.nuclei_p['cx'] = self.nuclei['cx']
        self.nuclei_p['cy'] = self.nuclei['cy'] - self.furrow[round(self.nuclei['cx']).astype('int')]
        self.nuclei_p['cz'] = self.nuclei['cz']
        self.nuclei_p['mCherry'] = self.nuclei['mCherry']
        self.nuclei_p['Venus'] = self.nuclei['Venus']
        self.nuclei_p['DAPI'] = self.nuclei['DAPI']
        self.nuclei_p['Volume'] = self.nuclei['Volume']
        self.nuclei_p['ext_mCherry'] = np.NaN
        self.nuclei_p['ext_Venus'] = np.NaN
        self.nuclei_p['ang_max_mCherry'] = np.NaN
        self.nuclei_p['ang_max_Venus'] = np.NaN
        self.nuclei_p.reset_index()

    def compute_neighbors(self):
        KDtree = spa.cKDTree(self.nuclei_p[['cx', 'cy', 'cz']].values)
        count = len(self.nuclei_p.index)
        for index, nucleus in self.nuclei_p.iterrows():
            distances, indices = KDtree.query(nucleus[['cx', 'cy', 'cz']].values, range(2, 28), distance_upper_bound=2)
            indices = indices[indices < count]
            neighbors = self.nuclei_p.iloc[indices]
            self.nuclei_p.at[index, 'ext_mCherry'] = nucleus['mCherry'] / neighbors['mCherry'].mean()
            self.nuclei_p.at[index, 'ext_Venus'] = nucleus['Venus'] / neighbors['Venus'].mean()
            max_cherry_value = neighbors['mCherry'].max()
            max_venus_value = neighbors['Venus'].max()
            max_cherry_neighbor = neighbors.loc[neighbors['mCherry'] == max_cherry_value]
            max_venus_neighbor = neighbors.loc[neighbors['Venus'] == max_venus_value]
            if len(max_cherry_neighbor.index) == 1:
                self.nuclei_p.at[index, 'ang_max_mCherry'] = self.nuclei_angle(nucleus, max_cherry_neighbor)
            if len(max_venus_neighbor.index) == 1:
                self.nuclei_p.at[index, 'ang_max_Venus'] = self.nuclei_angle(nucleus, max_venus_neighbor)

    def thumbnail(self, name, red, green=None, blue=None, fn=None, title=None):
        if green is not None and blue is not None:
            image = np.stack((
                self.display_normalize(red),
                self.display_normalize(green),
                self.display_normalize(blue)), axis=2)
            mode = 'RGB'
        elif len(red.shape) == 3:
            image = self.display_normalize(red)
            mode = 'RGB'
        else:
            image = self.display_normalize(red)
            mode = 'L'
        self.images[name] = SampleProcessor.Thumbnail(image, mode, title, fn)
        if self.options.show_thumbs == 'immediately':
            self.show_thumbnail(name)

    def save_thumbnail(self, name, formats="tif"):
        import matplotlib.pyplot as plt
        thumbnail = self.images[name]
        if "tif" in formats:
            img = Image.fromarray(thumbnail.image, thumbnail.mode)
            path_tif = os.path.join(self.directory, self.basename + "_thumb_" + name + ".tif")
            self.logger.debug("Saving %s image to %s", name, path_tif)
            img.save(path_tif)
            img.close()
        if "png" in formats or "pdf" in formats:
            fig = plt.figure()
            self.plot_thumbnail(thumbnail, fig, 111)
            if "png" in formats:
                path_png = os.path.join(self.directory, self.basename + "_thumb_" + name + ".png")
                self.logger.debug("Saving %s plot to %s", name, path_png)
                plt.savefig(path_png)
            if "pdf" in formats:
                path_pdf = os.path.join(self.directory, self.basename + "_thumb_" + name + ".pdf")
                self.logger.debug("Saving %s plot to %s", name, path_pdf)
                plt.savefig(path_pdf)
            plt.close()

    def save_thumbnails(self, formats="pdf"):
        import matplotlib.pyplot as plt
        if "tif" in formats:
            for thumbnail in self.images:
                self.save_thumbnail(thumbnail, formats="tif")
        if "png" in formats or "pdf" in formats:
            fig = plt.figure()
            self.plot_thumbnails(fig)
            if "png" in formats:
                path_png = os.path.join(self.directory, self.basename + "_thumbs.png")
                self.logger.debug("Saving thumbnails plot to %s", path_png)
                plt.savefig(path_png)
            if "pdf" in formats:
                path_pdf = os.path.join(self.directory, self.basename + "_thumbs.pdf")
                self.logger.debug("Saving thumbnails plot to %s", path_pdf)
                plt.savefig(path_pdf)
            plt.close()

    def show_thumbnail(self, name):
        import matplotlib.pyplot as plt
        thumbnail = self.images[name]
        fig = plt.figure()
        self.plot_thumbnail(thumbnail, fig, 111)
        plt.show()

    def show_thumbnails(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        self.plot_thumbnails(fig)
        plt.show()

    def plot_thumbnail(self, thumbnail, fig, mode, mode_y=None, mode_i=None):
        import matplotlib.pyplot as plt
        if mode_y is None or mode_i is None:
            ax = fig.add_subplot(mode)
        else:
            ax = fig.add_subplot(mode, mode_y, mode_i)
        ax.set_title(thumbnail.title)
        plt.imshow(thumbnail.image)
        if thumbnail.fn is not None:
            x = np.arange(0, thumbnail.image.shape[1])
            plt.plot(x, thumbnail.fn)
        ax.set_aspect('equal')

    def plot_thumbnails(self, fig):
        count = len(self.images)
        rows = round(math.sqrt(count))
        columns = math.ceil(count / rows)
        for index, thumbnail in enumerate(self.images.values()):
            self.plot_thumbnail(thumbnail, fig, rows, columns, index + 1)

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in degrees between vectors 'v1' and 'v2'. """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    def nuclei_angle(self, n1, n2):
        """ Returns the o'clock position of a nucleus relative to another nucleus """
        v1 = [0, - n1['cy']]
        v2 = [n2['cx'].values[0] - n1['cx'], n2['cy'].values[0] - n1['cy']]

        if v1 == [0, 0] or v2 == [0, 0]:
            return np.NaN
        else:
            return self.angle_between(v1, v2)

    def run(self):
        self.read_nuclei()
        self.logger.debug("Normalizing distance units")
        self.normalize_units()
        self.logger.debug("Fixing sample orientation")
        self.fix_rotation()
        self.thumbnail('raw',
                       self.disc_matrix(self.nuclei, 'mCherry'),
                       self.disc_matrix(self.nuclei, 'DAPI'),
                       self.disc_matrix(self.nuclei, 'Venus'),
                       title="Raw disc image")
        self.logger.info("Computing furrow position")
        self.find_furrow()
        self.logger.info("Normalizing sample intensities")
        self.normalize_intensities()
        self.thumbnail('normalized',
                       self.disc_matrix(self.nuclei, 'mCherry'),
                       self.disc_matrix(self.nuclei, 'DAPI'),
                       self.disc_matrix(self.nuclei, 'Venus'),
                       self.furrow,
                       title="Normalized disc image")
        self.logger.info("Aligning nuclei to furrow")
        self.align_nuclei()
        self.thumbnail('aligned',
                       self.disc_matrix(self.nuclei_p, 'mCherry'),
                       self.disc_matrix(self.nuclei_p, 'DAPI'),
                       self.disc_matrix(self.nuclei_p, 'Venus'),
                       np.full(self.furrow.shape, -(self.nuclei_p['cy'].min())),
                       title="Normalized disc image")
        self.logger.info("Computing neighbors")
        self.compute_neighbors()
        path_csv = os.path.join(self.directory, self.basename + "_normalized.csv")
        self.logger.debug("Saving normalized dataset to %s", path_csv)
        self.nuclei_p.to_csv(path_csv)
        if self.options.show_thumbs == 'combined':
            self.show_thumbnails()
        self.save_thumbnails("tif, png")
        self.logger.info("Processing finished")

    def exception(self):
        self.logger.error("Error processing sample")


parser = argparse.ArgumentParser(description='Nuclear point cloud postprocessing.')
parser.add_argument('--dir', required=True)
parser.add_argument('--csv')
parser.add_argument('--furrow')
parser.add_argument('--no-flip')
parser.add_argument('--flip-x')
parser.add_argument('--flip-y')
parser.add_argument('--workers', type=int, default=cpu_count())
parser.add_argument('--log')
parser.add_argument('--headless', action='store_true')
parser.add_argument('--show-thumbs', choices=['never', 'immediately', 'combined'], default='combined')

args = parser.parse_args()

options = SampleProcessor.Options
options.show_thumbs = args.show_thumbs

if args.log:
    logging.basicConfig(level=args.log.upper())
    logging.getLogger('PIL.Image').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)

if args.headless:
    matplotlib.use('Agg')
    options.show_thumbs = 'never'
if args.csv:
    logging.info("Single sample processing")
    flip='auto'
    if args.no_flip:
        flip='none'
    if args.flip_x:
        flip='x'
    if args.flip_y:
        if flip == 'x':
            flip = 'xy'
        else:
            flip = 'y'
    print(flip)
    process = SampleProcessor(args.dir, args.csv, options, furrow=args.furrow, flip=flip)
    process.run()
else:
    logger = logging.getLogger('analyse')
    logger.info("Processing %s with %i workers.", args.dir, args.workers)

    samples = [f for f in os.listdir(args.dir) if
               os.path.isfile(os.path.join(args.dir, f)) and f.endswith(".csv") and not f.endswith("normalized.csv")]

    def run_process(sample):
        proc = SampleProcessor(args.dir, sample, options)
        try:
            proc.run()
        except:
            proc.exception()

    if __name__ == '__main__':
        with Pool(args.workers) as p:
            p.map(run_process, samples)

# # Input dir #
# inputDir = "/Users/radoslaw.ejsmont/Desktop/rdn-wdp/samples-csv"
# sample = "63951_disc_5_CJH0WG"
