#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import math
import scipy.spatial as spa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import threshold_triangle
from skimage.morphology import skeletonize
from skimage.measure import label


def process_sample(directory, sample):

    print(" - processing " + sample)

    # Input file #
    inputFile = os.path.join(directory, sample)
    baseName = os.path.splitext(sample)[0]
    nuclei = pd.read_csv(inputFile)

    # Filter discs by mean volume +/- 50%
    volumeMean = nuclei['Volume'].mean()
    nuclei = nuclei[(nuclei['Volume'] > volumeMean * 0.5) & (nuclei['Volume'] < volumeMean * 1.5)]

    # Compute mean nucleus diameter
    unitD = 2 * math.pow((3 * volumeMean) / (4 * math.pi), 1.0 / 3.0)

    # Scale units to mean diameter
    nuclei['cx'] = nuclei['cx'] / unitD
    nuclei['cy'] = nuclei['cy'] / unitD
    nuclei['cz'] = nuclei['cz'] / unitD

    origin = find_origin(nuclei)
    if origin < 2:
        nuclei['cy'] = nuclei['cy'].max() - nuclei['cy']
    if origin % 2 != 0:
        nuclei['cx'] = nuclei['cx'].max() - nuclei['cx']

    nuclei['mCherry'] = nuclei['Mean 1']
    nuclei['Venus'] = nuclei['Mean 2']
    nuclei['DAPI'] = nuclei['Mean 0']

    furrow = find_furrow(nuclei)
    thumbnail(nuclei, None, os.path.join(directory, baseName + "_thumb_raw"))

    unitI = nuclei.loc[round(nuclei['cy']) == round(round(nuclei['cx']).map(furrow)), 'Mean 1'].mean()
    nuclei['mCherry'] = nuclei['Mean 1'] / unitI
    nuclei['Venus'] = nuclei['Mean 2'] / unitI
    nuclei['DAPI'] = nuclei['Mean 0'] / unitI
    nuclei.loc[nuclei['cy'] > round(round(nuclei['cx']).map(furrow)) + 10, 'mCherry'] = nuclei['mCherry'].min()
    thumbnail(nuclei, furrow)

    nuclei_p = pd.DataFrame()
    nuclei_p['cx'] = nuclei['cx']
    nuclei_p['cy'] = nuclei['cy'] - round(round(nuclei['cx']).map(furrow))
    nuclei_p['cz'] = nuclei['cz']
    nuclei_p['mCherry'] = nuclei['mCherry']
    nuclei_p['Venus'] = nuclei['Venus']
    nuclei_p['DAPI'] = nuclei['DAPI']
    nuclei_p['Volume'] = nuclei['Volume']
    nuclei_p['ext_mCherry'] = np.NaN
    nuclei_p['ext_Venus'] = np.NaN
    nuclei_p['ang_max_mCherry'] = np.NaN
    nuclei_p['ang_max_Venus'] = np.NaN
    nuclei_p.reset_index()

    thumbnail(nuclei_p, None)

    KDtree = spa.cKDTree(nuclei_p[['cx', 'cy', 'cz']].values)

    count = len(nuclei_p.index)
    for index, nucleus in nuclei_p.iterrows():
        distances, indices = KDtree.query(nucleus[['cx', 'cy', 'cz']].values, range(2, 28), distance_upper_bound=2)
        indices = indices[indices < count]
        neighbors = nuclei_p.iloc[indices]
        nuclei_p.at[index, 'ext_mCherry'] = nucleus['mCherry'] / neighbors['mCherry'].mean()
        nuclei_p.at[index, 'ext_Venus'] = nucleus['Venus'] / neighbors['Venus'].mean()
        max_cherry_value = neighbors['mCherry'].max()
        max_venus_value = neighbors['Venus'].max()
        max_cherry_neighbor = neighbors.loc[neighbors['mCherry'] == max_cherry_value]
        max_venus_neighbor = neighbors.loc[neighbors['Venus'] == max_venus_value]
        if len(max_cherry_neighbor.index) == 1:
            nuclei_p.at[index, 'ang_max_mCherry'] = nuclei_angle(nucleus, max_cherry_neighbor)
        if len(max_venus_neighbor.index) == 1:
            nuclei_p.at[index, 'ang_max_Venus'] = nuclei_angle(nucleus, max_venus_neighbor)

    #nuclei_p.to_csv(os.path.join(directory, baseName + "_normalized.csv"))
    return nuclei_p


def count_corners(nuclei, range=5):

    max_x = nuclei['cx'].max()
    max_y = nuclei['cy'].max()

    top_left = nuclei.loc[(round(nuclei['cy']) < range) & (round(nuclei['cx']) < range), 'Mean 1'].count()
    top_right =\
        nuclei.loc[(round(nuclei['cy']) < range) & (round(nuclei['cx']) > round(max_x - range)), 'Mean 1'].count()
    btm_left =\
        nuclei.loc[(round(nuclei['cy']) > round(max_y - range)) & (round(nuclei['cx']) < range), 'Mean 1'].count()
    btm_right =\
        nuclei.loc[(round(nuclei['cy']) > round(max_y - range)) & (round(nuclei['cx']) > round(max_x - range)),
                   'Mean 1'].count()

    corners = [top_left, top_right, btm_left, btm_right]

    return corners


def find_origin(nuclei, range=5):

    zeroes = 4
    i = 1

    while zeroes > 1 and i <= range:
        corners = count_corners(nuclei, i)
        zeroes = sum(c == 0 for c in corners)
        i = i + 1

    if zeroes == 0:
        print("   [Warning] - unable to detect empty corners - nuclei occupy whole field of view.")
        return 2
    if zeroes == 4:
        print("   [Warning] - more than one corner is empty.")
        return 2

    return corners.index(min(corners))


def disc_matrix(input_data, fields, method='max'):
    """
    :param input_data:  Nuclei from csv as Pandas DataFrame
    :param fields:      Field to read intensities from
    :param method:      Method used to project data (default max)
    :return:            2D Numpy array with intensity values
    """

    data = pd.DataFrame()

    # Compute rounded coordinates
    data['ux'] = round(input_data['cx'])
    data['uy'] = round(input_data['cy'])
    data['uz'] = round(input_data['cz'])
    data['int'] = input_data[fields]

    # Compute bounds
    x_min = int(data['ux'].min())
    x_max = int(data['ux'].max())
    y_min = int(data['uy'].min())
    y_max = int(data['uy'].max())

    # Initialize empty array
    matrix = np.zeros([x_max, y_max])

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

    return smooth


def channel_matrix(disc, channel, method=None):
    if method is not None:
        return np.transpose(disc_matrix(disc, channel, method))
    else:
        return np.transpose(disc_matrix(disc, channel))


def display_normalize(data):
    return ((data / (data.mean() * 2)) * 255).clip(0, 255).astype('uint8')


def detect_ridges(gray, sigma=3.0):

    pad = round(sigma * 3)
    scaled = np.pad(gray, pad, 'edge')

    hessian = hessian_matrix(scaled, sigma, order="rc")
    hessian[0] = np.zeros(hessian[0].shape)
    hessian[1] = np.zeros(hessian[1].shape)

    i1, i2 = hessian_matrix_eigvals(hessian)
    i1 = i1[pad:gray.shape[0]+pad, pad:gray.shape[1]+pad]
    i2 = i2[pad:gray.shape[0]+pad, pad:gray.shape[1]+pad]

    return i1, i2


def rms_fit(cx, cy):

    for degree in range(0, 20):
        fn = np.polyfit(cx, cy, degree)
        fy = np.polyval(fn, cx)
        error_mean = ((cy - fy) ** 2).mean()
        error_max = ((cy - fy) ** 2).max()

        if error_mean < 0.25 and error_max < 1:
            return fn


def find_furrow(disc):

    furrow_disc = pd.DataFrame()
    furrow_disc['cx'] = disc['cx']
    furrow_disc['cy'] = disc['cy']
    furrow_disc['cz'] = disc['cz']
    furrow_disc['mCherry'] = disc['mCherry']
    furrow_disc['DAPI'] = 1 / disc['DAPI']
    dapi_m = channel_matrix(furrow_disc, 'DAPI', 'mean')
    cherry_m = channel_matrix(furrow_disc, 'mCherry', 'mean')

    dhe_min, dhe_max = detect_ridges(dapi_m)
    xhe_min, che_max = detect_ridges(cherry_m)
    he_max = dhe_max * che_max

    threshold = threshold_triangle(he_max)
    thresholded = he_max > threshold
    skeleton = skeletonize(thresholded)
    labels = label(skeleton)

    lines = []

    for line in range(1, labels.max() + 1):
        image = (labels == line)
        lines.append((line, image[image == True].size, image.argmax(axis=0)))

    lines = sorted(lines, key=lambda line: line[1], reverse=True)

    for line in lines:
        line_label = line[0]
        positions = line[2]
        for i in range(0, len(positions)):
            if labels[int(positions[i]), i] != line_label:
                positions[i] = -1

    for i in range(0, len(lines)):
        if lines[i] is not None:
            for j in range(i + 1, len(lines)):
                if lines[j] is not None:
                    collisions = 0
                    positions_i = lines[i][2]
                    positions_j = lines[j][2]
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
        for i in range(0, len(positions)):
            if positions[i] == -1:
                for j in range(1, len(lines)):
                    if lines[j] is not None:
                        if lines[j][2][i] != -1:
                            positions[i] = lines[j][2][i]
                            break

    mask = positions >= 0
    indices = np.arange(0, len(positions))
    cx = indices[mask]
    cy = positions[mask]
    fn = rms_fit(cx, cy)

    return np.poly1d(fn)


def thumbnail(disc, f=None, basename=""):
    """ Generate thumbnail of nuclear image """

    min = disc['cy'].min()

    if min < 0:
        disc['cy'] = disc['cy'] - min
        f = np.poly1d([0, -min])

    mCherry = np.transpose(disc_matrix(disc, 'mCherry'))
    Venus = np.transpose(disc_matrix(disc, 'Venus'))
    DAPI = np.transpose(disc_matrix(disc, 'DAPI'))

    if mCherry.max() > 1:
        mCherry = (mCherry / mCherry.max() * 255).astype('uint8')
    if Venus.max() > 1:
        Venus = (Venus / Venus.max() * 255).astype('uint8')
    if DAPI.max() > 1:
        DAPI = (DAPI / DAPI.max() * 255).astype('uint8')

    stack = np.stack((display_normalize(mCherry), display_normalize(DAPI), display_normalize(Venus)), axis=2)
    img = Image.fromarray(stack, 'RGB')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(stack)
    x = np.arange(0, stack.shape[1])
    if f is not None:
        plt.plot(x, f(x))
    ax.set_aspect('equal')

    if basename:
        img.save(basename + ".tif")
        plt.savefig(basename + ".png")
    else:
        img.show()
        plt.show()

    img.close()
    plt.close()


def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'. """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def nuclei_angle(n1, n2):
    """ Returns the o'clock position of a nucleus relative to another nucleus """
    v1 = [0, - n1['cy']]
    v2 = [n2['cx'].values[0] - n1['cx'], n2['cy'].values[0] - n1['cy']]

    if v1 == [0, 0] or v2 == [0, 0]:
        return np.NaN
    else:
        return angle_between(v1, v2)


# # Input dir #
# inputDir = "/Users/radoslaw.ejsmont/Desktop/rdn-wdp/samples-csv"
# sample = "61069_disc_8_5VUZUB"
# process_sample(inputDir, sample + '.csv')


# Input dir #
inputDir = sys.argv[1]
workers = int(sys.argv[2])
print("Processing " + inputDir + " with " + str(workers) + " workers.")


def process_dir_sample(sample):
    process_sample(inputDir, sample)
    return True


samples = [f for f in os.listdir(inputDir) if
           os.path.isfile(os.path.join(inputDir, f)) and f.endswith(".csv") and not f.endswith("normalized.csv")]

if __name__ == '__main__':
    with Pool(workers) as p:
        p.map(process_dir_sample, samples)
