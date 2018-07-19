#!/usr/bin/env python3

import csv
import argparse
import re
import string

from numpy import *
from scipy import *
import os
import peakutils
import matplotlib.pyplot as plt
from scipy import stats

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyse DoG segmentation results.')
parser.add_argument('input', help='The input directory')
args = parser.parse_args()
inputdir = args.input


filelist = []
for path, subdirs, files in os.walk(str(inputdir)):
    for name in files:
        if name.endswith(".csv") and "sample_1" in name:
            filelist.append(os.path.join(path, name))

outfile = os.path.join(inputdir, "dog-param.csv")
output = open(outfile, 'w')
wr = csv.writer(output, dialect='excel')


for file in filelist:
    print(file)
    plotfile = file.replace(".csv", ".pdf")
    file2 = file.replace("sample_1", "sample_2")
    data1 = loadtxt(file, delimiter=',', dtype=float, skiprows=1, ndmin=2)
    if os.path.isfile(file2):
        print(file2)
        data2 = loadtxt(file2, delimiter=',', dtype=float, skiprows=1, ndmin=2)
        data = concatenate((data1, data2))
    else:
        data = data1

    count = data[:,0].size
    volume = data[:,4]
    volume_mean = mean(volume)
    volume_var = var(volume)
    volume_std = std(volume)

    if count > 1:
        density = stats.kde.gaussian_kde(volume)
        x = arange(min(volume), max(volume), 1)
        y = density(x)
        indexes = peakutils.indexes(y)
        interpolatedIndexes = peakutils.interpolate(x, y, ind=indexes)
        plt.plot(x, y)
        plt.savefig(plotfile, bbox_inches='tight')
        plt.close()

    pat = re.compile("[-_]")
    parts = pat.split(file.replace(".csv", ""))
    sigmaA = float(parts[2])
    ratio = float(parts[3])
    sigmaB = sigmaA / ratio
    radius = float(parts[4])
    cutoff = float(parts[5])

    row = [sigmaB, sigmaA, ratio, radius, cutoff, count, volume_mean, volume_var, volume_std]
    row.extend(interpolatedIndexes.tolist())

    wr.writerow(row)
