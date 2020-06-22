#!/usr/bin/env python3

import argparse
import logging
import os
import sys
sys.path.append(os.path.join(sys.path[0], '..'))

from CellModels.Cells.Filters import Masks
from CellModels.Cells.IO import CellReader
from CellModels.Clustering.Compute import Clustering
from CellModels.Clustering.Data import ClusteringConfig
from CellModels.Clustering.IO import ClusteringResultsWriter


def run(a):
    if a.log:
        logging.basicConfig(level=a.log.upper())

    data = CellReader.read(a.input)
    cells = data.cells
    logging.info("Read " + str(len(cells.index)) + " cells from " + a.input)

    if a.gene != 'Ato':
        hc_features = [
            ('Position', 'Normalized', 'y'),
            ('Measurements', 'Normalized', 'Venus'),
            ('Measurements', 'Prominence', 'Venus')
        ]
        rf_features = [
            ('Position', 'Normalized', 'x'),
            ('Position', 'Normalized', 'y'),
            ('Position', 'Normalized', 'z'),
            ('Measurements', 'Normalized', 'Venus'),
            ('Measurements', 'Prominence', 'Venus'),
            ('Measurements', 'Angle', 'Venus'),
            ('Measurements', 'Raw', 'Volume')
        ]
        if a.gene:
            cells = cells.loc[[a.gene]]
    else:
        hc_features = [
            ('Position', 'Normalized', 'y'),
            ('Measurements', 'Normalized', 'mCherry'),
            ('Measurements', 'Prominence', 'mCherry')
        ]
        rf_features = [
            ('Position', 'Normalized', 'x'),
            ('Position', 'Normalized', 'y'),
            ('Position', 'Normalized', 'z'),
            ('Measurements', 'Normalized', 'mCherry'),
            ('Measurements', 'Prominence', 'mCherry'),
            ('Measurements', 'Angle', 'mCherry'),
            ('Measurements', 'Raw', 'Volume')
        ]

    if a.sample_list:
        cells = cells.loc[cells.index.isin(a.sample_list.replace(',', '  ').split(), level='Sample')]

    training = None

    if a.furrow:
        if training is None:
            training = cells
        masks = Masks(training)
        training = training[masks.cells_mf_area]
    if a.no_bad:
        if training is None:
            training = cells
        masks = Masks(training)
        training = training[~ masks.samples_bad_segm]

    config = ClusteringConfig({
        'clusters': a.clusters.replace(',', ' ').split(),
        'samples': a.samples,
        'repeats': a.repeats,
        'cutoff': a.cutoff,
        'method': a.method,
        'metric': a.metric,
        'rf_features': rf_features,
        'hc_features': hc_features
    })

    logging.info("Will cluster " + str(len(cells.index)) + " cells")
    if training is not None:
        logging.info("Will train on " + str(len(training.index)) + " cells")

    rf_parallel = not args.rf_no_parallel

    clustering = Clustering(config, cells, training, rf_parallel=rf_parallel)
    if a.mode == 'classify':
        res = clustering.classify()
    elif a.mode == 'cluster':
        res = clustering.cluster()
    else:
        raise ValueError('Only \'classify\' and \'cluster\' can be specified as modes.')
    ClusteringResultsWriter.write(res, a.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster cells based on gene expression')
    parser.add_argument('input')
    parser.add_argument('-k', '--clusters', type=str, default='6')
    parser.add_argument('-n', '--samples', type=int, default=20)
    parser.add_argument('-r', '--repeats', type=int, default=100)
    parser.add_argument('-c', '--cutoff', type=float, default=1.0)
    parser.add_argument('-t', '--method', type=str, default='ward')
    parser.add_argument('-m', '--metric', type=str, default='euclidean')
    parser.add_argument('-g', '--gene', type=str, default='')
    parser.add_argument('-l', '--sample-list', type=str, default='')
    parser.add_argument('-f', '--furrow', action='store_true')
    parser.add_argument('-d', '--mode', type=str, default='classify')
    parser.add_argument('--no-bad', action='store_true')
    parser.add_argument('--rf-no-parallel', action='store_true')
    parser.add_argument('--log')
    parser.add_argument('output')
    args = parser.parse_args()

    run(args)
