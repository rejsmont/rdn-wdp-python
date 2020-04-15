#!/usr/bin/env python3

import argparse
import logging

from CellModels.Cluster import ClusteringConfig, Clustering
from CellModels.Filters import Masks
from CellModels.IO import CellReader, ClusteringResultsWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster cells based on gene expression')
    parser.add_argument('input')
    parser.add_argument('-k', '--clusters', type=int, default=6)
    parser.add_argument('-n', '--samples', type=int, default=20)
    parser.add_argument('-r', '--repeats', type=int, default=100)
    parser.add_argument('-c', '--cutoff', type=float, default=1.0)
    parser.add_argument('-t', '--method', type=str, default='ward')
    parser.add_argument('-m', '--metric', type=str, default='euclidean')
    parser.add_argument('-g', '--gene', type=str, default='')
    parser.add_argument('-f', '--furrow', action='store_true')
    parser.add_argument('--no-bad', action='store_true')
    parser.add_argument('--log')
    parser.add_argument('output')
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=args.log.upper())

    data = CellReader.read(args.input)
    cells = data.cells

    logging.info("Read " + str(len(cells.index)) + " cells from " + args.input)

    if args.gene != 'Ato':
        hc_features = ['cy', 'Venus', 'ext_Venus']
        rf_features = ['cx', 'cy', 'cz', 'Venus', 'ext_Venus', 'ang_max_Venus', 'Volume']
        if args.gene:
            cells = cells[cells['Gene'] == args.gene]
    else:
        hc_features = ['cy', 'mCherry', 'ext_mCherry']
        rf_features = ['cx', 'cy', 'cz', 'mCherry', 'ext_mCherry', 'ang_max_mCherry', 'Volume']

    training = None

    if args.furrow:
        if training is None:
            training = cells
        masks = Masks(training)
        training = training[masks.cells_mf_area]
    if args.no_bad:
        if training is None:
            training = cells
        masks = Masks(training)
        training = training[~ masks.samples_bad_segm]

    config = ClusteringConfig(args.clusters, args.samples, args.repeats, cutoff=args.cutoff,
                              method=args.method, metric=args.metric,
                              hc_features=hc_features, rf_features=rf_features)

    logging.info("Will cluster " + str(len(cells.index)) + " cells")
    if training is not None:
        logging.info("Will train on " + str(len(training.index)) + " cells")

    clustering = Clustering(config, cells, training)
    res = clustering.classify()
    ClusteringResultsWriter.write(res, args.output)

