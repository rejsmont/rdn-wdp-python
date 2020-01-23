#!/usr/bin/env python3

import argparse
import csv
import loompy
import logging
import os
import numpy as np
import pandas as pd
import warnings
from scipy.spatial import distance
from data import DiscData
from clustering import Clustering, ClusteredData


class RNAseq:

    _source = None
    _expression = None
    _exp_all = None

    def __init__(self, data, genes=None):
        self.logger = logging.getLogger('rdn-wdp-RNAseq')
        self.logger.info("Input is " + str(data))
        df = pd.read_csv(data)
        if 'DataSet' in df.columns:
            df.set_index('DataSet', append=True, inplace=True)
            self.data = df.reorder_levels([1, 0])
            del df
        else:
            self.data = df
        self._genes = genes

    def expression(self):
        if self._expression is None:
            matrix = (self._data[np.isin(self._data.ra.Gene, self._genes), :]).transpose()
            columns = [g for g in self._data.ra.Gene if g in self._genes]
            self._expression = pd.DataFrame(matrix, columns=columns).sort_index(axis=1)
        return self._expression

    def expression_all(self):
        if self._exp_all is None:
            matrix = (self._data[:, :]).transpose()
            columns = [g for g in self._data.ra.Gene]
            self._exp_all = pd.DataFrame(matrix, columns=columns).sort_index(axis=1)
        return self._exp_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ChIP data')
    parser.add_argument('--scngs', required=True)
    parser.add_argument('--cells', required=False)
    parser.add_argument('--log')
    parser.add_argument('--outdir')
    parser.add_argument('--reproducible', dest='reproducible', action='store_true')
    parser.add_argument('--not-reproducible', dest='reproducible', action='store_false')
    parser.set_defaults(reproducible=False)
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=args.log.upper())
        logging.getLogger('PIL.Image').setLevel(logging.INFO)
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger('joblib').setLevel(logging.INFO)
        logging.getLogger('cloudpickle').setLevel(logging.INFO)

    if args.reproducible:
        np.random.seed(0)
    else:
        np.random.seed()

    rnas = RNAseq(args.scngs)

