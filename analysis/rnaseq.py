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
        self._data = loompy.connect(data)
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


class ClusteredRNAseq(RNAseq):

    _clustered = None
    _assignments = None

    def __init__(self, data, reference):
        super().__init__(data, reference.genes())
        self._reference = reference.expression()
        self._matrix = None
        self._classes = None
        self._rdata = reference

    def _distance_vector(self, vector):
        distances = []
        indices = []
        midx = vector.idxmax()
        name = vector.name
        if vector.loc[midx] == 0:
            query = vector
        else:
            query = vector / vector[midx]
        for i, row in self._reference.iterrows():
            reference = row / row[midx]
            d = distance.euclidean(query, reference)
            distances.append(d)
            indices.append(i)
        return pd.Series(distances, index=indices, name=name)

    def _distance_vector2(self, vector):
        distances = []
        indices = []
        name = vector.name
        #if vector.loc[midx] == 0:
        #    query = vector
        #else:
        #    query = vector / vector[midx]
        print(len(vector))
        self.normatrix = self._divide_vector(vector)
        self.vector = vector

        for i, row in self._reference.iterrows():
            reference = row / row[midx]
            d = distance.euclidean(query, reference)
            distances.append(d)
            indices.append(i)
        return pd.Series(distances, index=indices, name=name)

    def _divide_vector(self, vector):
        normatrix = np.zeros([len(vector), len(vector)])
        for x in range(0, len(vector)):
            for y in range(0, len(vector)):
                normatrix[x, y] = vector[x] / vector[y]
        print(normatrix)
        return normatrix

    def _distance_matrix(self):
        if self._matrix is None:
            matrix = pd.DataFrame()
            for i, row in super().expression().iterrows():
                dv = self._distance_vector2(row)
                matrix = matrix.append(dv)
            self._matrix = matrix
        return self._matrix

    def _assign(self):
        if self._classes is None:
            self._classes = self._distance_matrix().idxmin(axis=1).apply(pd.Series, index=['Cluster_ward', 'cy'])
        return self._classes

    def expression(self):
        #assign = self._assign()
        expression = super().expression()
        return expression #.join(assign)

    def expression_all(self):
        return super().expression_all().join(self._assign())

    @staticmethod
    def q99(x):
        return np.percentile(x, 99)

    def histogram(self, gene):
        cells = self._rdata.cells()[
            (self._rdata.cells()['Gene'] == gene) &
            self._rdata.acceptable_mask() &
            ~self._rdata.bad_gene_mask()]

        rcells = cells['Venus'].sort_values()
        rmedian = rcells.median()
        rstep = rmedian / 10
        rgb = round(rcells / rstep)
        rhist = rcells.groupby(rgb).count().sort_index()

        qcells = self.expression()[gene].sort_values()
        qmedian = qcells.median()
        qstep = qmedian / 10
        qgb = round(qcells / qstep)
        qhist = qcells.groupby(qgb).count().sort_index()

        return rhist, qhist

    def profiles(self):

        self.sc_calib = self.expression()[['Abl', 'betaTub60D']]
        self.img_calib = self._reference[['Abl', 'betaTub60D']]

        return None
        return self.expression_all().groupby(['Cluster_ward', 'cy']).agg([np.mean, self.q99, 'count'])


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

    cells = DiscData(args.cells)
    clustering = Clustering(args.cells, disc_data=cells)
    clustered = ClusteredData(clustering.cells)
    rnas = ClusteredRNAseq(args.scngs, clustered)
    abl = rnas.histogram('Abl')
    beta = rnas.histogram('betaTub60D')
    fas = rnas.histogram('Fas2')
