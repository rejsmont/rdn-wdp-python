#!/usr/bin/env python3
from collections import Iterable

import numpy as np
import pandas as pd

from CellModels.Cells.Tools import CellColumns
from CellModels.Cluster import ClusteringResult as OriginalClusteringResult
from CellModels.Clustering.Tools import ClusteringTools, MultiClusteringTools


class ClusteringConfig(CellColumns):

    ITER_MAX = 1000
    MIN_SCORE = 1e-10
    HC_FEATURES = [
        ('Position', 'Normalized', 'y'),
        ('Measurements', 'Normalized', 'mCherry'),
        ('Measurements', 'Prominence', 'mCherry')
    ]
    RF_FEATURES = [
        ('Position', 'Normalized', 'x'),
        ('Position', 'Normalized', 'y'),
        ('Position', 'Normalized', 'z'),
        ('Measurements', 'Normalized', 'mCherry'),
        ('Measurements', 'Prominence', 'mCherry'),
        ('Measurements', 'Angle', 'mCherry'),
        ('Measurements', 'Raw', 'Volume')
    ]

    def __init__(self, m):
        if 'clustering' in m.keys():
            m = m['clustering']
        m = m['config']
        c = m.get('clusters', None)
        self._clusters = c if isinstance(c, Iterable) else [c]
        assert self._clusters is not None, 'Number of clusters must be specified.'
        self._samples = m.get('samples', None)
        assert self._clusters is not None, 'Number of samples must be specified.'
        self._repeats = m.get('repeats', None)
        assert self._clusters is not None, 'Number of repeats must be specified.'
        self._cutoff = m.get('cutoff', -1)
        self._method = m.get('method', 'ward')
        self._metric = m.get('metric', 'euclidean')
        self._hc_features = self._t_list(m.get('hc_features', self.HC_FEATURES))
        self._rf_features = self._t_list(m.get('rf_features', self.RF_FEATURES))

    @property
    def clusters(self):
        return self._clusters

    @property
    def samples(self):
        return self._samples

    @property
    def repeats(self):
        return self._repeats

    @property
    def cutoff(self):
        return self._cutoff

    @property
    def method(self):
        return self._method

    @property
    def metric(self):
        return self._metric

    @property
    def rf_features(self):
        return self._rf_features

    @property
    def hc_features(self):
        return self._hc_features

    def to_dict(self):
        return {
            'clusters': self.clusters,
            'samples': self.samples,
            'repeats': self.repeats,
            'cutoff': self.cutoff,
            'method': self.method,
            'metric': self.metric,
            'rf_features': self.rf_features,
            'hc_features': self.hc_features
        }


class SampleSets(dict):

    def __init__(self, m):
        if 'clustering' in m.keys():
            m = m['clustering']
        samples = m.get('samples', None)
        assert samples is not None, 'No sample sets were found in metadata.'
        super().__init__(samples)


class ClusteringResult(OriginalClusteringResult, ClusteringTools):

    def __init__(self, cells, sets, config, clusters=None, centroids=None, training=None, test=None, performance=None):
        self._cells = cells
        self._sample_sets = sets
        self._config = config
        self._clusters = clusters
        self._centroids = centroids
        self._training = training
        self._test = test
        self._cells = self._set_multi_index(self._cells, self._config)
        self._performance = performance

    @property
    def cells(self):
        return self._cells

    @property
    def sample_sets(self):
        return self._sample_sets

    @property
    def config(self):
        return self._config

    @property
    def clusters(self):
        return self._clusters

    @property
    def centroids(self):
        return self._centroids

    @property
    def training(self):
        return self._training

    @property
    def test(self):
        return self._test

    @classmethod
    def _set_multi_index(cls, data, config):
        e = [np.nan for x in range(data.columns.nlevels - 1)]
        n = [
            tuple(['Cluster_' + config.method] + e),
            tuple(['Cluster_' + config.method + '_' + str(config.clusters[0])] + e)
        ]
        t = data.columns.to_list()
        for i, c in enumerate(t):
            for m in n:
                if c == m:
                    t[i] = ('Cluster', config.method, config.clusters[0])
        data.columns = pd.MultiIndex.from_tuples(t)
        return data


class MultiClusteringResult(ClusteringResult):

    def __init__(self, results):
        reference = None
        cells = None
        k = []
        clusters = {}
        centroids = {}

        for result in results:
            current = result.config.to_dict()
            ck = current['clusters']
            del current['clusters']
            k.append(ck)
            if reference is None:
                reference = current
            assert current == reference, "Clustering parameters (except k) must be the same for all results."
            if cells is None:
                cells = result.cells.copy()
            else:
                column = ('Cluster', result.config.method, result.config.clusters)
                cells[column] = result.cells[column]
            clusters[ck] = result.clusters
            centroids[ck] = result.centroids

        reference['clusters'] = k
        self._cells = cells
        self._config = ClusteringConfig(reference)
        self._clusters = clusters
        self._centroids = centroids

    @property
    def cells(self):
        return self._cells

    @property
    def config(self):
        return self._config

    @property
    def clusters(self):
        return self._clusters

    @property
    def centroids(self):
        return self._centroids


class HarmonizedClusteringResult(MultiClusteringResult, MultiClusteringTools):

    def __init__(self, results):
        if isinstance(results, MultiClusteringResult):
            self._cells = results._cells
            self._config = results._config
        elif isinstance(results, list):
            super().__init__(results)
        else:
            raise ValueError("Results must be a list or an instance of MultiClusteringResult")
        harmonized = self._harmonize(self._cells, self._config.rf_features)
        self._clusters_t = self._cluster_table(self._cells).join(self._cluster_table(harmonized),
                                                                 rsuffix=' harmonized').rename(
            columns={'Cluster harmonized': 'Harmonized cluster'})
        self._cells = harmonized

    def get_linkage(self, features=None, rename=True, names=False):
        features = self._config.rf_features if features is None else features
        return self.linkage(self._cells, features, rename, names)

    def representative_sample(self):
        d = self.get_linkage()
        dc = d[:, 2]
        distances = []
        samples = self._cells.index.unique('Sample')
        for sample in samples:
            idx = pd.IndexSlice
            try:
                ds = self.linkage(self._cells.loc[idx[:, sample], :],
                                  self._config.rf_features)
            except KeyError:
                distances.append(np.nan)
                continue
            dsc = ds[:, 2]
            if d.shape == ds.shape:
                distances.append(np.sum(np.abs(dc - dsc)))
            else:
                d_mask = (dc[:, None] == dsc).all(-1).any(-1)
                ds_mask = (dsc[:, None] == dc).all(-1).any(-1)
                distance = np.sum(np.abs(d[d_mask, 2] - ds[ds_mask, 2])) \
                    + np.sum(np.square(d[~d_mask, 2])) + np.sum(np.square(ds[~ds_mask, 2]))
                distances.append(distance)

        return samples[distances.index(np.nanmin(distances))]

    def best_clustering(self):
        lk = self.get_linkage(rename=False)
        cutoff = np.mean(lk[:, 2])
        c = np.max(lk[lk[:, 2] > cutoff, 0:2])
        for column in self._cluster_columns(self._cells):
            if c in self._cells[column].values:
                return column, cutoff
