#!/usr/bin/env python3

import numpy as np
import pandas as pd
from CellModels.Cluster import ClusteringResult as OriginalClusteringResult
from CellModels.Clustering.Tools import ClusteringTools, MultiClusteringTools


class ClusteringConfig:

    ITER_MAX = 1000
    MIN_SCORE = 1e-10
    HC_FEATURES = ['cy', 'mCherry', 'ext_mCherry']
    RF_FEATURES = ['cx', 'cy', 'cz', 'mCherry', 'ext_mCherry', 'ang_max_mCherry', 'Volume']

    def __init__(self, k, n, r, cutoff=-1, method='ward', metric='euclidean', hc_features=None, rf_features=None):
        self._clusters = k
        self._samples = n
        self._repeats = r
        self._cutoff = cutoff
        self._method = method
        self._metric = metric
        if hc_features is None:
            self._hc_features = self.HC_FEATURES
        else:
            self._hc_features = hc_features
        if rf_features is None:
            self._rf_features = self.RF_FEATURES
        else:
            self._rf_features = rf_features

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


class ClusteringResult(OriginalClusteringResult, ClusteringTools):

    def __init__(self, cells, sets, config, clusters=None, centroids=None):
        self._cells = cells
        self._sample_sets = sets
        self._config = config
        self._clusters = clusters
        self._centroids = centroids

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


class MultiClusteringResult(ClusteringResult):

    def __init__(self, results):
        reference = None
        clusters = []
        cells = None

        for result in results:
            if reference is None:
                reference = result.config.to_dict()
                clusters.append(reference['clusters'])
                del reference['clusters']
            else:
                current = result.config.to_dict()
                clusters.append(current['clusters'])
                del current['clusters']
                if current != reference:
                    raise ValueError("Clustering parameters (except k) must be the same for all results.")

            if cells is None:
                cells = result.cells.copy()
            else:
                column = ('Cluster', result.config.method, result.config.clusters)
                cells[column] = result.cells[column]

        self._cells = cells
        self._config = ClusteringConfig(
            clusters,
            reference['samples'],
            reference['repeats'],
            reference['cutoff'],
            reference['method'],
            reference['metric'],
            reference['hc_features'],
            reference['rf_features']
        )

    @property
    def cells(self):
        return self._cells

    @property
    def config(self):
        return self._config


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
                    + np.sum(d[~d_mask, 2]) + np.sum(ds[~ds_mask, 2])
                distances.append(distance)

            return samples[distances.index(np.nanmin(distances))]

    def best_clustering(self):
        lk = self.get_linkage(rename=False)
        cutoff = np.mean(lk[:, 2])
        c = np.max(lk[lk[:, 2] > cutoff, 0:2])
        print(cutoff, c)
        for column in self._cluster_columns(self._cells):
            if c in self._cells[column].values:
                return column, cutoff
