#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
import hashlib
import statistics
from CellModels.Filters import Morphology


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


class ClusteringResult:

    def __init__(self):
        self.centroids = None
        self.clusters = pd.DataFrame()
        self.names = None
        self.sample_sets = {}
        self.cells = None
        self.status = 'empty'
        self.config = None


class Clustering:

    def __init__(self, config, data, training=None):
        self._data = data
        if training is None:
            self._training = data
        else:
            self._training = training
        self._config = config
        self._logger = logging.getLogger('rdn-wdp-clustering')
        self._result = None

    def cluster(self):
        if len(self._data) <= self._config.samples and self._config.repeats > 1:
            self._logger.warning("The total number of samples is lower than or equal to the sample size. " +
                                 "Setting the repeats to more than 1 does not make much sense.")
        elif self._config.repeats > 1:
            self._logger.warning("The repeats are set to more than 1. Consider using the classify method.")
        self._find_centroids()
        self._result.config = self._config
        return self._result

    def classify(self):
        if len(self._data) <= self._config.samples:
            self._logger.warning("The total number of samples is lower than or equal to the sample size. " +
                                 "Running random forest classifier over fully clustered samples " +
                                 "does not make much sense.")
            if self._config.repeats > 1:
                self._logger.warning("Setting the repeats to more than 1 does not make much sense.")
        self._find_centroids()
        self._cluster_centroids()
        self._random_forest()
        self._result.config = self._config
        return self._result

    @staticmethod
    def _is_one_gene(samples):
        last = None
        for sample, gene in samples:
            if last is None:
                last = gene
            if last != gene:
                return False
        return True

    def _get_samples(self, unique=True):
        samples = self._training[['Sample', 'Gene']].drop_duplicates().values.tolist()
        if unique:
            unique = not self._is_one_gene(samples)
        selected = []
        genes = []
        while len(selected) < self._config.samples:
            if len(samples) == 0:
                break
            index = np.random.randint(0, len(samples))
            sample, gene = samples.pop(index)
            if gene not in genes:
                genes.append(gene)
                selected.append(sample)
            elif not unique:
                selected.append(sample)
        return sorted(selected)

    def _cluster(self):
        c = self._config
        cells = self._training
        samples = self._get_samples()
        set_id = hashlib.md5(str(samples).encode()).hexdigest()
        self._logger.debug("Clustering sample set " + str(set_id) + ": ")
        self._logger.debug(str(samples))
        mask = cells['Sample'].isin(samples)
        cells = self._training[mask]
        x = stats.zscore(cells[c.hc_features].values, axis=0)
        i = cells.index.values
        # z = linkage(x, c.method)
        # fc = fcluster(z, c.clusters, criterion='maxclust')
        cluster = AgglomerativeClustering(n_clusters=c.clusters, affinity=c.metric, linkage=c.method)
        cluster.fit_predict(x)
        fc = cluster.labels_ + 1
        index = pd.MultiIndex.from_product([[set_id], [c.method], i], names=['SampleSet', 'Method', 'Cell'])
        clusters = pd.DataFrame(fc, columns=['LocalCluster'], index=index)
        return set_id, clusters, samples

    def _find_centroids(self):
        c = self._config
        if self._result is None:
            self._logger.debug('Creating new results object...')
            self._result = ClusteringResult()
        r = self._result
        for i in range(0, c.repeats):
            self._logger.debug('Iteration ' + str(i+1) + ': computing ' + str(c.method))
            try:
                set_id, clusters, samples = self._cluster()
                r.clusters = r.clusters.append(clusters)
                r.sample_sets[set_id] = samples
            except Exception as e:
                self._logger.warning('Computing ' + str(c.method) + ' failed', exc_info=e)
        r.centroids = r.clusters.join(
            self._training[self._config.hc_features], on='Cell').groupby(
            ['SampleSet', 'Method', 'LocalCluster'])[self._config.hc_features].mean()
        r.centroids[r.centroids.columns] = stats.zscore(r.centroids.values)
        r.centroids = r.centroids.sort_index()
        r.status = 'found_centroids'
        if c.repeats == 1 and len(self._data.index) == len(self._training.index):
            r.cells = self._training.copy()
            r.cells['Cluster_' + c.method] = r.clusters['LocalCluster'].values

    def _cluster_centroids(self):
        c = self._config
        r = self._result
        if r is None or r.status != 'found_centroids':
            raise RuntimeError("_cluster_centroids method must be run immediately after _find_centroids")

        r.centroids['Cluster'] = 0
        r.centroids['Distance'] = 0.0

        def create_df(array):
            centroids = pd.DataFrame(
                array, columns=c.hc_features,
                index=range(1, array.shape[0] + 1)).rename_axis('LocalCluster')
            clusters = pd.DataFrame(
                [['global', c.method, x, x, 0.0] for x in range(1, array.shape[0] + 1)],
                columns=['SampleSet', 'Method', 'LocalCluster', 'Cluster', 'Distance']
            ).set_index(['SampleSet', 'Method', 'LocalCluster'])
            return clusters.join(centroids, on='LocalCluster').reindex(columns=r.centroids.columns)

        self._logger.info('Clustering clusters...')
        iters = 0
        last = np.zeros(r.centroids.loc[(r.centroids.index.levels[0][0], c.method), c.hc_features].values.shape)
        prev = last
        score = np.inf
        while score > c.MIN_SCORE and iters < c.ITER_MAX:
            sample_sets = list(r.sample_sets.keys())
            if iters == 0:
                index = np.random.randint(0, len(sample_sets))
                last = r.centroids.loc[(sample_sets.pop(index), c.method), c.hc_features].values
                samples = 1
                new = last.copy()
            else:
                samples = 0
                new = np.zeros(last.shape)
                last = prev
            iters = iters + 1
            while len(sample_sets) > 0:
                index = np.random.randint(0, len(sample_sets))
                cid = sample_sets.pop(index)
                current = r.centroids.loc[(cid, c.method), c.hc_features].values
                distances = cdist(last, current)
                axis = 0
                for ax in [1, 0]:
                    pairs = np.argmin(distances, axis=ax)
                    ambiguous = len(pairs) - len(np.unique(pairs))
                    if ambiguous == 0:
                        axis = ax
                        break
                if ambiguous == 0:
                    samples = samples + 1
                    for i in range(0, new.shape[0]):
                        if axis == 1:
                            new[i] = new[i] + current[pairs[i]]
                            r.centroids.loc[(cid, c.method, pairs[i] + 1), ['Cluster', 'Distance']] = \
                                [(i + 1), distances[pairs[i], i]]
                        else:
                            new[pairs[i]] = new[pairs[i]] + current[i]
                            r.centroids.loc[(cid, c.method, i + 1), ['Cluster', 'Distance']] = \
                                [(pairs[i] + 1), distances[i, pairs[i]]]
                    if iters == 0:
                        last = new / samples
            last = new / samples
            score = np.sum(np.abs(prev - last))
            prev = last
        self._logger.info('Done in ' + str(iters) + ' iterations. Score is ' + str(score))
        r.centroids = r.centroids.append(create_df(last))
        r.clusters = r.clusters.join(r.centroids['Cluster'], on=['SampleSet', 'Method', 'LocalCluster'])
        r.status = 'clustered_centroids'

    def _random_forest(self):
        c = self._config
        r = self._result
        if r is None or r.status != 'clustered_centroids':
            raise RuntimeError("_random_forest method must be run after _cluster_centroids")

        def cluster_mode(series):
            try:
                return statistics.mode(series.tolist())
            except statistics.StatisticsError:
                return 0

        self._logger.info('Training classifier...')
        idx = pd.IndexSlice
        if c.cutoff == -1.0:
            clusters = r.clusters.loc[idx[:, c.method, :], :]
        else:
            clusters = r.clusters.loc[idx[:, c.method, :], :]\
                .join(r.centroids.loc[r.centroids['Distance'] < c.cutoff, 'Distance'],
                      on=['SampleSet', 'Method', 'LocalCluster'], how='right')
        global_clusters = clusters.groupby('Cell')['Cluster'].agg(cluster_mode)
        global_clusters.drop(global_clusters[global_clusters == 0].index, inplace=True)
        rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
        rf.fit(self._training.loc[global_clusters.index, c.rf_features], global_clusters)
        self._logger.info('Computing predictions...')
        cells = self._data.copy()
        cells['Cluster_' + c.method] = 0
        cells['Cluster_' + c.method] = rf.predict(self._data[c.rf_features])
        r.cells = cells
        r.status = 'random_forest'
        self._logger.info('Predictions done.')


class AtoClustering(Clustering):

    def _name_clusters(self):
        c = self._config
        r = self._result
        if r is None:
            raise RuntimeError("_name_clusters method must be run after _find_centroids")

        idx = pd.IndexSlice
        named_clusters = r.centroids.loc[idx['global', c.method, :], :]
        # Cluster A - R8 cells
        cluster_a = named_clusters['ext_mCherry'].idxmax()
        named_clusters = named_clusters.drop(cluster_a)
        # Cluster B - MF high ato
        cluster_b = named_clusters['mCherry'].idxmax()
        named_clusters = named_clusters.drop(cluster_b)
        # Cluster C - post MF
        cluster_c = named_clusters['cy'].idxmax()
        named_clusters = named_clusters.drop(cluster_c)
        # Cluster D - pre MF
        cluster_d = named_clusters['cy'].idxmin()
        named_clusters = named_clusters.drop(cluster_d)
        # Cluster E - MF ato
        cluster_e = named_clusters['mCherry'].idxmax()
        named_clusters = named_clusters.drop(cluster_e)
        # Cluster F - MF background
        cluster_f = named_clusters['mCherry'].idxmin()
        index = pd.MultiIndex.from_tuples([cluster_a, cluster_b, cluster_c, cluster_d, cluster_e, cluster_f],
                                          names=['SampleSet', 'Method', 'Cluster'])
        names = pd.DataFrame(['R8', 'MF-high', 'post-MF', 'pre-MF', 'MF-ato', 'MF'],
                             index=index, columns=['Name'])
        r.centroids = r.centroids.join(names.xs(('global', c.method)), on='Cluster')

    def _classify_remaining(self):
        c = self._config
        r = self._result
        if r is None:
            raise RuntimeError("_classify_remaining method must be run after _find_centroids")

        idx = pd.IndexSlice
        post_mf = r.centroids.loc[idx['global', c.method, :]]\
            .loc[r.centroids['Name'] == 'post-MF'].index.get_level_values(2)
        r.cells.loc[r.cells['cy'] > Morphology.FURROW_BROAD.max, ['Cluster_' + c.method]] = post_mf
        pre_mf = r.centroids.loc[idx['global', c.method, :]] \
            .loc[r.centroids['Name'] == 'pre-MF'].index.get_level_values(2)
        r.cells.loc[r.cells['cy'] < Morphology.FURROW_BROAD.min, ['Cluster_' + c.method]] = pre_mf


    # def compute(self):
    #     find_centroids(self.method)
    #     self.centroids['Cluster'] = 0
    #     self.centroids['Distance'] = 0.0
    #     cluster_centroids(self.method)
    #     self.clusters = self.clusters.join(self.centroids['Cluster'], on=['SampleSet', 'Method', 'LocalCluster'])
    #     random_forest(self.method, cutoff=self.cutoff)
    #     name_clusters()
    #     classify_remaining()
    #     self._logger.info('Cell classification done')
