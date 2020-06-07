#!/usr/bin/env python3

import logging
import multiprocessing
import tempfile

import numpy as np
import pandas as pd
from multiprocessing import Pool

from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
import hashlib
import statistics

from CellModels.Cells.Filters import Morphology
from CellModels.Clustering.Data import ClusteringResult, MultiClusteringResult


class Clustering:

    _logger = logging.getLogger('rdn-wdp-clustering')

    class Result:
        def __init__(self):
            self.centroids = None
            self.clusters = pd.DataFrame()
            self.sample_sets = {}
            self.cells = None
            self.status = 'empty'
            self.performance = {}

    def __init__(self, config, data, training=None, test=None):
        fields = config.hc_features + config.rf_features
        # Make sure that data does not contain NaNs in the fields used for clustering
        self._data = data.loc[data[fields].dropna().index]
        if training is None:
            self._training = data
        else:
            # Make sure that training data does not contain NaNs in the fields used for clustering
            self._training = training.loc[training[fields].dropna().index]
        self._config = config
        self._result = None
        if test is None:
            # Use 10% of data for performance testing
            self._test = self._training.sample(frac=0.1).sort_index().index
        else:
            # Make sure that test data does not contain NaNs in the fields used for clustering
            self._test = test.loc[test[fields].dropna().index]

    def cluster(self):
        if len(self._data.index.unique('Sample')) <= self._config.samples and self._config.repeats > 1:
            self._logger.warning("The total number of samples is lower than or equal to the sample size. " +
                                 "Setting the repeats to more than 1 does not make much sense.")
        elif self._config.repeats > 1:
            self._logger.warning("The repeats are set to more than 1. Consider using the classify method.")

        self._find_centroids()
        if self._config.repeats > 1:
            self._cluster_centroids()
            self._label_clusters()
        r = self._result

        if len(self._config.clusters) > 1:
            cls = MultiClusteringResult
        else:
            cls = ClusteringResult

        return cls(r.cells, r.sample_sets, self._config, r.clusters, r.centroids)

    def classify(self):
        if len(self._data.index.unique('Sample')) <= self._config.samples:
            self._logger.warning("The total number of samples is lower than or equal to the sample size. " +
                                 "Running random forest classifier over fully clustered samples " +
                                 "does not make much sense.")
            if self._config.repeats > 1:
                self._logger.warning("Setting the repeats to more than 1 does not make much sense.")
        self._find_centroids()
        self._cluster_centroids()
        self._random_forest()
        r = self._result

        if len(self._config.clusters) > 1:
            cls = MultiClusteringResult
        else:
            cls = ClusteringResult

        return cls(r.cells, r.sample_sets, self._config, r.clusters, r.centroids,
                   self._training[~self._training.index.isin(self._test)],
                   self._training[self._training.index.isin(self._test)],
                   r.performance)

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
        samples_lv = self._training.index.get_level_values('Sample')
        genes_lv = self._training.index.get_level_values('Gene')
        samples = list(pd.DataFrame(np.stack([samples_lv, genes_lv], axis=-1)).drop_duplicates().values.tolist())
        total_genes = len(genes_lv.unique())
        unique = unique and not self._is_one_gene(samples)
        selected = []
        genes = []
        while len(selected) < self._config.samples:
            if (len(samples) == 0) or (unique and (len(selected) == total_genes)):
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
        set_id = hashlib.md5(str(samples).encode()).hexdigest()[:8]
        self._logger.debug("Clustering sample set " + str(set_id) + ": ")
        self._logger.debug("Samples: " + str(samples))
        mask = cells.index.get_level_values('Sample').isin(samples)
        cells = self._training[mask]
        x = stats.zscore(cells[c.hc_features].values, axis=0)
        tmp_dir = tempfile.TemporaryDirectory()
        ac = AgglomerativeClustering(affinity=c.metric, linkage=c.method, compute_full_tree=True, memory=tmp_dir.name)
        clusters = c.clusters.copy()
        # Compute clustering for first n
        n = clusters.pop(0)
        y0 = self._ac_predict_job(ac, n, x)
        # Compute clustering for the remaining ns
        if clusters:
            params = zip([ac for _ in clusters], clusters, [x for _ in clusters])
            with Pool(max(multiprocessing.cpu_count() - 1, 1)) as p:
                yn = p.starmap(Clustering._ac_predict_job, params)
            y = np.stack([y0] + yn, axis=-1)
        else:
            y = y0
        tmp_dir.cleanup()
        # Collect clustering results into a DataFrame
        li = ['Sample set']
        clustering = pd.DataFrame(
                y, index=cells.index, columns=pd.MultiIndex.from_product(
                    [['Agglomerative cluster'], [c.method], c.clusters])) \
            .assign(**{'Sample set': set_id}) \
            .set_index(li, append=True) \
            .reorder_levels(li + cells.index.names)
        return set_id, clustering, samples

    @classmethod
    def _ac_predict_job(cls, ac, n, x):
        cls._logger.debug('Computing clusters for n=' + str(n))
        ac.set_params(n_clusters=n)
        ac.fit_predict(x)
        return ac.labels_ + 1

    def _find_centroids(self):
        c = self._config
        if self._result is None:
            self._logger.debug('Creating new results object...')
            self._result = self.Result()
        r = self._result
        self._logger.info('Computing agglomerative clustering')
        for i in range(0, c.repeats):
            self._logger.debug('Iteration ' + str(i+1) + ': computing ' + str(c.method))
            try:
                set_id, clusters, samples = self._cluster()
                r.clusters = r.clusters.append(clusters)
                r.sample_sets[set_id] = samples
            except Exception as e:
                self._logger.warning('Computing ' + str(c.method) + ' failed', exc_info=e)
        for n in c.clusters:
            li = ['Clustering', 'Method', 'Clusters']
            by = ['Sample set', ('Agglomerative cluster', c.method, n)]
            centroids = r.clusters.join(self._training[self._config.hc_features], on=self._training.index.names) \
                .groupby(by)[self._config.hc_features] \
                .mean() \
                .assign(**{
                    'Clustering': 'agglomerative',
                    'Method': c.method,
                    'Clusters': n}) \
                .set_index(li, append=True) \
                .reorder_levels(by[:1] + li + by[1:])
            centroids.index = centroids.index.rename('Cluster', level=(len(li) + 1))
            if r.centroids is None:
                r.centroids = centroids
            else:
                r.centroids = r.centroids.append(centroids)
        r.centroids = r.centroids.sort_index()
        r.status = 'found_centroids'
        if c.repeats == 1 and len(self._data.index) == len(r.clusters.index):
            r.cells = self._training.copy()
            if 'Cluster' in r.cells.columns.unique(0):
                r.cells = r.cells.drop(columns='Cluster', level=0)
            r.cells = r.cells.join(r.clusters.droplevel('Sample set')) \
                             .rename(columns={'Agglomerative cluster': 'Cluster'})

    def _cluster_centroids(self):
        c = self._config
        r = self._result
        if r is None or r.status != 'found_centroids':
            raise RuntimeError("_cluster_centroids method must be run immediately after _find_centroids")

        params = zip([r.centroids for _ in c.clusters], [r.sample_sets for _ in c.clusters],
                     [c for _ in c.clusters], c.clusters)
        with Pool(max(multiprocessing.cpu_count() - 1, 1)) as p:
            centroids = p.starmap(Clustering._cluster_controids_job, params)

        cen = r.centroids[c.hc_features].join(pd.concat(centroids)[['Global cluster', 'Distance']])
        r.centroids = cen.dropna() \
                         .groupby(['Clustering', 'Method', 'Clusters', 'Global cluster'])[c.hc_features] \
                         .mean() \
                         .reset_index() \
                         .rename(columns={'Global cluster': 'Cluster'}) \
                         .assign(**{'Sample set': 'global', 'Clustering': 'centroids',
                                    'Global cluster': np.nan, 'Distance': np.nan}) \
                         .set_index(cen.index.names) \
                         .append(cen) \
                         .sort_index()

        for n in c.clusters:
            r.clusters = r.clusters.join(
                cen.loc[pd.IndexSlice[:, :, :, n], 'Global cluster']
                   .droplevel(['Clustering', 'Method', 'Clusters'])
                   .rename(('Centroid cluster', 'ward', n)),
                on=['Sample set', ('Agglomerative cluster', 'ward', n)])

        r.status = 'clustered_centroids'

    def _label_clusters(self):
        c = self._config
        r = self._result
        clusters = self._create_labels()
        cells = self._data.copy()
        for i, n in enumerate(c.clusters):
            cells = cells.join(clusters[i], on=['Sample', 'Nucleus'], how='inner')
        r.cells = cells

    @classmethod
    def _cluster_controids_job(cls, cs, ss, c, n):
        cls._logger.info('Clustering controids (n=' + str(n) + ')')
        idx = pd.IndexSlice
        cols = ['Global cluster', 'Distance']
        cen = cs.loc[idx[:, 'agglomerative', c.method, n], :] \
                .copy() \
                .assign(**dict(zip(cols, [np.nan for _ in cols])))
        cen[cen.columns] = stats.zscore(cen)
        i = 0
        s = cen.index.levels[0][0]
        last = np.zeros(cen.loc[(s, 'agglomerative', c.method, n), c.hc_features].values.shape)
        prev = last
        score = np.inf
        while score > c.MIN_SCORE and i < c.ITER_MAX:
            if i == 0:
                sample_sets = list(ss.keys())
                index = np.random.randint(0, len(sample_sets))
                s = sample_sets.pop(index)
                cls._logger.info('Initiating (n=' + str(n) + ') with sample set ' + s)
                last = cen.loc[(s, 'agglomerative', c.method, n), c.hc_features].values
                samples = 1
                new = last.copy()
            else:
                samples = 0
                new = np.zeros(last.shape)
                last = prev
            i = i + 1
            sample_sets = list(ss.keys())
            while len(sample_sets) > 0:
                index = np.random.randint(0, len(sample_sets))
                s = sample_sets.pop(index)
                current = cen.loc[(s, 'agglomerative', c.method, n), c.hc_features].values
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
                            cen.loc[(s, 'agglomerative', c.method, n, pairs[i] + 1), cols] = \
                                [(i + 1), distances[pairs[i], i]]
                        else:
                            new[pairs[i]] = new[pairs[i]] + current[i]
                            cen.loc[(s, 'agglomerative', c.method, n, i + 1), cols] = \
                                [(pairs[i] + 1), distances[i, pairs[i]]]
                    if i == 0:
                        last = new / samples
                else:
                    cls._logger.info('Ambiguous distances (n=' + str(n) + ') for set ' + s)
            last = new / samples
            score = np.sum(np.abs(prev - last))
            prev = last
        cls._logger.info('Done (n=' + str(n) + ') in ' + str(i) + ' iterations. Score is ' + str(score))
        return cen

    def _create_labels(self):
        c = self._config
        r = self._result
        params = zip([r.clusters for _ in c.clusters], [r.centroids for _ in c.clusters],
                     [c for _ in c.clusters], c.clusters)
        with Pool(max(multiprocessing.cpu_count() - 1, 1)) as p:
            clusters = p.starmap(Clustering._labels_job, params)

        return clusters

    def _random_forest(self):
        c = self._config
        r = self._result
        if r is None or r.status != 'clustered_centroids':
            raise RuntimeError("_random_forest method must be run after _cluster_centroids")

        self._logger.info('Computing random forest classification')

        clusters = self._create_labels()
        cells = self._data.copy()

        # Exclude test data from training dataset for overfit estimation
        training = self._training.loc[~self._training.index.isin(self._test)]
        test = self._training.loc[self._training.index.isin(self._test)]

        for i, n in enumerate(c.clusters):
            training = training[c.rf_features].join(clusters[i], on=['Sample', 'Nucleus'], how='inner')
            test = test[c.rf_features].join(clusters[i], on=['Sample', 'Nucleus'], how='inner')
            rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
            rf.fit(training[c.rf_features], training[('Centroid cluster', c.method, n)])
            self._logger.debug('Computing predictions (n=' + str(n) + ')')
            cells[('Cluster', c.method, n)] = 0
            cells[('Cluster', c.method, n)] = rf.predict(self._data[c.rf_features])
            b = np.array(training.loc[:, ('Centroid cluster', c.method, n)] ==
                         cells.loc[training.index, ('Cluster', c.method, n)]).astype(int)
            train_performance = float(np.sum(b) / b.size)
            self._logger.debug('Training set performance: ' + str(train_performance))
            b = np.array(test.loc[:, ('Centroid cluster', c.method, n)] ==
                         cells.loc[test.index, ('Cluster', c.method, n)]).astype(int)
            test_performance = float(np.sum(b) / b.size)
            self._logger.debug('Test set performance: ' + str(test_performance))
            if train_performance - test_performance > 0.05:
                self._logger.warning('Possible model overfit')
            r.performance[n] = {'train': train_performance, 'test': test_performance}

        r.cells = cells
        r.status = 'random_forest'
        self._logger.info('Predictions done')

    @classmethod
    def _labels_job(cls, clusters, centroids, c, n):

        def cluster_mode(series):
            try:
                return statistics.mode(series.tolist())
            except statistics.StatisticsError:
                return 0

        cls._logger.debug('Generating training dataset for classifier (n=' + str(n) + ')...')
        idx = pd.IndexSlice
        clusters = clusters \
            .loc[:, [('Centroid cluster', c.method, n), ('Agglomerative cluster', c.method, n)]] \
            .dropna()

        if c.cutoff != -1.0:
            clusters = clusters.join(
                    centroids.loc[idx[centroids['Distance'] < c.cutoff, :, :, n], 'Distance']
                             .droplevel(['Clustering', 'Method', 'Clusters'])
                             .rename(('Distance', None, None)),
                    on=['Sample set', ('Agglomerative cluster', c.method, n)], how='right') \
                .drop(columns=('Distance', None, None))

        return clusters.groupby(['Sample', 'Nucleus'])[[('Centroid cluster', 'ward', n)]].agg(cluster_mode)


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
