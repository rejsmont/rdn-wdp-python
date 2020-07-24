#!/usr/bin/env python3

import logging
import multiprocessing
import tempfile

import numpy as np
import pandas as pd
from multiprocessing import Pool

from pandas.core.groupby import SeriesGroupBy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
import hashlib

from typing import Union

from CellModels.Cells.Data import Cells
from CellModels.Cells.Filters import Morphology
from CellModels.Clustering.Data import ClusteringResult, MultiClusteringResult, ClusteringConfig, SampleSets, \
    Performance


class Clustering:
    _logger = logging.getLogger('rdn-wdp-clustering')

    class Result:
        NEW = 0
        FOUND_CENTROIDS = 1
        CLUSTERED_CENTROIDS = 2
        LABELED_CLUSTERS = 3
        RANDOM_FOREST = 4

        def __init__(self):
            self.centroids = pd.DataFrame()
            self.clusters = pd.DataFrame()
            self.sample_sets = SampleSets()
            self.cells: pd.DataFrame = pd.DataFrame()
            self.status = self.NEW
            self.performance = Performance()

        def reset_status(self, status):
            if status < self.RANDOM_FOREST:
                self.performance = Performance()
                if 'Cluster' in self.cells.columns.get_level_values(0):
                    self.cells = self.cells.drop('Cluster', axis='columns')
                if 'Gene' in self.cells.columns.get_level_values(0):
                    self.cells = self.cells.drop('Gene', axis='columns')
            if status < self.CLUSTERED_CENTROIDS:
                if 'Centroid cluster' in self.clusters.columns.get_level_values(0):
                    self.clusters = self.clusters.drop('Centroid cluster', axis='columns')
                if 'Global cluster' in self.centroids.columns.get_level_values(0):
                    self.centroids = self.centroids.drop('Global cluster', axis='columns')
                if 'Distance' in self.centroids.columns.get_level_values(0):
                    self.centroids = self.centroids.drop('Distance', axis='columns')
                if 'global' in self.centroids.index.get_level_values(0):
                    self.centroids = self.centroids.drop('global')
            if status < self.FOUND_CENTROIDS:
                self.centroids = pd.DataFrame()
                self.clusters = pd.DataFrame()
                self.sample_sets = SampleSets()
                self.cells = pd.DataFrame()
            self.status = status

        def init_status(self):
            if 'Cluster' in self.cells.columns.get_level_values(0):
                if self.performance:
                    self.status = self.RANDOM_FOREST
                else:
                    self.status = self.CLUSTERED_CENTROIDS
            elif len(self.centroids.index) > 0:
                self.status = self.FOUND_CENTROIDS
            else:
                self.status = self.NEW

    def __init__(self, d: Union[Cells, ClusteringResult], c: Union[ClusteringConfig, None] = None,
                 training: Union[pd.Index, None] = None, test: Union[pd.Index, None] = None,
                 rf_parallel: bool = True, jobs: int = 0):
        if isinstance(d, Cells) and c is not None:
            fields = c.hc_features + c.rf_features
            # Make sure that data does not contain NaNs in the fields used for clustering
            self._data = d.loc[d[fields].dropna().index]
            self._config = c
            self._result = Clustering.Result()
        elif isinstance(d, ClusteringResult):
            self._data = d.cells
            if c is not None:
                self._config = c
            else:
                self._config = d.config
            self._result = Clustering.Result()
            self._result.sample_sets = d.sample_sets
            self._result.centroids = d.centroids
            self._result.clusters = d.clusters
            self._result.cells = d.cells
            self._result.performance = d.performance
            self._result.init_status()
        else:
            raise ValueError("Both d:Cells and c:ClusteringConfig must be specified.")
        if training is None:
            self._training = self._data.index
        else:
            self._training = self._data.loc[training].index
        if test is None:
            # Use 10% of data for performance testing
            self._test = self._data.loc[self._training].sample(frac=0.1).sort_index().index
            self._training = self._training.drop(self._test)
        else:
            self._test = self._data.loc[test].index
        self._rf_parallel = rf_parallel
        self._jobs = jobs if jobs > 0 else max(multiprocessing.cpu_count() - 1, 1)

    def cluster(self):
        if len(self._data.index.unique('Sample')) <= self._config.samples and self._config.repeats > 1:
            self._logger.warning("The total number of samples is lower than or equal to the sample size. " +
                                 "Setting the repeats to more than 1 does not make much sense.")
        elif self._config.repeats > 1:
            self._logger.warning("The repeats are set to more than 1. Consider using the classify method.")

        if self._result.status < self.Result.FOUND_CENTROIDS:
            self._find_centroids()
        if self._config.repeats > 1 and self._result.status < self.Result.FOUND_CENTROIDS:
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

    def reset(self, status):
        if status > self._result.status:
            raise ValueError("Cannot reset computation status to a higher level.")
        self._result.reset_status(status)

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
            with Pool(self._jobs) as p:
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
        r = self._result
        self._logger.info('Computing agglomerative clustering...')
        for i in range(0, c.repeats):
            self._logger.debug('Iteration ' + str(i + 1) + ': computing ' + str(c.method))
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
        r.status = r.FOUND_CENTROIDS
        if c.repeats == 1 and len(self._data.index) == len(r.clusters.index):
            r.cells = self._training.copy()
            if 'Cluster' in r.cells.columns.unique(0):
                r.cells = r.cells.drop(columns='Cluster', level=0)
            r.cells = r.cells.join(r.clusters.droplevel('Sample set')) \
                .rename(columns={'Agglomerative cluster': 'Cluster'})

    def _cluster_centroids(self):
        c = self._config
        r = self._result
        if r is None or r.status != self.Result.FOUND_CENTROIDS:
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

        r.status = self.Result.CLUSTERED_CENTROIDS

    def _label_clusters(self):
        c = self._config
        r = self._result
        clusters = self._create_labels()
        cells = self._data.copy()
        for i, n in enumerate(c.clusters):
            cells = cells.join(clusters[i], on=['Sample', 'Nucleus'], how='inner')
        r.cells = cells
        r.status = self.Result.LABELED_CLUSTERS

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
                _, pairs = linear_sum_assignment(distances)
                samples = samples + 1
                for i in range(0, new.shape[0]):
                    new[i] = new[i] + current[pairs[i]]
                    cen.loc[(s, 'agglomerative', c.method, n, pairs[i] + 1), cols] = \
                        [(i + 1), distances[pairs[i], i]]
                if i == 0:
                    last = new / samples
            last = new / samples
            score = np.sum(np.abs(prev - last))
            prev = last
        cls._logger.info('Done (n=' + str(n) + ') in ' + str(i) + ' iterations. Score is ' + str(score))
        return cen

    def _create_labels(self):
        c = self._config
        r = self._result
        p = self._rf_parallel

        groups = [r.clusters[('Centroid cluster', c.method, n)] for n in c.clusters]
        if p:
            with Pool(self._jobs) as p:
                m = p.map(Clustering._label_consensus_job, groups)
        else:
            m = []
            for i in range(len(groups)):
                m.append(Clustering._label_consensus_job(groups[i]))

        return self._radius_filter(pd.concat(m, axis=1).dropna())

    def _radius_filter(self, b):
        c = self._config
        r = self._result
        f = list(set(c.hc_features + c.rf_features))
        d = self._data.loc[self._training, f].join(b, on=['Sample', 'Nucleus'], how='inner')

        if c.cutoff > 0:
            groups = [d[c.hc_features + [('Centroid cluster', 'ward', n)]].join(
                r.centroids.loc[('global', 'centroids', 'ward', n), c.hc_features],
                on=[('Centroid cluster', 'ward', n)], rsuffix='_centroid') for n in c.clusters]
            z = None
            for g in groups:
                c_features = [tuple([x[i] if i > 0 else x[i] + '_centroid' for i in range(len(x))])
                              for x in c.hc_features]
                dists = np.linalg.norm(g[c.hc_features].values - g[c_features].values, axis=1)
                t = dists < c.cutoff
                if z is None:
                    z = t
                else:
                    z = z & t
            return d[z]
        else:
            return d

    def _random_forest(self):
        c = self._config
        r = self._result
        if r is None or r.status != self.Result.CLUSTERED_CENTROIDS:
            raise RuntimeError("_random_forest method must be run after _cluster_centroids")

        self._logger.info('Computing random forest classification')

        clusters = self._create_labels()
        cells = self._data.copy()

        training = self._data.loc[self._training, c.rf_features].join(clusters, on=['Sample', 'Nucleus'], how='inner')
        test = self._data.loc[self._test, c.rf_features].join(clusters, on=['Sample', 'Nucleus'], how='inner')

        for n in c.clusters:
            rf = RandomForestClassifier(n_estimators=1000, n_jobs=self._jobs)
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
        r.status = self.Result.RANDOM_FOREST
        self._logger.info('Predictions done')

    @classmethod
    def _label_consensus_job(cls, d):

        # def slow_mode(g: SeriesGroupBy):
        #     def st_mode(series):
        #         try:
        #             return statistics.mode(series.tolist())
        #         except statistics.StatisticsError:
        #             return np.nan
        #
        #     return g.agg(st_mode)

        def mode(g: SeriesGroupBy):
            c = g.value_counts()
            b = c.index.names[:-1]
            f = c.index.names[-1]
            c.index.names = b + ['groupBy_mode_Value']
            x = c.to_frame('groupBy_mode_Count').reset_index()
            i = x.set_index(b).index
            p = x.pivot(columns='groupBy_mode_Value', values='groupBy_mode_Count').set_index(i).groupby(b).max()
            m = np.not_equal(np.sum(np.equal(p.values,
                                             np.nanmax(p.values, axis=1).reshape(p.values.shape[0], 1)),
                                    axis=1), 1)
            r = p.idxmax(axis='columns')
            r[m] = np.nan
            r.name = f
            return r

        return mode(d.groupby(['Sample', 'Nucleus']))


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
        post_mf = r.centroids.loc[idx['global', c.method, :]] \
            .loc[r.centroids['Name'] == 'post-MF'].index.get_level_values(2)
        r.cells.loc[r.cells['cy'] > Morphology.FURROW_BROAD.max, ['Cluster_' + c.method]] = post_mf
        pre_mf = r.centroids.loc[idx['global', c.method, :]] \
            .loc[r.centroids['Name'] == 'pre-MF'].index.get_level_values(2)
        r.cells.loc[r.cells['cy'] < Morphology.FURROW_BROAD.min, ['Cluster_' + c.method]] = pre_mf
