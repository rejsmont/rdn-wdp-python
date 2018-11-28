#!/usr/bin/env python3

import argparse
import logging
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import warnings
import hashlib
import statistics
import yaml
from data import DiscData


class Clustering:

    ITER_MAX = 1000
    MIN_SCORE = 1e-10
    C_FEATURES = ['cy', 'mCherry', 'ext_mCherry']
    FEATURES = ['cx', 'cy', 'cz', 'mCherry', 'ext_mCherry', 'ang_max_mCherry', 'Volume']

    _source = None
    clean = None
    train_clean = None
    data: DiscData = None
    cells: pd.DataFrame = None
    centroids: pd.DataFrame = None
    clusters: pd.DataFrame = None
    names: pd.DataFrame = None
    sample_sets = {}

    method = 'ward'
    metric = 'euclidean'
    k = 6
    n = 5
    r = 3
    cutoff = -1
    can_compute = False
    computed = False
    outdir = None
    prefix = None

    def __init__(self, data, disc_data=None, **kwargs):
        print("Input is", data)
        self.data = disc_data
        self.init_params(**kwargs)
        initialized = self.from_data(data) or self.from_csv(data) or self.from_yml(data)
        if not initialized:
            raise RuntimeError("Failed to initialize Clustering class")
        self.init_params(**kwargs)

    def from_csv(self, datafile):
        print("Trying from CSV", datafile)
        self._source = datafile
        try:
            return self.from_data(DiscData(datafile, try_metadata=False))
        except RuntimeError:
            return False

    def from_yml(self, datafile):
        print("Trying from YML", datafile)
        self._source = datafile
        with open(datafile, 'r') as stream:
            metadata = yaml.safe_load(stream)
            return self.from_metadata(metadata)

    def from_data(self, data):
        print("Trying from DiscData", data)
        if isinstance(data, DiscData):
            self.data = data
            self.can_compute = self.init_cells()
            return True
        return False

    def from_metadata(self, metadata):
        print("Trying from metadata")

        def explore(func, path, basedir=None, **kwargs):
            if basedir is None:
                paths = [path, os.path.basename(path)]
            else:
                path = os.path.basename(path)
                paths = [path, os.path.join(basedir, path), os.path.join(os.path.dirname(self._source), path)]
            result = None
            for path in paths:
                try:
                    print("Trying", path)
                    result = func(path, **kwargs)
                except:
                    continue
                break
            return result

        def valid(df):
            return df is not None and ((isinstance(df, pd.DataFrame) and not df.empty) or bool(df))

        if metadata is None:
            return False

        if self.data is None:
            self.data = explore(DiscData, metadata['input']['cells'], metadata['input']['dir'])
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            self.cells = explore(pd.read_csv, metadata['output']['cells'], metadata['output']['dir'], index_col=0)
            self.centroids = explore(pd.read_csv, metadata['output']['centroids'], metadata['output']['dir'],
                                     index_col=[0, 1, 2])
            self.clusters = explore(pd.read_csv, metadata['output']['clusters'], metadata['output']['dir'],
                                    index_col=[0, 1, 2])
        self.sample_sets = metadata['classification']['sets']
        params = {
            'outdir': metadata['output'].get('dir', None),
            'k': metadata['clustering'].get('clusters', None),
            'method': metadata['clustering'].get('method', None),
            'metric': metadata['clustering'].get('metric', None),
            'train_clean': metadata['clustering'].get('clean', None),
            'cutoff': metadata['classification'].get('cutoff', None),
            'n': metadata['classification'].get('samples', None),
            'r': metadata['classification'].get('repeats', None),
            'clean': metadata['classification'].get('clean', None)
        }
        self.init_params(**params)
        self.can_compute = self.init_cells()
        self.computed = \
            valid(self.cells) and valid(self.centroids) and valid(self.clusters) and valid(self.sample_sets)

        return self.can_compute or self.computed

    def init_cells(self, override=False):
        if (override or self.cells is None or self.cells.empty) and self.data is not None:
            if self.clean or self.clean is None:
                print("Will classify only clean cells...")
                self.cells = self.data.cells()[self.data.clean_mask()].dropna().copy()
            else:
                print("Will classify all cells...")
                self.cells = self.data.cells().dropna().copy()
            if override:
                self.computed = False
        if self.cells is not None and not self.cells.empty:
            return True
        return False

    def init_params(self, **kwargs):
        allowed = ['method', 'metric', 'k', 'n', 'r', 'cutoff', 'outdir', 'clean', 'train_clean']
        args = kwargs.pop('args', argparse.Namespace())
        for key in allowed:
            for value in [kwargs.get(key, None), getattr(args, key, None)]:
                if value is not None:
                    if key != 'outdir' and self.computed and getattr(self, key) != value:
                        self.computed = False
                    setattr(self, key, value)
                    break

    def reset(self):
        self.sample_sets = {}
        index = pd.MultiIndex.from_product([[], [], []], names=['SampleSet', 'Method', 'Cluster'])
        self.centroids = pd.DataFrame(columns=self.C_FEATURES, index=index)
        self.names = pd.DataFrame(columns=['Name'], index=index)
        index = pd.MultiIndex.from_product([[], [], []], names=['SampleSet', 'Method', 'Cell'])
        self.clusters: pd.DataFrame = pd.DataFrame(columns=['LocalCluster'], index=index).astype('int64')
        if self.cells is not None:
            self.cells['Cluster_' + self.method] = 0
        self.computed = False

    def compute(self):
        if not self.can_compute:
            raise RuntimeError("Nothing to compute")
        self.reset()

        def get_samples(n, unique=False):
            if self.train_clean:
                clean = self.cells['Gene'].isin(self.data.genes_clean())
                samples = self.cells.loc[clean, ['Sample', 'Gene']].drop_duplicates().values.tolist()
            else:
                samples = self.cells[['Sample', 'Gene']].drop_duplicates().values.tolist()
            selected = []
            genes = []
            while len(selected) < n:
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

        def cluster(samples=None, method=None, id=None):
            filer = (self.cells['cy'] >= DiscData.FURROW_MIN) & (self.cells['cy'] <= DiscData.FURROW_MAX)
            cells = self.cells.loc[filer]
            if method is None:
                method = self.method
            if samples is None:
                samples = sorted(cells['Sample'].unique().tolist())
            if id is None:
                id = hashlib.md5(str(samples).encode()).hexdigest()
            filter = cells['Sample'].isin(samples)
            x = stats.zscore(cells.loc[filter, self.C_FEATURES].values, axis=0)
            z = linkage(x, method)
            fclusters = fcluster(z, self.k, criterion='maxclust')
            index = pd.MultiIndex.from_product([[id], [method], cells.loc[filter].index.values],
                                               names=['SampleSet', 'Method', 'Cell'])
            self.clusters = self.clusters.append(pd.DataFrame(fclusters, columns=['LocalCluster'], index=index))
            self.sample_sets[id] = samples
            return z

        def find_centroids(method):
            for i in range(0, self.r):
                samples = get_samples(self.n)
                id = hashlib.md5(str(samples).encode()).hexdigest()
                print(i, id, samples)
                print("\tComputing ", method)
                try:
                    z = cluster(samples, method, id)
                except Exception as e:
                    print("Computing " + method + " for dataset " + id + " failed: " + str(e))
            self.centroids: pd.DataFrame = self.clusters.join(
                self.cells[self.C_FEATURES], on='Cell').groupby(
                ['SampleSet', 'Method', 'LocalCluster'])[self.C_FEATURES].mean()
            self.centroids[self.centroids.columns] = stats.zscore(self.centroids.values)
            self.centroids = self.centroids.sort_index()

        def cluster_centroids(method):

            def create_dataframe(array):
                centroids = pd.DataFrame(
                    array, columns=self.C_FEATURES,
                    index=range(1, array.shape[0] + 1)).rename_axis('LocalCluster')
                clusters = pd.DataFrame(
                    [['global', method, x, x, 0.0] for x in range(1, array.shape[0] + 1)],
                    columns=['SampleSet', 'Method', 'LocalCluster', 'Cluster', 'Distance']
                ).set_index(['SampleSet', 'Method', 'LocalCluster'])
                return clusters.join(centroids, on='LocalCluster').reindex(columns=self.centroids.columns)

            iters = 0
            last = np.zeros(self.centroids.loc[(self.centroids.index.levels[0][0], method), self.C_FEATURES].values.shape)
            prev = last
            score = np.inf
            while score > Clustering.MIN_SCORE and iters < Clustering.ITER_MAX:
                sample_sets = list(self.sample_sets.keys())
                if iters == 0:
                    index = np.random.randint(0, len(sample_sets))
                    last = self.centroids.loc[(sample_sets.pop(index), method), self.C_FEATURES].values
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
                    current = self.centroids.loc[(cid, method), self.C_FEATURES].values
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
                                self.centroids.loc[(cid, method, pairs[i] + 1),
                                                   ['Cluster', 'Distance']] = \
                                    [(i + 1), distances[pairs[i], i]]
                            else:
                                new[pairs[i]] = new[pairs[i]] + current[i]
                                self.centroids.loc[(cid, method, i + 1),
                                                   ['Cluster', 'Distance']] = \
                                    [(pairs[i] + 1), distances[i, pairs[i]]]
                        if iters == 0:
                            last = new / samples
                last = new / samples
                score = np.sum(np.abs(prev - last))
                prev = last
            print("Done in", iters, "iterations!", "Score is", score)
            self.centroids = self.centroids.append(create_dataframe(last))

        def random_forest(method, cutoff=-1.0):
            def cluster_mode(series):
                try:
                    return statistics.mode(series.tolist())
                except statistics.StatisticsError:
                    return 0

            print("Training classifier...")
            idx = pd.IndexSlice
            if cutoff == -1.0:
                clusters = self.clusters.loc[idx[:, method, :], :]
            else:
                clusters = self.clusters.loc[idx[:, method, :], :]\
                    .join(self.centroids.loc[self.centroids['Distance'] < cutoff, 'Distance'],
                          on=['SampleSet', 'Method', 'LocalCluster'], how='right')
            global_clusters = clusters.groupby('Cell')['Cluster'].agg(cluster_mode)
            global_clusters.drop(global_clusters[global_clusters == 0].index, inplace=True)
            rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
            rf.fit(self.cells.loc[global_clusters.index, self.FEATURES], global_clusters)
            print("Computing predictions...")
            self.cells['Cluster_' + method] = 0
            filter = (self.cells['cy'] >= DiscData.FURROW_MIN) & (self.cells['cy'] <= DiscData.FURROW_MAX)
            self.cells.loc[filter, ['Cluster_' + method]] = rf.predict(self.cells.loc[filter, self.FEATURES])
            print("Done.")

        def name_clusters():
            idx = pd.IndexSlice
            named_clusters = self.centroids.loc[idx['global', self.method, :], :]
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
            self.centroids = self.centroids.join(names.xs(('global', self.method)), on='Cluster')

        def classify_remaining():
            idx = pd.IndexSlice
            post_mf = self.centroids.loc[idx['global', self.method, :]]\
                .loc[self.centroids['Name'] == 'post-MF'].index.get_level_values(2)
            self.cells.loc[self.cells['cy'] > DiscData.FURROW_MAX, ['Cluster_' + self.method]] = post_mf
            pre_mf = self.centroids.loc[idx['global', self.method, :]] \
                .loc[self.centroids['Name'] == 'pre-MF'].index.get_level_values(2)
            self.cells.loc[self.cells['cy'] < DiscData.FURROW_MIN, ['Cluster_' + self.method]] = pre_mf

        find_centroids(self.method)
        self.centroids['Cluster'] = 0
        self.centroids['Distance'] = 0.0
        cluster_centroids(self.method)
        self.clusters = self.clusters.join(self.centroids['Cluster'], on=['SampleSet', 'Method', 'LocalCluster'])
        random_forest(self.method, cutoff=self.cutoff)
        name_clusters()
        classify_remaining()

    def base_filename(self):
        base_name = 'k' + str(self.k) + 'n' + str(self.n) + 'r' + str(self.r)
        if self.cutoff == -1:
            return str(self.prefix) + base_name
        else:
            return str(self.prefix) + 'c' + str(round(self.cutoff*100)) + base_name

    def clusters_filename(self):
        return self.base_filename() + '_' + 'clusters.csv'

    def centroids_filename(self):
        return self.base_filename() + '_' + 'centroids.csv'

    def cells_filename(self):
        return self.base_filename() + '_' + 'cells.csv'

    def meta_filename(self):
        return self.base_filename() + '_' + 'metadata.yml'

    def save(self, outdir=None):
        if outdir is None:
            outdir = '.' if self.outdir is None else self.outdir

        metadata = {
            'input': {
                'dir': os.path.dirname(self.data.source()),
                'cells': os.path.basename(self.data.source())
            },
            'output': {
                'dir': outdir,
                'clusters': self.clusters_filename(),
                'centroids': self.centroids_filename(),
                'cells': self.cells_filename()
            },
            'clustering': {
                'clean': self.train_clean or self.clean,
                'method': self.method,
                'metric': self.metric,
                'clusters': self.k,
            },
            'classification': {
                'clean': self.clean,
                'samples': self.n,
                'repeats': self.r,
                'cutoff': self.cutoff,
                'sets': self.sample_sets,
            },
        }
        with open(os.path.join(outdir, self.meta_filename()), 'w') as metafile:
            yaml.dump(metadata, metafile, default_flow_style=False)
        self.clusters.to_csv(os.path.join(outdir, self.clusters_filename()))
        self.centroids.to_csv(os.path.join(outdir, self.centroids_filename()))
        self.cells.to_csv(os.path.join(outdir, self.cells_filename()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot all data.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--log')
    parser.add_argument('--outdir')
    parser.add_argument('--prefix')
    parser.add_argument('-k', '--clusters', dest='k', type=int)
    parser.add_argument('-n', '--samples', dest='n', type=int)
    parser.add_argument('-r', '--repeats', dest='r', type=int)
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--reproducible', dest='reproducible', action='store_true')
    parser.add_argument('--not-reproducible', dest='reproducible', action='store_false')
    parser.add_argument('--clean', dest='clean', action='store_true')
    parser.add_argument('--train-clean', dest='train_clean', action='store_true')
    parser.set_defaults(reproducible=False)
    parser.set_defaults(clean=False)
    parser.set_defaults(train_clean=False)

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

    print(args)

    clustering = Clustering(args.data, args=args)
    clustering.compute()
    clustering.save(args.outdir)
