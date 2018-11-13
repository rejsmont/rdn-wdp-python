#!/usr/bin/env python3

from analysis.figures_ng import Figure, DiscData
import argparse
import logging
import os
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import operator
import hashlib
import statistics


class Clustering(Figure):

    ITER_MAX = 1000
    MIN_SCORE = 1e-10
    C_FEATURES = ['cy', 'mCherry', 'ext_mCherry']
    FEATURES = ['cx', 'cy', 'cz', 'mCherry', 'ext_mCherry', 'ang_max_mCherry', 'Volume']

    def __init__(self, data, method=None, metric='euclidean', k=6, n=5, r=3):
        super().__init__(data)
        self.cells = self.data.cells()[self.data.clean_mask() & self.data.furrow_mask()].dropna()
        self.method = method
        self.metric = metric
        self.k = k
        self.r = r
        self.n = n
        self.methods = [
            # 'single',
            # 'complete',
            # 'average',
            # 'weighted',
            # 'centroid',
            # 'median',
            'ward',
        ]
        index = pd.MultiIndex.from_product([[], []], names=['SampleSet', 'Method'])
        self.sample_sets = pd.DataFrame(columns=['Sample_{0}'.format(s) for s in range(0, self.n)], index=index)
        index = pd.MultiIndex.from_product([[], [], []], names=['SampleSet', 'Method', 'Cluster'])
        self.centroids = pd.DataFrame(columns=['cy', 'mCherry', 'ext_mCherry'], index=index)
        index = pd.MultiIndex.from_product([[], [], []], names=['SampleSet', 'Method', 'Cell'])
        self.clusters = pd.DataFrame(columns=['LocalCluster'], index=index).astype('int64')

    def compute(self):

        def get_samples(n, unique=False):
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
            cells = self.cells
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
            index = pd.MultiIndex.from_product([[id], [method], cells.loc[filter].index.values], names=['SampleSet', 'Method', 'Cell'])
            self.clusters = self.clusters.append(pd.DataFrame(fclusters, columns=['LocalCluster'], index=index))
            index = pd.MultiIndex.from_product([[id], [method]], names=['SampleSet', 'Method'])
            samples = pd.DataFrame([samples], index=index, columns=['Sample_{0}'.format(s) for s in range(0, self.n)])
            self.sample_sets = self.sample_sets.append(samples)
            return z

        def find_centroids():
            for i in range(0, self.r):
                samples = get_samples(self.n)
                id = hashlib.md5(str(samples).encode()).hexdigest()
                print(i, id, samples)
                for method in self.methods:
                    print("\tComputing ", method)
                    try:
                        z = cluster(samples, method, id)
                    except Exception as e:
                        print("Computing " + method + " for dataset " + id + " failed: " + str(e))
            self.centroids = self.clusters.join(
                self.cells[self.C_FEATURES], on='Cell').groupby(
                ['SampleSet', 'Method', 'LocalCluster'])[self.C_FEATURES].mean()
            self.centroids[self.centroids.columns] = stats.zscore(self.centroids.values)
            self.centroids = self.centroids.sort_index()

        def cluster_centroids(method):
            iters = 0
            last = np.zeros(self.centroids.loc[(self.centroids.index.levels[0][0], method), self.C_FEATURES].values.shape)
            prev = last
            score = np.inf
            while score > Clustering.MIN_SCORE and iters < Clustering.ITER_MAX:
                sample_sets = self.sample_sets.index.get_level_values('SampleSet').values.tolist()
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
            return last

        def random_forest(method):
            def cluster_mode(series):
                try:
                    return statistics.mode(series.tolist())
                except statistics.StatisticsError:
                    return 0

            print("Training classifier...")
            idx = pd.IndexSlice
            global_clusters = self.clusters.loc[idx[:, method, :], :].groupby('Cell')['Cluster'].agg(cluster_mode)
            global_clusters.drop(global_clusters[global_clusters == 0].index, inplace=True)
            # Random Forest: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
            rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
            # Train the model on training data
            rf.fit(self.cells.loc[global_clusters.index, self.FEATURES], global_clusters)
            print("Computing predictions...")
            self.cells['Cluster_' + method] = rf.predict(self.cells[self.FEATURES])
            print("Done.")

        def plot_centroids(method, centroids):
            idx = pd.IndexSlice
            cids = self.centroids.loc[idx[:, method], :]
            fig = plt.figure(figsize=[5, 5])
            ax = fig.add_subplot(1, 1, 1)
            clustered = cids.loc[cids['Cluster'] != 0]
            ax.scatter(clustered['cy'], clustered['mCherry'], c=clustered['Cluster'], s=160, cmap='Paired')
            ax.scatter(cids['cy'], cids['mCherry'], c=cids['ext_mCherry'])
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker='*')
            fig.show()

        find_centroids()

        # Clustering cluster centroids xD
        self.centroids['Cluster'] = 0
        self.centroids['Distance'] = 0.0
        for method in self.methods:
            centroids = cluster_centroids(method)
            plot_centroids(method, centroids)
        self.clusters = self.clusters.join(self.centroids['Cluster'], on=['SampleSet', 'Method', 'LocalCluster'])

        for method in self.methods:
            random_forest(method)
            s_centroids = self.cells.groupby(['Sample', 'Cluster_' + method])[self.C_FEATURES].mean()
            g_centroids = self.cells.groupby(['Cluster_' + method])[self.C_FEATURES].mean()
            fig = plt.figure(figsize=[5, 5])
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(s_centroids['cy'], s_centroids['mCherry'],
                       c=s_centroids.index.get_level_values('Cluster_' + method), s=160, cmap='Paired')
            ax.scatter(s_centroids['cy'], s_centroids['mCherry'], c=s_centroids['ext_mCherry'])
            ax.scatter(g_centroids['cy'], g_centroids['mCherry'], c='red', s=80, marker='*')
            fig.show()

    def plot(self, outdir):

        def sorted_legend(handles, labels=None):
            if labels is None:
                handles, labels = handles
            hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
            handles2, labels2 = zip(*hl)
            labels3 = [l.replace(' 0', ' ') for l in labels2]
            return handles2, labels3

        def h_cluster_dendrogram(z, ax, c_colors):
            dd = dendrogram(
                z,
                truncate_mode='lastp',  # show only the last p merged clusters
                p=self.k,               # show only the last p merged clusters
                leaf_rotation=90.,      # rotates the x axis labels
                leaf_font_size=8,       # font size for the x axis labels
                show_contracted=True,   # to get a distribution impression in truncated branches
                ax=ax,
                link_color_func=lambda c: 'black'
            )
            ax.set_xticklabels(['Cluster %d' % c for c in range(1, len(ax.get_xmajorticklabels()) + 1)])
            x_lbls = ax.get_xmajorticklabels()
            num = -1
            for lbl in x_lbls:
                num += 1
                lbl.set_color(c_colors(num))
            return dd

        # fig = plt.figure(figsize=[10, 10])
        # ax = fig.add_subplot(2, 2, 1)
        # ax_xi = fig.add_subplot(2, 2, 2)
        # ax_xy = fig.add_subplot(2, 2, 4)

        # cells.loc[cells.groupby('Cluster')['Cluster'].transform('count').sort_values(ascending=False).index]

        # clusters = cells.groupby('Cluster')['cx'].count().sort_values(ascending=False).index.tolist()
        # c_colors = plt.cm.get_cmap("gist_rainbow", len(clusters))
        # h_cluster_dendrogram(z, ax, c_colors)
        #
        # for cluster in clusters:
        #     if cluster == 0:
        #         continue
        #     c_cells = cells[cells['Cluster'] == cluster]
        #     c_count = c_cells['Cluster'].count()
        #     label = "Cluster " + '%02d' % cluster + " (" + str(c_count) + " cells)"
        #     ax_xi.scatter(c_cells['cy'], c_cells['mCherry'], c=[c_colors(cluster - 1)], label=label)
        #     ax_xy.scatter(c_cells['cx'], c_cells['cy'], c=[c_colors(cluster - 1)], label=label)
        #
        # handles, labels = sorted_legend(ax_xy.get_legend_handles_labels())
        # ax = fig.add_subplot(2, 2, 3)
        # ax.set_axis_off()
        # ax.legend(handles, labels, frameon=False, fontsize=15, loc='center')
        #
        # filename = id + '_' + method + '_' + str(self.k) + '.png'
        # fig.savefig(os.path.join(outdir, filename))
        # plt.close(fig)
        #
        # samples = cells['Sample'].unique().tolist()
        # for sample in samples:
        #     fig = plt.figure(figsize=[10, 10])
        #     ax_xyi = fig.add_subplot(2, 2, 1)
        #     ax_xi = fig.add_subplot(2, 2, 4)
        #     ax_xy = fig.add_subplot(2, 2, 2)
        #     s_cells = cells[cells['Sample'] == sample].sort_values('mCherry')
        #     ax_xyi.scatter(s_cells['cx'], s_cells['cy'], c=s_cells['mCherry'])
        #     for cluster in clusters:
        #         c_cells = s_cells[s_cells['Cluster'] == cluster]
        #         c_count = c_cells['Cluster'].count()
        #         label = "Cluster " + '%02d' % cluster + " (" + str(c_count) + " cells)"
        #         ax_xi.scatter(c_cells['cy'], c_cells['mCherry'], c=[c_colors(cluster - 1)], label=label)
        #         ax_xy.scatter(c_cells['cx'], c_cells['cy'], c=[c_colors(cluster - 1)], label=label)
        #     handles, labels = sorted_legend(ax_xy.get_legend_handles_labels())
        #     ax = fig.add_subplot(2, 2, 3)
        #     ax.set_axis_off()
        #     ax.legend(handles, labels, frameon=False, fontsize=15, loc='center')
        #     filename = id + '_' + method + '_' + str(self.k) + '_' + sample + '.png'
        #     fig.savefig(os.path.join(outdir, filename))
        #     plt.close(fig)


parser = argparse.ArgumentParser(description='Plot all data.')
parser.add_argument('--data', required=True)
parser.add_argument('--log')
parser.add_argument('--outdir')
parser.add_argument('--clusters', default=6)
parser.add_argument('--samples', default=5)
parser.add_argument('--repeats', default=3)
parser.add_argument('--reproducible', dest='reproducible', action='store_true')
parser.add_argument('--not-reproducible', dest='reproducible', action='store_false')
parser.set_defaults(reproducible=False)

args = parser.parse_args()

if args.log:
    logging.basicConfig(level=args.log.upper())
    logging.getLogger('PIL.Image').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    logging.getLogger('joblib').setLevel(logging.INFO)

data = DiscData(args)

# def run_process(option):
#     figure = Figure_9a76(data, gene=gene, sample='all', method=option, k=7)
#     try:
#         figure.plot(args.outdir)
#     except Exception as e:
#         print("Plotting " + figure.metric + " failed: " + str(e))
#
#
# if __name__ == '__main__':
#     with Pool(2) as p:
#         p.map(run_process, multiopt)

if args.reproducible:
    np.random.seed(0)
else:
    np.random.seed()

fig_9a76 = Clustering(data, k=args.clusters, n=args.samples, r=args.repeats)
fig_9a76.compute()
