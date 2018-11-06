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
from scipy import stats
import operator
import hdbscan
import hashlib


class Clustering(Figure):

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
        self.centroids = pd.DataFrame(columns=['cy', 'mCherry', 'ext_mCherry'])

    def compute(self):
        def h_cluster(x, method=None):
            if method is None:
                method = self.method

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
            x = stats.zscore(cells.loc[filter, ['cy', 'mCherry', 'ext_mCherry']].values, axis=0)
            z = linkage(x, method)
            cells.loc[filter, method + '_' + id] = fcluster(z, self.k, criterion='maxclust')
            index = pd.MultiIndex.from_product([[id], [method]], names=['SampleSet', 'Method'])
            samples = pd.DataFrame([samples], index=index, columns=['Sample_{0}'.format(s) for s in range(0, self.n)])
            self.sample_sets = self.sample_sets.append(samples)
            return z

        for i in range(0, self.r):
            samples = get_samples(self.n)
            id = hashlib.md5(str(samples).encode()).hexdigest()
            print(id, samples)
            for method in self.methods:
                print("\tComputing " + method)
                try:
                    z = cluster(samples, method, id)
                except Exception as e:
                    print("Computing " + method + " for dataset " + id + " failed: " + str(e))

        for sample_set, method in self.sample_sets.index.values:
            centroids = self.cells.groupby([method + '_' + sample_set])['cy', 'mCherry', 'ext_mCherry'].mean()
            index = pd.MultiIndex.from_product([[sample_set], [method], range(1, self.k + 1)],
                                               names=['SampleSet', 'Method', 'Cluster'])
            centroids = centroids.reindex(index, level=2)
            self.centroids = self.centroids.append(centroids, sort=False)

        print(self.sample_sets.index.get_level_values('SampleSet').values)



        fig = plt.figure(figsize=[5, 5])
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.centroids['cy'], self.centroids['mCherry'], c=self.centroids['ext_mCherry'])
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
                # max_d=p,              # plot distance cutoff line
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
parser.add_argument('--repeats', default=5)
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
