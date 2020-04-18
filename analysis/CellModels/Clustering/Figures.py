import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.cluster.hierarchy import dendrogram

from CellModels.Cells.Figures import DiscPlot, Figure, LogScaleExtPlot, SamplePlot


class ClusterPlot(DiscPlot, LogScaleExtPlot):

    def __init__(self, data, sample=None, column=None, cells=None):
        super().__init__(data, sample, cells)
        if column is None:
            self._column = self._data.get_cluster_columns()[0]
        else:
            self._column = column
        sort = self._cells.groupby(self._column)[[self._column]].transform(len).sort_values(self._column,
                                                                                            ascending=False).index
        self._cells = self._cells.loc[sort]

    def _get_c(self):
        return (self._cells[self._column] - 1) / 10

    def _get_sc_params(self):
        return {
            'cmap': 'tab10',
            's': 5,
            'vmin': 0,
            'vmax': 1
        }

    def plot_legend(self, ax, loc='center'):
        handles = []
        n = int(max(self._cells[self._column].unique()))
        for i in range(n):
            handles.append(Line2D([0], [0], marker='o', color='C' + str(i), markersize=10, lw=0))
        labels = [('Cluster ' + str(i + 1)) for i in range(n)]
        ax.legend(handles, labels, frameon=False, ncol=3, loc=loc)


class CentroidPlot(DiscPlot, LogScaleExtPlot):

    def __init__(self, data, sample=None, index=0, features=None, cells=None, prev=True):
        super().__init__(data, sample, cells)
        if features is None:
            self._features = self._data.config.rf_features
        else:
            self._features = features
        self._columns = self._data.get_cluster_columns()
        if str(index).isnumeric() and index < len(self._columns):
            self._index = index
        else:
            self._index = self._columns.index(index)
        for c in self._features:
            if 'Measurements' in c and 'Normalized' in c:
                self._field = c
        self._ax = None
        self._sc0 = None
        self._sc1 = None
        self._prev = prev

    def plot(self):
        ax = super().plot()
        self.plot_colorbar()
        return ax

    def plot_axes(self, ax=None):
        if ax is None:
            ax = self._get_ax(rect=[0, 0, 1, 1])
        if self._prev:
            if self._index > 0:
                self._sc0 = self._centroid_scatter(ax, self._columns[self._index - 1], 0.2)
            else:
                self._sc0 = self._centroid_scatter(ax, None, 0.2)
        self._sc1 = self._centroid_scatter(ax, self._columns[self._index])
        ax.set_ylabel('A-P position')
        ax.set_xlabel(str(self._field[-1]) + ' expression')
        ax.set_ylim(self.GY_MAX, self.GY_MIN)
        ax.set_xlim(*self.v_lim())
        ax.set_xscale(self.v_scale())
        ax.set_xticks(self.v_ticks())
        ax.set_xticklabels(self.v_ticks())
        self._ax = ax
        return ax

    def plot_colorbar(self, cax=None, orientation='vertical', fig=None):
        fig = self._fig if fig is None else fig
        ax = self._ax if cax is None else None
        cb = fig.colorbar(self._sc1[1], cax=cax, ax=ax, orientation=orientation,
                          ticks=self.v_ticks(), format=self.major_formatter_log)
        cb.set_label(label=str(self._field[-1]) + ' prominence')

    def plot_legend(self, ax, full=True, loc='center'):
        handles = []
        if full:
            n = int(max(self._cells[self._columns[self._index]].unique()))
            for i in range(n):
                handles.append(Line2D([0], [0], marker='o', color='C' + str(i), markersize=10, lw=0))
            labels = [('Cluster ' + str(i + 1)) for i in range(n)]
            ncol = 3
        else:
            labels = []
            ncol = 2
        handles.append(Line2D([0], [0], marker='o', color='k', markersize=10, lw=0))
        handles.append(Line2D([0], [0], marker='o', color='k', markersize=10, lw=0, alpha=0.2))
        handles.append(Line2D([0], [0], marker='o', color='y', markersize=5, lw=0))
        labels.append('Cluster')
        labels.append('Previous cluster')
        labels.append(str(self._field[-1]) + ' prominence')
        ax.legend(handles, labels, frameon=False, ncol=ncol, loc=loc)

    def _centroid_scatter(self, ax, column, alpha=1.0):
        def find_field(a, b):
            for f in self._features:
                if a in f and b in f:
                    return f

        if column is None:
            centroids = self._cells[self._features].mean()
            x = [centroids[find_field('Measurements', 'Normalized')]]
            y = [centroids[find_field('Position', 'y')]]
            c = 'k'
            k = [centroids[find_field('Measurements', 'Prominence')]]
            s = len(self._cells.index)
        else:
            centroids = self._cells.groupby(column)[self._features].mean().sort_values(by=column).assign(
                Count=self._cells.groupby(column)[self._features].size())
            x = centroids[find_field('Measurements', 'Normalized')]
            y = centroids[find_field('Position', 'y')]
            c = (centroids.index - 1) / 10
            k = centroids[find_field('Measurements', 'Prominence')]
            s = centroids['Count']
        sc0 = ax.scatter(x, y, c=c, cmap='tab10', s=100 + s, vmin=0, vmax=1, alpha=alpha)
        sc1 = ax.scatter(x, y, c=k, norm=colors.LogNorm(*self.v_lim()),
                         cmap=self.cmap(), s=20, alpha=alpha)
        return sc0, sc1


class Dendrogram(Figure):

    def __init__(self, data, sample=None, features=None, cells=None):
        super().__init__(data)
        if features is None:
            features = self._data.config.rf_features if features is None else features
        if cells is None:
            if sample is None:
                cells = self._data.cells
            else:
                idx = pd.IndexSlice
                cells = self._data.cells.loc[idx[:, sample], :]
        self._linkage = self._data.linkage(cells, features)

    def plot_axes(self, ax=None):
        if ax is None:
            ax = self._get_ax(rect=[0, 0, 1, 1])
        dn = dendrogram(self._linkage, ax=ax, leaf_label_func=lambda l: str(int(l) + 1),
                        leaf_font_size=ax.xaxis.get_label().get_fontsize())
        lookup = {}
        c_dict = {}

        for i, z in enumerate(self._linkage):
            r = i + self._linkage.shape[0] + 1
            j = lookup[z[0]] if z[0] in lookup.keys() else z[0]
            k = lookup[z[1]] if z[1] in lookup.keys() else z[1]
            lookup[r] = min(j, k)
            if i < self._linkage.shape[0] - 1:
                c_dict[z[2]] = 'C' + str(int(lookup[r]))
            else:
                c_dict[z[2]] = 'k'

        for i, d, c in zip(dn['icoord'], dn['dcoord'], dn['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            ax.plot(x, y, 'o', c=c_dict[y], markersize=10)

        for i, x in zip(dn['leaves'], ax.get_xticks()):
            ax.plot(x, 0, 'o', c='C' + str(i), markersize=10)

        ax.set_ylabel('Distance')
        ax.set_xlabel('Cluster')

        return ax


class MultiClusteringPlot(Figure):

    def __init__(self, data, sample=None, features=None, field=None):
        super().__init__(data)
        if features is None:
            self._features = self._data.config.rf_features if features is None else features
        if sample is None:
            self._cells = self._data.cells
        else:
            if sample == 'best':
                sample = self._data.representative_sample()
            idx = pd.IndexSlice
            self._cells = self._data.cells.loc[idx[:, sample], :]
        self._sample = sample
        if field is None:
            for c in self._features:
                if 'Measurements' in c and 'Normalized' in c:
                    self._field = c
        else:
            self._field = field
        self._linkage = self._data.linkage(self._cells, self._features)
        self._columns = self._data.get_cluster_columns(self._cells)

    def plot_axes(self, ax=None):
        if ax is not None:
            raise ValueError("Cannot specify axis for plot with subplots")
        r = []
        gs = self._fig.add_gridspec(len(self._columns) + 2, 2)

        sp = SamplePlot(self._data, self._sample, self._field, self._cells)
        r.append(sp.plot_axes(self._fig.add_subplot(gs[0, 0])))

        dn = Dendrogram(self._data, self._sample, self._features, self._cells)
        r.append(dn.plot_axes(self._fig.add_subplot(gs[0, 1])))

        lp = None
        cp = None
        for i, column in enumerate(self._columns):
            lp = ClusterPlot(self._data, self._sample, column, self._cells)
            r.append(lp.plot_axes(self._fig.add_subplot(gs[i + 1, 0])))
            cp = CentroidPlot(self._data, self._sample, i, self._features, self._cells)
            r.append(cp.plot_axes(self._fig.add_subplot(gs[i + 1, 1])))

        ax = self._fig.add_subplot(gs[-1, 0])
        ax.set_axis_off()
        cax = inset_axes(ax, width="100%", height="10%", loc='upper left')
        sp.plot_colorbar(cax, 'horizontal', self._fig)
        lp.plot_legend(ax)
        r.append(ax)

        ax = self._fig.add_subplot(gs[-1, 1])
        ax.set_axis_off()
        cax = inset_axes(ax, width="100%", height="10%", loc='upper left')
        cp.plot_colorbar(cax, 'horizontal', self._fig)
        cp.plot_legend(ax, full=False)

        ax = self._fig.add_subplot(gs[-1, 0:])
        ax.set_axis_off()
        cax = inset_axes(ax, width="100%", height="15%", loc='lower left')
        text = 'Sample single-gene clustering plots: ' + str(
            self._cells.index.unique(0)[0]) + ', sample ' + self._sample
        cax.text(0.5, 1, text, size='xx-large', ha='center')
        cax.set_axis_off()
        r.append(ax)

        return r

    def _size(self):
        return 15, 5 * (len(self._columns) + 2)


class GeneClusteringPlot(Figure):

    def __init__(self, data, features=None, field=None):
        super().__init__(data)
        if features is None:
            self._features = self._data.config.rf_features if features is None else features
        self._cells = self._data.cells
        if field is None:
            for c in self._features:
                if 'Measurements' in c and 'Normalized' in c:
                    self._field = c
        else:
            self._field = field
        self._columns = self._data.get_cluster_columns(self._cells)
        self._samples = self._cells.index.unique('Sample')
        self._best_clustering, self._m_dist = self._data.best_clustering()

    def plot_axes(self, ax=None):
        if ax is not None:
            raise ValueError("Cannot specify axis for plot with subplots")
        r = []
        gs = self._fig.add_gridspec(len(self._samples) + 2, 2)
        cp = CentroidPlot(self._data, None, self._best_clustering,
                          self._features, self._cells, False)
        cp_ax = cp.plot_axes(self._fig.add_subplot(gs[0, 0]))
        cp_ax.text(0.975, 0.915, "k = " + str(self._best_clustering[2]),
                   transform=cp_ax.transAxes, ha='right', size='large')
        r.append(cp_ax)

        dn = Dendrogram(self._data, None, self._features, self._cells)
        dn_ax = dn.plot_axes(self._fig.add_subplot(gs[0, 1]))
        dn_ax.hlines(self._m_dist, *dn_ax.get_xlim(), ls='dotted', alpha=0.5)
        r.append(dn_ax)

        lp = None
        for i, sample in enumerate(self._samples):
            idx = pd.IndexSlice
            cells = self._data.cells.loc[idx[:, sample], :]
            lp = ClusterPlot(self._data, sample, self._best_clustering, cells)
            lp_ax = lp.plot_axes(self._fig.add_subplot(gs[i + 1, 0]))
            lp_ax.text(0.025, 0.05, str(sample),
                       transform=lp_ax.transAxes, ha='left', size='xx-large')
            r.append(lp_ax)
            try:
                dn = Dendrogram(self._data, sample, self._features, cells)
                r.append(dn.plot_axes(self._fig.add_subplot(gs[i + 1, 1])))
            except KeyError:
                continue

        ax = self._fig.add_subplot(gs[-1, 0])
        ax.set_axis_off()
        cax = inset_axes(ax, width="100%", height="10%", loc='upper left')
        cp.plot_colorbar(cax, 'horizontal', self._fig)
        lp.plot_legend(ax)
        r.append(ax)

        ax = self._fig.add_subplot(gs[-1, 1])
        ax.set_axis_off()
        cp.plot_legend(ax, full=False)

        ax = self._fig.add_subplot(gs[-1, 0:])
        ax.set_axis_off()
        cax = inset_axes(ax, width="100%", height="15%", loc='lower left')
        text = 'Single-gene clustering plots: ' + str(self._cells.index.unique(0)[0])
        cax.text(0.5, 1, text, size='xx-large', ha='center')
        cax.set_axis_off()
        r.append(ax)

        return r

    def _size(self):
        return 15, 5 * (len(self._samples) + 2)
