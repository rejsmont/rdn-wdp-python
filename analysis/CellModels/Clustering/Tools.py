import numpy as np
from scipy.spatial import distance


class ClusteringTools:
    _cells = None

    @staticmethod
    def _cluster_columns(df, column='Cluster'):
        return sorted(df.xs(column, axis='columns', level=0, drop_level=False).columns)

    @classmethod
    def _cluster_table(cls, df, column='Cluster'):
        columns = cls._cluster_columns(df, column)
        return df.groupby(columns, as_index=False).max()[columns]

    def get_cluster_columns(self, cells=None):
        if cells is None:
            cells = self._cells
        return self._cluster_columns(cells)


class MultiClusteringTools(ClusteringTools):

    @staticmethod
    def _centroids(df, column, features):
        return df.groupby(column)[features].mean().sort_values(by=column).assign(
            Count=df.groupby(column)[features].size()).rename_axis('Cluster', axis='index')

    @classmethod
    def _harmonize(cls, df, features):
        t = cls._cluster_table(df)
        nf = df.copy()
        columns = t.columns
        column = None
        h_column = None
        centroids = []
        for i, column in enumerate(columns):
            h_column = ('Harmonized ' + column[0], column[1], column[2])
            centroids.append(cls._centroids(df, column, features))
            c = [columns[i]] if i == 0 else [columns[i - 1], columns[i]]
            d = t.groupby(c, as_index=False).max()[c]
            if i > 0:
                last = d.loc[:, columns[i - 1]].max()
                for m in d.loc[:, columns[i - 1]].unique():
                    dist = []
                    for n in d.loc[d[columns[i - 1]] == m, columns[i]].unique():
                        dist.append((n, distance.euclidean(
                            centroids[i].loc[n, features],
                            centroids[i - 1].loc[m, features])))
                    for j, (n, _) in enumerate(sorted(dist, key=lambda x: x[1])):
                        if j == 0:
                            v = m
                        else:
                            last += 1
                            v = last
                        t.loc[t[column] == n, h_column] = v
                        nf.loc[df[column] == n, h_column] = v

                t[column] = t[h_column]
                t.drop(columns=h_column, inplace=True)
            else:
                nf[h_column] = df[column]
        return nf.drop(columns=cls._cluster_columns(nf)).rename(columns={h_column[0]: column[0]})

    @classmethod
    def linkage(cls, df, features, rename=True):

        t = cls._cluster_table(df)
        columns = list(reversed(t.columns))
        centroids = []
        linkage = []
        last = 0
        lookup = {}
        counts = {}
        dists = {}

        def update(rc, ix, la):
            if len(rc) == 2:
                a = lookup[rc[0]]
                b = lookup[rc[1]]
                dist = distance.euclidean(
                    centroids[ix].loc[rc[0], features],
                    centroids[ix].loc[rc[1], features]) + dists[a] + dists[b]
                count = counts[a] + counts[b]
                linkage.append([a - 1, b - 1, dist, count])
                la = la + 1 if rename else rc[0]
                lookup[rc[0]] = la
                counts[la] = count
                dists[la] = dist
            return la

        for i, column in enumerate(columns):
            centroids.append(cls._centroids(df, column, features))
            c = [columns[i]] if i == 0 else [columns[i - 1], columns[i]]
            d = t.groupby(c, as_index=False).max()[c]
            if i > 0:
                sup = d.loc[:, columns[i]].unique()
                for m in sup:
                    sub = d.loc[d[columns[i]] == m, columns[i - 1]].unique()
                    last = update(sub, i - 1, last)
                last = update(sup, i, last)
            else:
                last = d.loc[:, columns[i]].max()
                v = d.values.flatten()
                lookup = dict(zip(v, v))
                counts = dict(zip(v, [1 for _ in range(len(v))]))
                dists = dict(zip(v, [0 for _ in range(len(v))]))

        return np.array(linkage)
