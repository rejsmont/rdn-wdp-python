import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance


class ClusteringTools:
    _cells = None

    @staticmethod
    def _cluster_columns(df, column='Cluster'):
        return sorted(df.xs(column, axis='columns', level=0, drop_level=False).columns)

    @classmethod
    def _cluster_table(cls, df, column='Cluster'):
        c = cls._cluster_columns(df, column)
        t = df.groupby(c, as_index=False).max()[c]
        if len(t.index) != t.iloc[:, -1].max():
            for i in range(len(c) - 1):
                cur = c[i]
                nxt = c[i + 1]
                u = t[t[cur] != t[nxt]]
                n = t.groupby(cur).size()
                f = (n.index.isin(u[cur])) & (n == 1)
                for v in n[f].index:
                    t.loc[t[cur] == v, nxt] = v
        return t

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
    def linkage(cls, df, features, rename=True, names=False):
        t = cls._cluster_table(df)
        columns = list(reversed(t.columns))
        centroids, linkage = [], []
        last = 0
        n_dict, lookup, counts, dists = {}, {}, {}, {}

        def update(rc, ix, la):
            if len(rc) <= 1:
                return la
            cen = centroids[ix].loc[rc, features].values
            l_lookup = dict(zip(range(len(cen)), rc))
            l_last = len(cen) - 1
            for oa, ob, dist, n in hierarchy.linkage(cen):
                a, b = lookup[l_lookup[oa]], lookup[l_lookup[ob]]
                if a == b:
                    continue
                dist += dists[a] + dists[b]
                count = counts[a] + counts[b]
                linkage.append([a - 1, b - 1, dist, count])
                l_last = l_last + 1
                l_lookup[l_last] = rc[int(oa)]
                la = la + 1
                lookup[l_lookup[oa]] = lookup[l_lookup[ob]] = la
                n_dict[la - 1] = min(l_lookup[oa], l_lookup[ob])
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
                if len(sup) == 2:
                    last = update(sup, i, last)
            else:
                last = d.loc[:, columns[i]].max()
                v = d.values.flatten()
                lookup = dict(zip(v, v))
                n_dict = dict(zip(v - 1, v))
                counts = dict(zip(v, [1 for _ in range(len(v))]))
                dists = dict(zip(v, [0 for _ in range(len(v))]))

        linkage = np.array(linkage)

        m_val = np.max(t.values) - 1
        if linkage.shape[0] < m_val:
            n_val = n_dict.values()
            n_keys = np.array(list(n_dict.keys()))
            for c, i in enumerate(np.setdiff1d(np.arange(m_val), linkage[:, 0:2])):
                lc = linkage[:, 0:2]
                lc[lc > (i - c)] -= 1
                linkage[:, 0:2] = lc
                n_keys[n_keys > (i - c)] -= 1
            n_dict = dict(zip(n_keys, n_val))

        if not rename:
            c0 = [n_dict[v] for v in linkage[:, 0]]
            c1 = [n_dict[v] for v in linkage[:, 1]]
            linkage[:, 0] = c0
            linkage[:, 1] = c1

        if names:
            return linkage, n_dict
        else:
            return linkage
