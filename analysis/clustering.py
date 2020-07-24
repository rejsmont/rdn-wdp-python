#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import pandas as pd
from data import DiscData


class Clustering:
    pass


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
    parser.set_defaults(prefix='')

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

    clustering = Clustering(args.data, args=args)
    clustering.compute()
    clustering.save(args.outdir)


class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


class QCData:

    bad = [
            'hVJG7F', 't4JOuq', 'ylcNLd', '1NP0AI', '7SNHJY', 'BQ5PR8', 'DWYW27', 'I9BTQL', 'A5UIWC', 'KWGZHQ',
            'M6GF25', 'IJ5NUJ', 'WSSIG8', '3JR9EY', 'F7ZYLW', 'HDJ4XA', 'K2TCAP', 'OS7ENG', 'XC0ZUP', '61C1JS',
            '7RF4GF', 'AT7VB2', 'CF4H1M', 'M946YU', 'S9PW43', '1BBD7G', '88Q70R', 'FC3922', 'QVBXHU', 'X88DWV',
            '05U0AU', '3LS10Z', 'CD2J9U', 'BWLDNN', 'K0LKCY', 'KI9HFA', 'QTP049', 'SBD6VF', 'UUQTVA', 'ZL7PFF',
            'CZXV1N', '2SZXIH', '4V7B84', '5VUZUB', 'DGSCYR', 'FDMJRR', 'I0DO4Q', 'TAMCRX', '3SKX4V', '7AMINR',
            '80IkVQ', 'APKoAe', 'J0RYWJ', 'lgxpL6', 'VH2DCR', 'zfroDh', 'ZNVOPe', '8CGDYO', '0F8P68', 'SD5LUQ',
            'SN53VN', 'LVMMH9', 'LGNZBA', '1KAZBK', '8E3JMX', '9MCg4f', 'EXQ2NE', 'PrPFDy', 'PVVBJS', 'Sd8PAv',
            'T8USJQ', 'tXGki9', 'EMGTEW', 'JN0V15', 'PRCAE4', 'ergOuv', 'Ftc2VC', 'iBP4Fr', 'qqvRoI', 'V0qhzb',
            '1BLYXM', 'CX8CT6', '8XHGAV', 'LMeg2F', 'R6F3J5',
    ]
    marginal = [
            'TY2COS', 'YZDFTH', 'NSOK5I', 'RVOX2M', 'D0CTSD', 'H7WOUU', 'RWPK00', '4SDPFB', 'H9OS9A', 'IU43O6',
            'OHXME8', 'TYEOK6', 'VDOERX', 'hsZ6pl', 'R4JB64', 'RKNAC2', '64EQVJ', 'lCUF2l', 'HG24ZO', 'TWP10V',
            '3QOEXZ', '8HTKXA', 'ORR8VN', '5IEVIW', 'ETUQB6', 'VK3DU4', 'PV2RPF', 'QLOP4L', 'W4CQNL', 'Y7UX2N',
            'Z2JP1B', '4FYXUO', 'HJ4QFN', 'JP0OGS', '3ALXQE', 'FIQKAB', 'SZ3DCM', 'XRQVVZ', '4CNFTO', '6PC2MI',
            'JLCD5B', '249YNC', '5PL9UP', 'AVTGN7', 'QJ2QGS', '05O5MD', 'S2QZ2B', '9899TQ', '0obMbL', 'KJB859',
            'OUJ0DB', '1KDDWQ', '502MKQ', '0687TE', '9473S9', 'VOK2S9', 'J55205', '0NDb1T', '2441VU', 'HCABGW',
            'OFTUDJ', 'S5CO9J', 'VD2V13',
    ]


class ClusteredData(DiscData, QCData):

    CLUSTER_NAMES = bidict({1: 'R8', 2: 'MF-high', 3: 'MF', 4: 'post-MF', 5: 'pre-MF', 6: 'MF-ato'})
    AGGREGATE_NAMES = bidict({10: 'no-ato', 11: 'med-ato', 12: 'high-ato'})
    _good_mask = None
    _acceptable_mask = None
    _good_cells = None
    _expression = None
    _r_profiles = None

    def __init__(self, data):
        super().__init__(data.cells)
        self.clustering = data
        self.cleanup_clusters()

    def cleanup_clusters(self):
        post_mf = self._cells['cy'] > 2
        pre_mf = self._cells['cy'] < -2
        pre = self._cells['cy'] < -4
        post = self._cells['cy'] > 4

        c_mf_high = self._cells['Cluster_ward'] == self.CLUSTER_NAMES.inverse['MF-high'][0]
        c_r8 = self._cells['Cluster_ward'] == self.CLUSTER_NAMES.inverse['R8'][0]
        c_pre_mf = self._cells['Cluster_ward'] == self.CLUSTER_NAMES.inverse['pre-MF'][0]
        c_post_mf = self._cells['Cluster_ward'] == self.CLUSTER_NAMES.inverse['post-MF'][0]
        c_mf = self._cells['Cluster_ward'] == self.CLUSTER_NAMES.inverse['MF'][0]
        c_mf_ato = self._cells['Cluster_ward'] == self.CLUSTER_NAMES.inverse['MF-ato'][0]

        self._cells.loc[post_mf & c_mf_high, 'Cluster_ward'] = self.CLUSTER_NAMES.inverse['R8'][0]
        self._cells.loc[~post_mf & c_r8, 'Cluster_ward'] = self.CLUSTER_NAMES.inverse['MF-high'][0]
        self._cells.loc[~post_mf & c_post_mf, 'Cluster_ward'] = self.CLUSTER_NAMES.inverse['MF'][0]
        self._cells.loc[pre, 'Cluster_ward'] = self.CLUSTER_NAMES.inverse['pre-MF'][0]
        self._cells.loc[~pre_mf & c_pre_mf, 'Cluster_ward'] = self.CLUSTER_NAMES.inverse['MF'][0]
        self._cells.loc[post & c_mf, 'Cluster_ward'] = self.CLUSTER_NAMES.inverse['post-MF'][0]

        def aggregate(cluster):
            if cluster == self.CLUSTER_NAMES.inverse['pre-MF'][0]:
                return 10
            elif cluster == self.CLUSTER_NAMES.inverse['MF'][0]:
                return 10
            elif cluster == self.CLUSTER_NAMES.inverse['post-MF'][0]:
                return 10
            elif cluster == self.CLUSTER_NAMES.inverse['MF-ato'][0]:
                return 11
            elif cluster == self.CLUSTER_NAMES.inverse['MF-high'][0]:
                return 12
            elif cluster == self.CLUSTER_NAMES.inverse['R8'][0]:
                return 12
            else:
                return 0

        self._cells['Cluster_agg'] = self._cells['Cluster_ward'].apply(aggregate)

    def good_mask(self):
        if self._good_mask is None:
            self._good_mask = self.acceptable_mask() & ~self._cells['Sample'].isin(self.marginal)
        return self._good_mask

    def acceptable_mask(self):
        if self._acceptable_mask is None:
            self._acceptable_mask = ~self._cells['Sample'].isin(self.bad)
        return self._acceptable_mask

    def profiles(self):
        if self._profiles is None:
            self._c_profiles()
        return self._profiles

    def r_profiles(self):
        if self._r_profiles is None:
            self._c_r_profiles()
        return self._r_profiles

    def dv_profiles(self):
        return None

    def matrices(self):
        return None

    def _c_profiles(self):
        profiles = []
        cells = self.cells()[self.acceptable_mask() & ~self.bad_gene_mask()]
        cells_clean = self.cells()[self.clean_mask() & self.acceptable_mask() & ~self.bad_gene_mask()]

        cy = cells['cy'].round().astype('int')
        cluster = cells['Cluster_ward'].astype('int')
        aggregate = cells['Cluster_agg'].astype('int')

        profile = cells_clean.groupby([cluster, cy])['mCherry'].agg([np.mean, self.q99, np.std])
        profiles.append(pd.concat([profile], keys=['AtoClean'], names=['Gene']))
        profile = cells_clean.groupby([aggregate, cy])['mCherry'].agg([np.mean, self.q99, np.std])
        profiles.append(pd.concat([profile], keys=['AtoClean'], names=['Gene']))

        profile = cells.groupby([cluster, cy])['mCherry'].agg([np.mean, self.q99, np.std])
        profiles.append(pd.concat([profile], keys=['Ato'], names=['Gene']))
        profile = cells.groupby([aggregate, cy])['mCherry'].agg([np.mean, self.q99, np.std])
        profiles.append(pd.concat([profile], keys=['Ato'], names=['Gene']))

        profile = cells.groupby(['Gene', cluster, cy])['Venus'].agg([np.mean, self.q99, np.std])
        profiles.append(profile)
        profile = cells.groupby(['Gene', aggregate, cy])['Venus'].agg([np.mean, self.q99, np.std])
        profiles.append(profile)

        self._profiles = pd.concat(profiles)

    def _c_r_profiles(self):
        profiles = []
        cells = self.cells()[self.acceptable_mask() & ~self.bad_gene_mask()].copy()
        cells_clean = self.cells()[self.clean_mask() & self.acceptable_mask() & ~self.bad_gene_mask()].copy()

        cells['RelVenus'] = cells['Venus'] / cells['mCherry']

        cy = cells['cy'].round().astype('int')
        cluster = cells['Cluster_ward'].astype('int')
        aggregate = cells['Cluster_agg'].astype('int')

        profile = cells_clean.groupby([cluster, cy])['mCherry'].agg([np.mean, self.q99, np.std])
        profiles.append(pd.concat([profile], keys=['AtoClean'], names=['Gene']))
        profile = cells_clean.groupby([aggregate, cy])['mCherry'].agg([np.mean, self.q99, np.std])
        profiles.append(pd.concat([profile], keys=['AtoClean'], names=['Gene']))

        profile = cells.groupby([cluster, cy])['mCherry'].agg([np.mean, self.q99, np.std])
        profiles.append(pd.concat([profile], keys=['Ato'], names=['Gene']))
        profile = cells.groupby([aggregate, cy])['mCherry'].agg([np.mean, self.q99, np.std])
        profiles.append(pd.concat([profile], keys=['Ato'], names=['Gene']))

        profile = cells.groupby(['Gene', cluster, cy])['RelVenus'].agg([np.mean, self.q99, np.std])
        profiles.append(profile)
        profile = cells.groupby(['Gene', aggregate, cy])['RelVenus'].agg([np.mean, self.q99, np.std])
        profiles.append(profile)

        self._r_profiles = pd.concat(profiles)

    def genes(self):
        if self._genes is None:
            self._genes = self.good_cells()['Gene'].unique().tolist()
        return self._genes

    def good_cells(self):
        if self._good_cells is None:
            self._good_cells = self._cells.loc[self.acceptable_mask() & ~self.bad_gene_mask()]
        return self._good_cells

    def expression(self):
        if self._expression is None:
            cells = self.good_cells()
            cy = cells['cy'].round().astype('int')
            cluster = cells['Cluster_ward'].astype('int')
            self._expression = cells.groupby([cluster, cy, 'Gene'])['Venus'].mean().unstack(level=2).dropna()

        return self._expression
