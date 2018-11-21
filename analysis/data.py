#!/usr/bin/env python3

import numpy as np
import pandas as pd
import warnings


class DiscData:

    def __init__(self, datafile):
        self._source = datafile
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            self._cells = pd.read_csv(self._source, index_col=0)
        if self._cells.empty:
            raise RuntimeError("No data was found in specified file!")
        self._cells_clean = None
        self._cells_background = None
        self._cells_ato = None
        self._cells_no_ato = None
        self._genes = None
        self._genes_sorted = None
        self._profiles = None
        self._dv_profiles = None
        self._matrices = None
        self.clean_up()
        self._clean_mask = None
        self._background_mask = None
        self._furrow_mask = None

    def clean_up(self):
        # Remove artifact from sample ZBO7IH
        artifact = self._cells[(self._cells['Sample'] == 'ZBO7IH') &
                               (self._cells['cy'] > 35) &
                               (self._cells['cx'] > 20) &
                               (self._cells['cx'] < 30)].index
        self._cells = self._cells.drop(artifact)

        # Mark and remove bad CG9801 samples
        bad_samples = ['J0RYWJ', '3SKX4V', '7AMINR', '4EAAEF', 'VH2DCR', 'WJ8F8M', 'ZNVOPe', 'APKoAe', 'zfroDh',
                              'lgxpL6', 'pcTNzE', '80IkVQ', 'UQZJ3K']
        self._cells.loc[self._cells['Sample'].isin(bad_samples), 'Gene'] = 'CG9801-B'
        bad_cells = self._cells[self._cells['Gene'] == 'CG9801-B'].index
        self._cells = self._cells.drop(bad_cells)

    # Binary masks for identifying cell types #

    def clean_mask(self):
        if self._clean_mask is None:
            self._clean_mask = self._cells['Gene'].isin(self.genes_clean())
        return self._clean_mask

    def furrow_mask(self):
        if self._furrow_mask is None:
            self._furrow_mask = (self._cells['cy'] >= -10) & (self._cells['cy'] <= 10)
        return self._furrow_mask

    def background_mask(self):
        if self._background_mask is None:
            self._background_mask = (self._cells['cy'] >= -10) & (self._cells['cy'] <= -5)
        return self._background_mask


    # Shortcuts to get cells (filtered) #

    def source(self):
        return self._source

    def cells(self):
        return self._cells

    def cells_clean(self):
        if self._cells_clean is None:
            self._cells_clean = self._cells[self._cells['Gene'].isin(self.genes_clean())]
        return self._cells_clean

    def cells_background(self):
        if self._cells_background in None:
            self._cells_background = self._cells[(self._cells['cy'] >= -10) & (self._cells['cy'] <= -5)]
        return self._cells_background

    def cells_ato(self):
        if self._cells_ato is None:
            cells = self._cells()
            background = self.cells_background()
            self._cells_ato = cells[(cells['mCherry'] > background['mCherry'].quantile(0.90))]
        return self._cells_ato

    def cells_no_ato(self):
        if self._cells_no_ato is None:
            cells = self._cells()
            background = self.cells_background()
            self._cells_no_ato = cells[(cells['mCherry'] < background['mCherry'].quantile(0.50))]
        return self._cells_no_ato

    def genes(self):
        if not self._genes:
            self._genes = self._cells['Gene'].unique().tolist()
        return self.genes_sorted

    def genes_sorted(self):
        if self._genes_sorted is None:
            before = self._cells[self._cells['cy'] < 0].groupby(['Gene'])['Venus'].quantile(0.99)
            after = self._cells[(self._cells['cy'] > 0) & (self._cells['cy'] < 20)].groupby(['Gene'])['Venus'].quantile(0.99)
            ratio = after / before
            self._genes_sorted = ratio.sort_values(ascending=False).index.tolist()
        return self._genes_sorted

    @staticmethod
    def genes_clean():
        return ['CG31176', 'beat-IIIc', 'king-tubby', 'lola-P', 'nmo', 'sNPF', 'Vn', 'Fas2', 'siz']

    def profiles(self):
        if self._profiles is None:
            self._profiles_matrices()
        return self._profiles

    def dv_profiles(self):
        if self._dv_profiles is None:
            self._profiles_matrices()
        return self._dv_profiles

    def matrices(self):
        if self._matrices is None:
            self._profiles_matrices()
        return self._matrices

    @staticmethod
    def q99(x): return np.percentile(x, 99)

    def _profiles_matrices(self):
        profiles = []
        dv_profiles = []
        matrices = []

        cells = self.cells()
        cells_mf = cells[(cells['cy'] >= -3) & (cells['cy'] <= 3)]
        cells_clean = self.cells_clean()
        cells_mf_clean = cells_clean[(cells_clean['cy'] >= -3) & (cells_clean['cy'] <= 3)]

        cx = cells_clean['cx'].round().astype('int')
        cy = cells_clean['cy'].round().astype('int')
        profile = cells_clean.groupby(cy)['mCherry'].agg([np.mean, self.q99])
        profiles.append(pd.concat([profile], keys=['AtoClean'], names=['Gene']))
        matrix = cells_clean.groupby([cx, cy])['mCherry', 'ext_mCherry'].agg(
            {'mCherry': [np.mean, np.max], 'ext_mCherry': np.max})
        matrix.columns = ['mean', 'max', 'ext']
        matrices.append(pd.concat([matrix], keys=['AtoClean'], names=['Gene']))
        cx = cells_mf_clean['cx'].round().astype('int')
        profile = cells_mf_clean.groupby(cx)['mCherry'].agg([np.mean, self.q99])
        dv_profiles.append(pd.concat([profile], keys=['AtoClean'], names=['Gene']))

        cx = cells['cx'].round().astype('int')
        cy = cells['cy'].round().astype('int')
        profile = cells.groupby(cy)['mCherry'].agg([np.mean, self.q99])
        profiles.append(pd.concat([profile], keys=['Ato'], names=['Gene']))
        matrix = cells.groupby([cx, cy])['mCherry', 'ext_mCherry'].agg(
            {'mCherry': [np.mean, np.max], 'ext_mCherry': np.max})
        matrix.columns = ['mean', 'max', 'ext']
        matrices.append(pd.concat([matrix], keys=['Ato'], names=['Gene']))

        profile = cells.groupby(['Gene', cy])['Venus'].agg([np.mean, self.q99])
        profiles.append(profile)

        matrix = cells.groupby(['Gene', cx, cy])['Venus', 'ext_Venus'].agg(
            {'Venus': [np.mean, np.max], 'ext_Venus': np.max})
        matrix.columns = ['mean', 'max', 'ext']
        matrices.append(matrix)

        cx = cells_mf['cx'].round().astype('int')
        profile = cells_mf.groupby(cx)['mCherry'].agg([np.mean, self.q99])
        dv_profiles.append(pd.concat([profile], keys=['Ato'], names=['Gene']))
        profile = cells_mf.groupby(['Gene', cx])['Venus'].agg([np.mean, self.q99])
        dv_profiles.append(profile)

        self._profiles = pd.concat(profiles)
        self._dv_profiles = pd.concat(dv_profiles)
        self._matrices = pd.concat(matrices)
