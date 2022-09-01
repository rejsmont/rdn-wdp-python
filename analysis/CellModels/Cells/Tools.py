import logging
import os
from pathlib import Path

import pandas as pd
from collections.abc import Iterable

from typing import Optional, Any

from CellModels.Cells.Filters import Masks


class classproperty(property):
    def __get__(self, obj: Any, type: Optional[type] = ...):
        return classmethod(self.fget).__get__(None, type)()


class CellColumns:
    _cols = {
        'cx_orig': ('Position', 'Raw', 'x'),
        'cy_orig': ('Position', 'Raw', 'y'),
        'cz_orig': ('Position', 'Raw', 'z'),
        'cx': ('Position', 'Normalized', 'x'),
        'cy': ('Position', 'Normalized', 'y'),
        'cz': ('Position', 'Normalized', 'z'),
        'cy_scaled': ('Position', 'Scaled', 'y'),
        'Volume': ('Measurements', 'Raw', 'Volume'),
        'mCherry_orig': ('Measurements', 'Raw', 'Venus'),
        'Venus_orig': ('Measurements', 'Raw', 'mCherry'),
        'DAPI_orig': ('Measurements', 'Raw', 'DAPI'),
        'mCherry': ('Measurements', 'Normalized', 'mCherry'),
        'Venus': ('Measurements', 'Normalized', 'Venus'),
        'DAPI': ('Measurements', 'Normalized', 'DAPI'),
        'ext_mCherry': ('Measurements', 'Prominence', 'mCherry'),
        'ext_Venus': ('Measurements', 'Prominence', 'Venus'),
        'ang_max_mCherry': ('Measurements', 'Angle', 'mCherry'),
        'ang_max_Venus': ('Measurements', 'Angle', 'Venus')
    }

    @classmethod
    def _multi_index(cls, index, misc=None):
        if misc is None:
            misc = {}
        tuples = cls._t_list(index, misc)

        return pd.MultiIndex.from_tuples(tuples)

    @classmethod
    def _t_list(cls, lst, misc=None):
        if misc is None:
            misc = {}
        tuples = []
        for i in lst:
            if not isinstance(i, str) and isinstance(i, Iterable):
                tuples.append(tuple(i))
            elif i in cls._cols.keys():
                tuples.append(cls._cols[i])
            elif i in misc.keys():
                tuples.append(misc[i])
            else:
                tuples.append((i, None))

        return tuples


class CleanUp:

    _logger = logging.getLogger('cell-reader')

    SYNONYMS = {'CG1625': 'dila', 'CG6860': 'Lrch', 'CG8965': 'rau', 'HLHmdelta': 'E(spl)mdelta-HLH',
                'king-tubby': 'ktub', 'n-syb': 'nSyb', 'Vn': 'vn', 'lola-P': 'lola'}

    @classmethod
    def remove_artifacts(cls, cells):
        if 'ZBO7IH' in cells.index.unique('Sample'):
            idx = pd.IndexSlice
            cls._logger.debug("Removing ZBO7IH blob artifact...")
            mask = (cells[('Position', 'Normalized', 'y')] > 35) & \
                   (cells[('Position', 'Normalized', 'x')] > 20) & \
                   (cells[('Position', 'Normalized', 'x')] < 30)
            artifact = cells.loc[idx[:, 'ZBO7IH'], :].index.intersection(cells.loc[mask].index)
            cells.drop(artifact, inplace=True)

    @classmethod
    def mark_bad_samples(cls, cells):
        if 'CG9801' in cells.index.unique('Gene'):
            cls._logger.debug("Marking bad CG9801 samples...")
            n = cells.index.names
            cells.reset_index(inplace=True, level='Gene')
            cells.loc[Masks(cells).samples_bad_segm, ['Gene']] = 'CG9801-B'
            cells.reset_index(inplace=True)
            cells.set_index(n, inplace=True)
            cells.sort_index(inplace=True)

    @classmethod
    def rename_cgs(cls, cells):
        genes = cells.index.unique('Gene')
        if set(cls.SYNONYMS.keys()).intersection(genes):
            cls._logger.debug("Renaming CGs to their proper names...")
            for gene, syn in cls.SYNONYMS.items():
                if gene in genes:
                    cells.rename(index={gene: syn}, level='Gene', inplace=True)

    @classproperty
    def all(cls):
        return [cls.remove_artifacts, cls.mark_bad_samples, cls.rename_cgs]
