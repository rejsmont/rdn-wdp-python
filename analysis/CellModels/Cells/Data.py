import logging
import pandas as pd
from typing import Set
from collections.abc import Iterable

from CellModels.Cells.Tools import CleanUp, CellColumns


class Cells(pd.DataFrame, CellColumns):

    _logger = logging.getLogger('cell-reader')
    _cells_internal_names = ['_source', '_raw', '_cells_metadata', '_clean_up']
    _internal_names_set: Set[str] = set(
        pd.DataFrame._internal_names + _cells_internal_names
    )

    def __init__(self, data: pd.DataFrame, metadata=None, source=None, clean_up=None):
        self._source = source
        self._raw = self._set_multi_index(data)
        self._cells_metadata = metadata
        clean_up = CleanUp.all if clean_up is None else clean_up
        if callable(clean_up):
            clean_up = [clean_up]
        if isinstance(clean_up, Iterable):
            for v in clean_up:
                assert callable(v), 'clean_up must be a callable, a list of callables, or evaluate to False'
        else:
            assert not clean_up, 'clean_up must be a callable, a list of callables, or evaluate to False'
        self._clean_up = clean_up
        super().__init__(self._do_clean_up(self._raw))

    @property
    def cells(self):
        return self

    @property
    def raw(self):
        return self._raw

    @property
    def metadata(self):
        return self._cells_metadata

    @property
    def source(self):
        return self._source

    @property
    def genes(self):
        return self.index.unique('Gene')

    @property
    def samples(self):
        return self.index.unique('Sample')

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError("Not supported for Cells!")

    @classmethod
    def _set_multi_index(cls, data):
        if not isinstance(data.index, pd.MultiIndex):
            data = data.set_index(['Gene', 'Sample', 'Nucleus']).sort_index()
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = cls._multi_index(data.columns)
        return data

    def _do_clean_up(self, df):
        c = df.copy()
        if self._clean_up:
            for fn in self._clean_up:
                fn(c)
        return c
