import logging
import pandas as pd

from collections.abc import Iterable
from pathlib import Path
from typing import Union, Set, Optional, Callable

from CellModels.Cells.Tools import CleanUp, CellColumns


class CellMetadata(dict):

    def __init__(self, seq=None, **kwargs):
        super().__init__(seq, **kwargs)


class Cells(pd.DataFrame, CellColumns):
    _logger = logging.getLogger('cell-reader')
    _cells_internal_names = ['_source', '_raw', '_cells_metadata', '_clean_up']
    _internal_names_set: Set[str] = set(
        pd.DataFrame._internal_names + _cells_internal_names
    )

    def __init__(self, data: pd.DataFrame, metadata: Optional[CellMetadata] = None, source: Optional[Path] = None,
                 clean_up: Union[Callable, Iterable, None] = None):
        self._source = source
        self._raw = self._set_multi_index(data)
        self._cells_metadata = metadata
        self._clean_up = self.__class__._check_clean_up(clean_up)
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

    @staticmethod
    def _check_clean_up(v: Union[Callable, Iterable, None]):
        if v is None:
            v = CleanUp.all
        if isinstance(v, Callable):
            v = [v]
        elif isinstance(v, Iterable):
            for i in v:
                assert callable(i), 'clean_up must be a callable, a list of callables, or evaluate to False'
        else:
            assert not v, 'clean_up must be a callable, a list of callables, or evaluate to False'
        return v

    @classmethod
    def _set_multi_index(cls, data: pd.DataFrame):
        if not isinstance(data.index, pd.MultiIndex):
            data = data.set_index(['Gene', 'Sample', 'Nucleus']).sort_index()
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = cls._multi_index(data.columns)
        return data

    def _do_clean_up(self, df: pd.DataFrame):
        c = df.copy()
        if self._clean_up:
            for fn in self._clean_up:
                fn(c)
        return c
