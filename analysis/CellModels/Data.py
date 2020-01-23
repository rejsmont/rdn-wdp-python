import pandas as pd


class Cells:

    def __init__(self, data: pd.DataFrame, source=None):
        self._source = source
        self._raw = data
        self._cells = None

    @property
    def cells(self):
        if self._cells is None:
            self._cells = self._raw.dropna()
        return self._cells

    @property
    def raw(self):
        return self._raw


class ClusteredCells(Cells):
    pass
