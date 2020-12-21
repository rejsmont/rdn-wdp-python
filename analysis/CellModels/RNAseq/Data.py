import logging
import numpy as np
import pandas as pd

class RNAseq:

    _source = None
    _expression = None
    _exp_all = None

    def __init__(self, data, genes=None):
        self.logger = logging.getLogger('rdn-wdp-RNAseq')
        self.logger.info("Input is " + str(data))
        df = pd.read_csv(data)
        if 'DataSet' in df.columns:
            df.set_index('DataSet', append=True, inplace=True)
            self.data = df.reorder_levels([1, 0])
            del df
        else:
            self.data = df
        self._genes = genes

    def expression(self):
        if self._expression is None:
            matrix = (self._data[np.isin(self._data.ra.Gene, self._genes), :]).transpose()
            columns = [g for g in self._data.ra.Gene if g in self._genes]
            self._expression = pd.DataFrame(matrix, columns=columns).sort_index(axis=1)
        return self._expression

    def expression_all(self):
        if self._exp_all is None:
            matrix = (self._data[:, :]).transpose()
            columns = [g for g in self._data.ra.Gene]
            self._exp_all = pd.DataFrame(matrix, columns=columns).sort_index(axis=1)
        return self._exp_all