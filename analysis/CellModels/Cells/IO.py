import logging
import os
import warnings
import yaml
import pandas as pd

from CellModels.Cells.Data import Cells


class CellReader:

    _logger = logging.getLogger('cell-reader')

    READERS = ['_from_df', '_from_csv', '_from_hdf', '_from_yml']

    @classmethod
    def read(cls, p, m=None):
        cls._logger.info("Reading cells from " + str(p))
        cells = None
        for reader in [getattr(cls, x) for x in cls.READERS]:
            cells = reader(p, m)
            if cells is not None:
                break
        return cells

    @classmethod
    def _from_df(cls, d, m=None, s=None):
        if d is not None and isinstance(d, pd.DataFrame) and len(d.index) != 0:
            cls._logger.debug("Attempting to read from DF")
            return Cells(d, m, s)
        else:
            return None

    @classmethod
    def _from_csv(cls, f, m=None):
        try:
            cls._logger.debug("Attempting to read from CSV")
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                d = pd.read_csv(f, low_memory=False).set_index('ID')
                if m is None:
                    m = cls._from_yml('.'.join(f.split('.')[:-1]) + '.yml', m_only=True)
            return cls._from_df(d, m, f)
        except Exception as e:
            raise e
            return None

    @classmethod
    def _from_hdf(cls, f, m=None):
        try:
            cls._logger.debug("Attempting to read from HDF")
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                if ':' in f:
                    fn = ':'.join(f.split(':')[:-1])
                    ds = f.split(':')[-1]
                else:
                    fn = f
                    ds = None
                d = pd.read_hdf(fn, ds)
                if m is None:
                    m = cls._from_yml('.'.join(f.split('.')[:-1]) + '.yml', m_only=True)
            return cls._from_df(d, m, f)
        except:
            return None

    @classmethod
    def _from_yml(cls, f, m_only=False):
        if str(f).endswith('yml'):
            with open(f, 'r') as s:
                cls._logger.debug("Attempting to read from YML")
                m = yaml.safe_load(s)
                if m_only:
                    return m
                else:
                    return cls._from_metadata(m, f)
        else:
            return None

    @classmethod
    def _from_metadata(cls, m, s=None):

        def explore(func, path, basedir=None, **kwargs):
            paths = [path, os.path.basename(path)]
            if basedir is not None:
                paths.append(os.path.join(basedir, path))
            if s is not None:
                paths.append(os.path.join(os.path.dirname(s), path))
            result = None
            for path in paths:
                try:
                    cls._logger.debug("Trying to read data from " + str(path))
                    result = func(path, **kwargs)
                except:
                    continue
                break
            return result

        if m is None:
            return None

        cells = None
        readers = list(filter(lambda x: (x != '_from_yml') and (x != '_from_df'), cls.READERS))
        for reader in [getattr(cls, x) for x in readers]:
            cells = explore(reader, m['input']['cells'], m['input']['dir'])
            if cells is not None:
                break
        return cells
