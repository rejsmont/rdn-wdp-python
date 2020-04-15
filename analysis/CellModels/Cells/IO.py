import logging
import os
import warnings
import yaml

import pandas as pd

from CellModels.Data import Cells
from CellModels.Filters import Masks


class CellReader:

    _logger = logging.getLogger('cell-reader')
    SYNONYMS = {'CG1625': 'dila', 'CG6860': 'Lrch', 'CG8965': 'rau', 'HLHmdelta': 'E(spl)mdelta-HLH',
                'king-tubby': 'ktub', 'n-syb': 'nSyb'}

    @staticmethod
    def read(data, try_metadata=True):
        CellReader._logger.info("Reading cells from " + str(data))
        cells = None
        readers = ['_from_df', '_from_csv']
        if try_metadata:
            readers.append('_from_yml')
        for reader in [getattr(CellReader, x) for x in readers]:
            cells = reader(data)
            if cells is not None:
                break
        return cells

    @staticmethod
    def _from_df(data, source=None):
        CellReader._logger.debug("Attempting to read from DF")
        if data is not None and isinstance(data, pd.DataFrame) and len(data.index) != 0:
            data = CellReader._clean_up(data)
            return Cells(data, source)
        else:
            return None

    @staticmethod
    def _from_csv(datafile):
        try:
            CellReader._logger.debug("Attempting to read from CSV")
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                data = pd.read_csv(datafile, low_memory=False).set_index('ID')
            return CellReader._from_df(data, datafile)
        except:
            return None

    @staticmethod
    def _from_yml(datafile):
        if str(datafile).endswith('yml'):
            with open(datafile, 'r') as stream:
                CellReader._logger.debug("Attempting to read from YML")
                metadata = yaml.safe_load(stream)
                return CellReader._from_metadata(metadata, datafile)
        else:
            return None

    @staticmethod
    def _from_metadata(metadata, source=None):

        def explore(func, path, basedir=None, **kwargs):
            paths = [path, os.path.basename(path)]
            if basedir is not None:
                paths.append(os.path.join(basedir, path))
            if source is not None:
                paths.append(os.path.join(os.path.dirname(source), path))
            result = None
            for path in paths:
                try:
                    CellReader._logger.debug("Trying to read data from " + str(path))
                    result = func(path, **kwargs)
                except:
                    continue
                break
            return result

        if metadata is None:
            return None

        return explore(CellReader._from_csv, metadata['input']['cells'], metadata['input']['dir'])

    @staticmethod
    def _clean_up(cells):
        CellReader._logger.debug("Data cleanup...")
        # Remove artifact from sample ZBO7IH
        CellReader._logger.debug("Removing ZBO7IH blob artifact...")
        artifact = cells[(cells['Sample'] == 'ZBO7IH') &
                         (cells['cy'] > 35) &
                         (cells['cx'] > 20) &
                         (cells['cx'] < 30)].index
        cells = cells.drop(artifact)

        # Mark bad CG9801 samples
        CellReader._logger.debug("Marking bad CG9801 samples...")
        m = Masks(cells)
        cells.loc[m.samples_bad_segm, 'Gene'] = 'CG9801-B'

        # Rename some CGs to proper gene names
        CellReader._logger.debug("Renaming CGs to their proper names...")
        for gene, syn in CellReader.SYNONYMS.items():
            cells.loc[cells['Gene'] == gene, 'Gene'] = syn

        return cells
