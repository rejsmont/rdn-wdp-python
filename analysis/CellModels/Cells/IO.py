import logging
import os
import warnings
import yaml
import pandas as pd

from CellModels.Cells.Data import Cells, CellMetadata
from pathlib import Path
from typing import Optional, Callable


class IncompatibleInputError(ValueError):
    pass


class CellReader:

    _logger = logging.getLogger('cell-reader')

    READERS = ['_from_df', '_from_yml', '_from_csv', '_from_hdf']

    @classmethod
    def read(cls, p, m: Optional[CellMetadata] = None, **kwargs):
        cls._logger.info("Reading cells from " + str(p))
        cells = None
        error = False
        for reader in [getattr(cls, x) for x in cls.READERS]:
            try:
                cells = reader(p, m, **kwargs)
            except IncompatibleInputError as e:
                cells = None
            except FileNotFoundError as e:
                cells = None
                error = e
            if cells is not None:
                break
        if cells is None and error:
            raise error
        else:
            return cells

    @classmethod
    def read_metadata(cls, f):
        if not f.endswith('.yml'):
            f = Path('.'.join(f.split('.')[:-1]) + '.yml')
        try:
            return cls._from_yml(f, metadata_only=True)
        except ValueError:
            return None
        except RuntimeError:
            return None
        except FileNotFoundError:
            return None

    @classmethod
    def _from_df(cls, d: pd.DataFrame, m: Optional[CellMetadata] = None, s: Optional[Path] = None, **kwargs):
        if not isinstance(d, pd.DataFrame):
            raise IncompatibleInputError
        if len(d.index) > 0:
            cls._logger.debug("Attempting to read from DataFrame")
            if 'ID' in d.columns:
                d = d.set_index('ID')
            if 'Nucleus' not in d.columns:
                if 'Unnamed: 0' in d.columns:
                    d = d.rename(columns={'Unnamed: 0': 'Nucleus'})
                else:
                    d['Nucleus'] = d.index
            if 'Gene' not in d.columns or 'Sample' not in d.columns:
                raise IncompatibleInputError
            try:
                return Cells(d, m, s)
            except ValueError:
                raise IncompatibleInputError
        else:
            raise IncompatibleInputError

    @classmethod
    def _from_csv(cls, f, m: Optional[CellMetadata] = None, **kwargs):
        cls._logger.debug("Attempting to read from CSV")
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            try:
                d = pd.read_csv(f, low_memory=False)
            except ValueError:
                raise IncompatibleInputError
            except KeyError:
                raise IncompatibleInputError
            if m is None:
                m = cls.read_metadata(f)
        return cls._from_df(d, m, Path(f))

    @classmethod
    def _from_hdf(cls, f, m: Optional[CellMetadata] = None, **kwargs):
        cls._logger.debug("Attempting to read from HDF")
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if ':' in f:
                fn = ':'.join(f.split(':')[:-1])
                ds = f.split(':')[-1]
            else:
                fn = f
                ds = None
            try:
                d = pd.DataFrame(pd.read_hdf(fn, ds))
            except OSError:
                raise IncompatibleInputError
            except ValueError:
                raise IncompatibleInputError
            except RuntimeError:
                raise IncompatibleInputError
            if m is None:
                m = cls.read_metadata(fn)
        return cls._from_df(d, m, Path(fn))

    @classmethod
    def _from_yml(cls, f, metadata_only: bool = False, **kwargs):
        if str(f).endswith('yml'):
            try:
                with open(f, 'r') as s:
                    cls._logger.debug("Attempting to read from YML")
                    m = CellMetadata(yaml.safe_load(s))
                    if metadata_only:
                        return m
                    else:
                        cls._logger.debug("Metadata loaded!")
                        cells = cls._from_metadata(m, Path(f))
            except ValueError:
                raise IncompatibleInputError
        else:
            raise IncompatibleInputError
        return cells

    @classmethod
    def _from_metadata(cls, m: CellMetadata, s: Optional[Path] = None, **kwargs):

        def explore(func: Callable, path: Path, basedir: Optional[Path] = None, **f_kwargs):
            paths = [path, os.path.basename(path)]
            if basedir is not None:
                paths.append(os.path.join(basedir, path))
            if s is not None:
                paths.append(os.path.join(os.path.dirname(s), path))
            result = None
            for path in paths:
                try:
                    cls._logger.debug("Trying to read data from " + str(path))
                    result = func(path, **f_kwargs)
                except IncompatibleInputError:
                    continue
                except FileNotFoundError:
                    continue
                break
            return result

        cls._logger.debug("Attempting to read from Metadata")

        cells = None
        readers = list(filter(lambda x: (x != '_from_yml') and (x != '_from_df'), cls.READERS))

        in_cells = out_cells = None
        for reader in [getattr(cls, x) for x in readers]:
            if 'input' in m:
                in_cells = explore(reader, m['input']['cells'], m['input']['dir'], m=m)
            if 'output' in m:
                out_cells = explore(reader, m['output']['cells'], m['output']['dir'], m=m)
            cells = out_cells if out_cells is not None else in_cells
            if cells is not None:
                break
        return cells
