import logging
import os
import os.path as path
import yaml

from CellModels.Cells.IO import CellReader
from CellModels.Clustering.Data import ClusteringConfig, ClusteringResult, MultiClusteringResult, SampleSets


class ClusteringReader:

    @classmethod
    def read(cls, p):
        cells = CellReader.read(p)
        config = ClusteringConfig(cells.metadata)
        sample_sets = SampleSets(cells.metadata)
        return ClusteringResult(cells, sample_sets, config)


class MultiClusteringReader:

    @staticmethod
    def read(data_dir, gene):
        csv_files = [f for f in os.listdir(data_dir) if
                     path.isfile(path.join(data_dir, f)) and
                     str(f).endswith('.csv') and
                     gene in str(f)]

        results = []

        for csv in csv_files:
            file = path.join(data_dir, csv)
            results.append(ClusteringReader.read(file))

        return MultiClusteringResult(results)


class ClusteringResultsWriter:

    _logger = logging.getLogger('cell-writer')

    @staticmethod
    def write(rs: ClusteringResult, fn, of='hdf5'):
        ClusteringResultsWriter._logger.info("Writing results to " + str(fn))
        dirname = os.path.dirname(fn)
        basename = os.path.basename(fn)
        fn = os.path.splitext(basename)[0]
        clustering = {
            'config': rs.config.to_dict(),
            'samples': rs.sample_sets,
            'performance': rs.performance
        }
        metadata = {'clustering': clustering}
        with open(os.path.join(dirname, fn + '.yml'), 'w') as mf:
            yaml.dump(metadata, mf)
        if of == 'hdf5':
            pass
        elif of == 'csv':
            rs.cells.to_csv(os.path.join(dirname, fn + '.csv'))

        else:
            raise ValueError('Wrong output format specified (only hdf5 and csv allowed).')
