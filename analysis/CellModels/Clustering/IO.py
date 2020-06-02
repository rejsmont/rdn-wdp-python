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
        dirname = os.path.dirname(fn)
        basename = os.path.basename(fn)
        fn = os.path.splitext(basename)[0]
        ex = os.path.splitext(basename)[1]
        clustering = {
            'config': rs.config.to_dict(),
            'samples': rs.sample_sets,
            'performance': rs.performance
        }
        metadata = {'clustering': clustering}
        mf_name = os.path.join(dirname, fn + '.yml')
        with open(mf_name, 'w') as mf:
            ClusteringResultsWriter._logger.info("Writing metadata to " + mf_name)
            yaml.dump(metadata, mf)
        if of == 'hdf5':
            rf_name = os.path.join(dirname, fn + ex)
            ClusteringResultsWriter._logger.info("Writing clustering results to " + rf_name + " using HDF5 writer")
            rs.cells.to_hdf(rf_name, 'clustering/cells')
            rs.clusters.to_hdf(rf_name, 'clustering/clusters')
            rs.centroids.to_hdf(rf_name, 'clustering/centroids')
            if rs.training is not None:
                rs.training.index.to_frame().to_hdf(rf_name, 'clustering/training')
            if rs.test is not None:
                rs.test.index.to_frame().to_hdf(rf_name, 'clustering/test')
        elif of == 'csv':
            ClusteringResultsWriter._logger.info("Writing clustering results to " + str(fn) + " using CSV writer")
            rs.cells.to_csv(os.path.join(dirname, fn + '.csv'))
        else:
            raise ValueError('Wrong output format specified (only hdf5 and csv allowed).')
