import logging
import os
import os.path as path
import pandas as pd
import yaml

from CellModels.Cells.Data import Cells
from CellModels.Cells.IO import CellReader
from CellModels.Clustering.Data import ClusteringConfig, ClusteringResult, MultiClusteringResult, SampleSets, \
    Performance


class ClusteringReader:

    @classmethod
    def read(cls, p):
        if p.endswith('.h5') or p.endswith('.hdf5'):
            try:
                cells_df = pd.DataFrame(pd.read_hdf(p, 'clustering/cells'))
                m = cls._read_metadata(p)
                cells = Cells(cells_df, m)
            except:
                return None
            try:
                centroids = pd.DataFrame(pd.read_hdf(p, 'clustering/centroids'))
            except:
                centroids = None
            try:
                clusters = pd.DataFrame(pd.read_hdf(p, 'clustering/clusters'))
            except:
                clusters = None
            try:
                training = pd.DataFrame(pd.read_hdf(p, 'clustering/training'))
            except:
                training = None
            try:
                test = pd.DataFrame(pd.read_hdf(p, 'clustering/test'))
            except:
                test = None
        else:
            try:
                cells = CellReader.read(p)
                centroids = None
                clusters = None
                training = None
                test = None
            except:
                return None

        try:
            config = ClusteringConfig(cells.metadata)
            sample_sets = SampleSets(cells.metadata)
            performance = Performance(cells.metadata)
        except:
            return None

        if len(config.clusters) > 1:
            return MultiClusteringResult(cells, sample_sets, config, clusters=clusters, centroids=centroids,
                                         training=training, test=test, performance=performance)
        else:
            return ClusteringResult(cells, sample_sets, config, clusters=clusters, centroids=centroids,
                                    training=training, test=test, performance=performance)

    @staticmethod
    def _read_metadata(p):
        if not p.endswith('.yml'):
            p = '.'.join(p.split('.')[:-1]) + '.yml'
        try:
            with open(p, 'r') as s:
                m = yaml.safe_load(s)
                return m
        except Exception as e:
            return None


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
