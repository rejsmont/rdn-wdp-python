import os
import os.path as path
import yaml

from CellModels.Cells.IO import CellReader
from CellModels.Cells.Tools import CellColumns
from CellModels.Clustering.Data import ClusteringConfig, ClusteringResult, MultiClusteringResult


class ClusteringReader(CellColumns):

    @classmethod
    def read(cls, p):
        if p.endswith(".yml"):
            yml_path = p
            csv_path = p.replace(".yml", ".csv")
        elif p.endswith(".csv"):
            yml_path = p.replace(".csv", ".yml")
            csv_path = p
        else:
            raise ValueError("Input path must be a CSV or YAML file")

        cells = CellReader.read(csv_path)

        with open(yml_path) as yml_file:
            metadata = yaml.load(yml_file, Loader=yaml.FullLoader)
            mc = metadata['config']
            config = ClusteringConfig(
                mc['clusters'],
                mc['samples'],
                mc['repeats'],
                mc['cutoff'],
                mc['method'],
                mc['metric'],
                cls._t_list(mc['hc_features']),
                cls._t_list(mc['rf_features'])
            )
            sample_sets = metadata['samples']

        misc = {'Cluster_' + config.method: ('Cluster', config.method, config.clusters)}
        c = cells.cells.set_index(['Gene', 'Sample', 'Nucleus']).sort_index()
        c.columns = cls._multi_index(c.columns, misc)
        result = ClusteringResult(c, sample_sets, config)

        return result


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