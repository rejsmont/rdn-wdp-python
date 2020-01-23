import logging

from CellModels.Clustering import ClusteringConfig, Clustering
from CellModels.IO import CellReader, ClusteringResultsWriter

datafile = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/processing/samples_complete.csv'

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    data = CellReader.read(datafile)

