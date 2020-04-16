#!/usr/bin/env python3
from CellModels.Clustering.Data import HarmonizedClusteringResult
from CellModels.Clustering.Figures import GeneClusteringPlot, MultiClusteringPlot
from CellModels.Clustering.IO import MultiClusteringReader

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Visualize clustering result')
    # parser.add_argument('input')
    # parser.add_argument('outdir')
    # args = parser.parse_args()
    #
    # data_file = args.input
    # out_dir = args.outdir

    sample_dir = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/new-clustering/single-gene'
    gene = 'sNPF'

    mf = MultiClusteringReader.read(sample_dir, gene)
    hf = HarmonizedClusteringResult(mf)

    fig = GeneClusteringPlot(hf)
    fig.show()

    fig = MultiClusteringPlot(hf, 'best')
    fig.show()
