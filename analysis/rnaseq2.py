#!/usr/bin/env python3

import argparse
import csv
import loompy
import logging
import os
import numpy as np
import pandas as pd
import warnings
from scipy.spatial import distance
from data import DiscData
from clustering import Clustering, ClusteredData


class RNAseq:

    datasets = [
        'GSM3178860_DMS1.expr.txt', 'GSM3178863_DMS4.expr.txt', 'GSM3178866_DMS7.expr.txt', 'GSM3178869_DMS10.expr.txt',
        'GSM3178861_DMS2.expr.txt', 'GSM3178864_DMS5.expr.txt', 'GSM3178867_DMS8.expr.txt', 'GSM3178870_DMS11.expr.txt',
        'GSM3178862_DMS3.expr.txt', 'GSM3178865_DMS6.expr.txt', 'GSM3178868_DMS9.expr.txt'
    ]

    _source = None
    _expression = None
    _exp_all = None

    def __init__(self, data, genes=None):
        self.logger = logging.getLogger('rdn-wdp-RNAseq')
        self.logger.info("Input is " + str(data))
        dataframes = []
        columns = []

        for i, ds in enumerate(self.datasets):
            dspath = os.path.join(data, ds)
            print(dspath)
            df = pd.read_csv(dspath, sep='\t').transpose()
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            df['DataSet'] = i
            dataframes.append(df)
            print("Items:", len(df.index), "Genes:", len(df.columns))
            columns = columns + df.columns.tolist()
        columns = pd.unique(pd.Series(columns))
        for i, df in enumerate(dataframes):
            print("Reindexing", i)
            df = df.reindex(columns=columns, fill_value=0)
            if i == 0:
                df.to_csv('/tmp/scRNAseq.csv', index=False)
            else:
                df.to_csv('/tmp/scRNAseq.csv', mode='a', index=False, header=False)
        print("Done")
        self._genes = genes

datasets = [
    'GSM3178860_DMS1.expr.txt', 'GSM3178863_DMS4.expr.txt', 'GSM3178866_DMS7.expr.txt', 'GSM3178869_DMS10.expr.txt', 
    'GSM3178861_DMS2.expr.txt', 'GSM3178864_DMS5.expr.txt', 'GSM3178867_DMS8.expr.txt', 'GSM3178870_DMS11.expr.txt', 
    'GSM3178862_DMS3.expr.txt', 'GSM3178865_DMS6.expr.txt', 'GSM3178868_DMS9.expr.txt'
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ChIP data')
    parser.add_argument('--scngsdir', required=True)
    parser.add_argument('--cells', required=False)
    parser.add_argument('--log')
    parser.add_argument('--outdir')
    parser.add_argument('--reproducible', dest='reproducible', action='store_true')
    parser.add_argument('--not-reproducible', dest='reproducible', action='store_false')
    parser.set_defaults(reproducible=False)
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=args.log.upper())
        logging.getLogger('PIL.Image').setLevel(logging.INFO)
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger('joblib').setLevel(logging.INFO)
        logging.getLogger('cloudpickle').setLevel(logging.INFO)

    if args.reproducible:
        np.random.seed(0)
    else:
        np.random.seed()

    rnas = RNAseq(args.scngsdir)

