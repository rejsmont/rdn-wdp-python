#!/usr/bin/env python3

import argparse
import csv
import logging
import os
import numpy as np
import pandas as pd
import warnings
from formats import GTFFeature, NarrowPeak


class ChIP:

    _source = None

    def __init__(self, data, genes=None):
        self.logger = logging.getLogger('rdn-wdp-ChIP')
        self.logger.info("Input is " + str(data))
        self.genes = genes
        self._peaks = {}
        self.from_csv(data)

    def from_csv(self, datafile):
        self.logger.debug("Trying from CSV " + str(datafile))
        self._source = datafile
        try:
            with open(datafile) as f:
                reader = csv.reader(f, delimiter='\t')
                for peak in reader:
                    gene = GTFFeature(peak[10:19])
                    peak = NarrowPeak(peak)
                    if self.genes is None or gene.name in self.genes:
                        if gene.name not in self._peaks.keys():
                            self._peaks[gene.name] = []
                        if peak not in self._peaks[gene.name]:
                            self._peaks[gene.name].append(peak)

        except RuntimeError:
            return False

    def peaks(self):
        binding = {'Gene': [], 'p_area': []}
        for gene, peaks in self._peaks.items():
            for peak in peaks:
                binding['Gene'].append(gene)
                binding['p_area'].append((int(peak.end) - int(peak.start)) * float(peak.enrichment))

        return pd.DataFrame.from_dict(binding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ChIP data')
    parser.add_argument('--data', required=True)
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

    chip = ChIP(args.data)
    peaks = chip.peaks().groupby('Gene').agg(['sum', 'count'])
