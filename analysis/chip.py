#!/usr/bin/env python3

import argparse
import csv
import logging
import os
import numpy as np
import pandas as pd
import warnings
from io import StringIO
from Bio import motifs, SeqIO, Seq
from Bio.Alphabet import IUPAC
from formats import GTFFeature, NarrowPeak
from data import DiscData


class ChIP:

    _source = None
    # Atonal PFM
    PFM = {'a': [23, 22,  0, 33,  0,  0,  0,  0,  0,  0, 17],
           'c':  [0,  0, 33,  0,  0, 11,  0,  0,  0, 17,  0],
           'g': [10,  0,  0,  0, 30, 22,  0, 33, 26,  0, 13],
           't':  [0, 11,  0,  0,  3,  0, 33,  0,  7, 16,  3]}

    def __init__(self, data, genome, genes=None):
        self.logger = logging.getLogger('rdn-wdp-ChIP')
        self.logger.info("Input is " + str(data))
        self.genes = genes
        self.genome = genome
        self._peaks = {}
        self.from_csv(data)
        self.motif = self.pfm()

    def from_csv(self, datafile):
        self.logger.debug("Trying from CSV " + str(datafile))
        self._source = datafile
        try:
            with open(datafile) as f:
                reader = csv.reader(f, delimiter='\t')
                print(DiscData.SYNONYMS.keys())
                for peak in reader:
                    gene = GTFFeature(peak[10:19])
                    if gene.name in DiscData.SYNONYMS.keys():
                        gene.name = DiscData.SYNONYMS[gene.name]
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

    def pfm(self):
        s = ''
        for l in ['a', 'c', 'g', 't']:
            for n in self.PFM[l]:
                s += str(n) + ' '
            s += '\n'

        sio = StringIO(s)
        m = motifs.read(sio, 'pfm')

        return m

    def pwm_scan(self, left=0, right=0):
        records = SeqIO.index(self.genome, 'fasta', alphabet=IUPAC.IUPACUnambiguousDNA())
        print(self.motif.consensus)
        print(self.motif.pssm)
        print(self.motif)
        for gene, peaks in self._peaks.items():
            for peak in peaks:
                chrom = str(peak.chrom).replace('chr', '')
                seq = records[chrom][int(peak.start) - left:int(peak.end) + right].seq
                matches = list(self.motif.pssm.search(seq))
                print("Gene: " + str(gene) + ", height: " + str(peak.enrichment) + ", sites: " + str(len(matches)))
                if matches:
                    print(matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ChIP data')
    parser.add_argument('--data', required=True)
    parser.add_argument('--genome', required=True)
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

    chip = ChIP(args.data, args.genome)
    peaks = chip.peaks().groupby('Gene').agg(['sum', 'count'])
    print(chip.pwm_scan(left=1000, right=1000))
