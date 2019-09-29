#!/usr/bin/env python3

import argparse
import logging
import os
import pandas as pd
import yaml

parser = argparse.ArgumentParser(description='Nuclear point cloud sample combining.')
parser.add_argument('--dir', required=True)
parser.add_argument('--out')
parser.add_argument('--log')

args = parser.parse_args()

if args.log:
    logging.basicConfig(level=args.log.upper())

samples = [f for f in os.listdir(args.dir) if
           os.path.isfile(os.path.join(args.dir, f)) and not f.endswith("raw.csv") and not f == 'samples.csv']
datasets = []

used = 0
skipped = 0

for sample in samples:
    sample_metadata = yaml.load(os.path.join(args.dir, sample.replace(".csv", ".yml")))
    sample_data = pd.read_csv(os.path.join(args.dir, sample), index_col=0)
    sample_data['Sample'] = sample_metadata['Sample']
    sample_data['Gene'] = sample_metadata['GeneName']
    furrow = sample_metadata['Quality']['Furrow'] == 'ok' or sample_metadata['Quality']['Furrow'] == 'manual'
    crop = sample_metadata['Quality']['Crop'] == 'ok'
    usable = sample_metadata['Quality']['Usable'] == 'yes'
    if furrow and crop and usable:
        logging.info("Using    %s", sample_data['Sample'])
        used = used + 1
        datasets.append(sample_data)
    else:
        logging.info("Skipping %s", sample_data['Sample'])
        skipped = skipped + 1
    data = pd.concat(datasets, ignore_index=True)

if args.out:
    data.to_csv(args.out)
