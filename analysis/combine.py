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
           os.path.isfile(os.path.join(args.dir, f)) and f.endswith(".csv")
           and not f.endswith("raw.csv") and not f == 'samples.csv']
print(samples)
datasets = []

used = 0
skipped = 0

metadata = None

for sample in samples:
    with open(os.path.join(args.dir, sample.replace(".csv", ".yml"))) as file:
        sample_metadata = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
    print(sample)
    sample_data = pd.read_csv(os.path.join(args.dir, sample), index_col=0)
    sample_data['Sample'] = sample_metadata['Sample']
    sample_data['Gene'] = sample_metadata['GeneName']
    usable = sample_metadata['Quality']['Usable'] == 'yes'
    if usable:
        furrow = sample_metadata['Quality']['Furrow'] == 'ok' or sample_metadata['Quality']['Furrow'] == 'manual'
        crop = sample_metadata['Quality']['Crop'] == 'ok'
    else:
        furrow = False
        crop = False
    if usable and furrow and crop:
        logging.info("Using    %s", sample_metadata['Sample'])
        used = used + 1
        datasets.append(sample_data)
    else:
        logging.info("Skipping %s", sample_metadata['Sample'])
        skipped = skipped + 1
data = pd.concat(datasets, ignore_index=True)

if args.out:
    data.to_csv(args.out)
