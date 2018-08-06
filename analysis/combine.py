#!/usr/bin/env python3

import argparse
import logging
import os
import pandas as pd

parser = argparse.ArgumentParser(description='Nuclear point cloud sample combining.')
parser.add_argument('--dir', required=True)
parser.add_argument('--list', required=True)
parser.add_argument('--out')
parser.add_argument('--log')

args = parser.parse_args()

if args.log:
    logging.basicConfig(level=args.log.upper())

samples = [f for f in os.listdir(args.dir) if
           os.path.isfile(os.path.join(args.dir, f)) and f.endswith("normalized.csv")]
sample_list = pd.read_csv(args.list, sep='\t')
datasets = []


def value(data, field):
    return data[field].values[0]


used = 0
skipped = 0

for sample in samples:
    sample_id = sample.split("_")[-2]
    sample_metadata = sample_list[sample_list['Sample'] == sample_id]
    sample_data = pd.read_csv(os.path.join(args.dir, sample), index_col=0)
    sample_data['Sample'] = sample_id
    sample_data['Gene'] = value(sample_metadata, 'Gene')
    furrow = value(sample_metadata, 'Furrow') == 'ok' or value(sample_metadata, 'Furrow') == 'manual'
    crop = value(sample_metadata, 'Crop') == 'ok'
    usable = value(sample_metadata, 'Usable') == 'yes'
    if furrow and crop and usable:
        logging.info("Using    %s", sample_id)
        used = used + 1
        datasets.append(sample_data)
    else:
        logging.info("Skipping %s", sample_id)
        skipped = skipped + 1
    data = pd.concat(datasets)
    if args.out:
        data.to_csv(args.out)
