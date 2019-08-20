#!/usr/bin/env python3

import sys
import os
import math
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def split_file_name(file_name):
    parts = os.path.basename(file_name).replace('.yml', '').split("_")
    if len(parts) == 4:
        vial_id = int(parts[0])
        disc_no = int(parts[2])
        hash_id = parts[3]
        gene = None
    else:
        vial_id = 'undefined'
        disc_no = parts[3]
        hash_id = parts[4]
        gene = parts[1]
    return vial_id, disc_no, hash_id, gene


def load_metadata(file_name):
    with open(file_name) as file:
        metadata = yaml.load(file)
        file.close()
        return metadata


def get_imaging_date(metadata):
    if metadata:
        try:
            return datetime.strptime(
                metadata['DAPI']['Acquisition Parameters Common']['ImageCaputreDate'], "'%Y-%m-%d %H:%M:%S'").date()
        except:
            return datetime.strptime(
                metadata['dapi']['Acquisition Parameters Common']['ImageCaputreDate'], "'%Y-%m-%d %H:%M:%S'").date()


def parse_vial_id(field):
    value = field.values[0]
    if value and not math.isnan(value):
        return int(value)
    else:
        return None


def parse_result_notes(field):
    value = str(field.values[0])
    if value and not 'nan' == value:
        value = str(value)
        imaging_result = "ok" if value.lower() == "ok" else "fail"
        imaging_notes = value if imaging_result == "fail" else None
        return imaging_result, imaging_notes
    else:
        return None, None


def parse_date(field):
    value = field.values[0]
    if value and not math.isnan(value):
        return datetime.strptime(str(int(value)), '%y%m%d').date()
    else:
        return None


def parse_row_part(row, text):
    vial = parse_vial_id(row['Vial (' + text + ')'])
    result, notes = parse_result_notes(row['Result ' + text])
    date = parse_date(row['Date ' + text])
    return [vial, date, result, notes]


def parse_row(row):
    return [parse_row_part(row, '1st'), parse_row_part(row, '2nd')]


def lookup_data(data, vial_id, imaging_date):
    for row in data:
        if row[2] is not None:
            if row[0] == vial_id and (row[1] == imaging_date or row[1] == imaging_date - timedelta(days=1)):
                return row[0], row[1], row[2], row[3]


def get_imaging_result(file_name, outfile):
    vial_id, disc_no, hash_id, gene_name = split_file_name(file_name)
    metadata = load_metadata(file_name)
    imaging_date = get_imaging_date(metadata)
    try:
        row = csv[csv['Sample'] == hash_id]
        gene_name = row['Gene'].values[0]
        notes = row['Notes'].values[0].strip('\"\'') if row['Notes'].values[0] is not np.nan else 'None'
        result = {'Crop': row['Crop'].values[0].strip('\"\''),
                'Furrow': row['Furrow'].values[0].strip('\"\''),
                'mCherry': row['Ato'].values[0].strip('\"\''),
                'Venus': row['Venus'].values[0].strip('\"\''),
                'dsRed': row['dsRed'].values[0].strip('\"\''),
                'Usable': row['Usable'].values[0].strip('\"\'')}
    except:
        gene_name = 'unknown'
        notes = 'Imaging failed'
        result = {'Usable': 'no'}

    new_meta = {'Sample': hash_id, 'VialNo': vial_id, 'DiscNo': disc_no, 'GeneName': gene_name, 'Quality': result,
                'Notes': notes, 'Metadata': {}}
    new_meta['Metadata']['Microscope'] = metadata
    new_meta['Metadata']['Classification'] = \
        {'Classifier': 'DAPI-classifier.weka', 'FeatureSigmaMin': 1, 'FeatureSigmaMax': 16}
    new_meta['Metadata']['Classification']['Features'] = \
        ['Gaussian_blur', 'Hessian', 'Derivatives', 'Laplacian', 'Structure',
         'Edges', 'Difference_of_Gaussian', 'Mean', 'Median', 'Variance']
    new_meta['Metadata']['Segmentation'] = \
        {'DoGsigma': 8, 'DoGratio': 1.5, 'LocalMaxRadius': 3, 'MaskThreshold': 0.2, 'MaximaThreshold': 0.0}
    new_meta['Datasets'] = \
        {'File': os.path.basename(file_name).replace('.yml', '.h5'),
         'Raw': ['/raw/DAPI', '/raw/Venus', '/raw/mCherry'],
         'Scaled': ['/scaled/DAPI', '/scaled/Venus', '/scaled/mCherry'],
         'Segmentation': ['/segmentation/DoG', '/segmentation/mask', '/segmentation/maxima',
                          '/segmentation/objects'],
         'Classification': ['/weka/background', '/weka/nuclei']}
    new_meta['Objects'] = \
        {'Raw': os.path.basename(file_name).replace('.yml', '_raw.csv'),
         'Processed': os.path.basename(file_name).replace('.yml', '.csv')}

    return yaml.dump(new_meta, outfile, default_flow_style=False)


if __name__ == "__main__":

    csvFile = sys.argv[1]
    metadataDir = sys.argv[2]
    outputDir = sys.argv[3]
    csv = pd.read_csv(csvFile, sep=';')

    print("Processing " + metadataDir + " using annotations from " + csvFile + ".")

    samples = [f for f in os.listdir(metadataDir) if
               os.path.isfile(os.path.join(metadataDir, f)) and f.endswith(".yml")]

    for sample in samples:
        print(" - " + sample)
        input_file_name = os.path.join(metadataDir, sample)
        output_file_name = os.path.join(outputDir, sample)
        with open(output_file_name, 'w') as output_file:
            try:
                get_imaging_result(input_file_name, output_file)
            except:
                print("Something went wring with this file!")
