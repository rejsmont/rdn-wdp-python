import numpy as np
import pandas as pd
import math
import os
from sklearn.linear_model import LinearRegression

orig_dir = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/samples'
proc_dir = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/samples/processed'
combined = '/Users/rejsmont/Google Drive File Stream/My Drive/Projects/RDN-WDP/processing/samples-combined.csv'


def read_data(orig, proc, comb):
    c_orig = read_dir(orig)
    c_proc = read_dir(proc)
    c_comb = pd.read_csv(comb)

    return c_orig, c_proc, c_comb


def read_dir(d):
    d_cells = pd.DataFrame()
    for f in os.listdir(d):
        if f.endswith('.csv') and '_' in f:
            s = pd.read_csv(os.path.join(d, f))
            if '_normalized.csv' in f:
                s_loc = -2
            else:
                s_loc = -1
            s['Sample'] = str(f).split('_')[s_loc].replace('.csv', '')
            d_cells = d_cells.append(s)
    d_cells.reindex()
    return d_cells


c_orig, c_proc, c_comb = read_data(orig_dir, proc_dir, combined)
c_proc = c_proc.rename(index=str, columns={"Unnamed: 0": "Particle"})
c_proc['Particle'] = c_proc['Particle'] + 1
c_orig = c_orig.rename(index=str, columns={'Mean 1': "mCherry", "Mean 2": "Venus", "Mean 0": "DAPI"})
o_p = c_orig.merge(c_proc, 'left', ['Sample', 'Particle', 'Volume'], suffixes=('_orig', ''))
res = o_p.merge(c_comb, 'left', ['Sample', 'cx', 'cy', 'cz'], suffixes=('', '_comb'))

c_orig, c_proc, c_comb, o_p = None, None, None, None

fields = ['cx', 'cy', 'cz', 'mCherry', 'Venus', 'DAPI']
for sample in res['Sample'].unique():
    data = res[res['Sample'] == sample]
    gene = data['Gene'].dropna().unique()
    if len(gene) > 0:
        gene = gene[0]
        res.loc[res['Sample'] == sample, 'Gene'] = gene
    print("Processing", sample, gene)
    coeffs = {}
    for field in fields:
        field_orig = field + '_orig'
        field_pred = field + '_scaled'
        pairs = data[[field_orig, field]].dropna()
        if len(pairs) == 0:
            continue
        x = pairs[field_orig].values.reshape(-1, 1)
        y = pairs[field].values.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(x, y)
        print(reg.coef_.flatten())
        coeffs[field] = (reg.coef_.flatten()[0], reg.intercept_.flatten()[0])

    if 'cy' in coeffs.keys():
        coeffs['cy_shift'] = coeffs['cy']
        coeffs['cy'] = coeffs['cz'] * np.sign(coeffs['cy'])

    print(coeffs)

    for field in fields:
        if field in coeffs.keys():
            field_orig = field + '_orig'
            field_pred = field + '_scaled'
            res.loc[res['Sample'] == sample, field_pred] = data[field_orig] * coeffs[field][0] + coeffs[field][1]

result = res[['Gene', 'Sample', 'Particle', 'cx_orig', 'cy_orig', 'cz_orig', 'cx_scaled', 'cy', 'cz_scaled',
              'cy_scaled', 'Volume', 'mCherry_orig', 'Venus_orig', 'DAPI_orig', 'mCherry_scaled', 'Venus_scaled',
              'DAPI_scaled', 'ext_mCherry', 'ext_Venus', 'ang_max_mCherry', 'ang_max_Venus']]

result = result.rename(index=str, columns={'Particle': 'Nucleus', 'cx_scaled': 'cx', 'cz_scaled': 'cz',
                                           'mCherry_scaled': 'mCherry', 'Venus_scaled': 'Venus',
                                           'DAPI_scaled': 'DAPI'})
