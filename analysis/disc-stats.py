#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *

plt.figure()

discs = pd.read_excel('/Users/radoslaw.ejsmont/Desktop/rdn-wdp/dog-param-final.xlsx', usecols=list(range(14)))

#discs.rename(columns={
#        'dapi_mean': 'dapi_integral',
#        'mcherry_mean': 'mcherry_integral',
#        'venus_mean': 'venus_integral'},
#    inplace=True)
#discs['dapi_mean'] = discs['dapi_integral'] / discs['volume_mean']
#discs['mcherry_mean'] = discs['mcherry_integral'] / discs['volume_mean']
#discs['venus_mean'] = discs['venus_integral'] / discs['volume_mean']


# ax = discs.plot(kind='scatter', x='nuclei_count', y='volume_mean', title='Nuclear volume vs nuclei count')
# ax.set_xlabel('Count')
# ax.set_ylabel('Volume [vx]')
#
# csv_dir = '/Users/radoslaw.ejsmont/Desktop/dog-all'
#
# print('Loading data: ')
#
# ndflist = []
# for csv_file in os.listdir(csv_dir):
#     print('.', end='', flush=True)
#     nuclei = pd.read_csv(os.path.join(csv_dir, csv_file))
#     ndflist.append(nuclei)
# nuclei = pd.concat(ndflist, ignore_index=True)
# ndflist = None
#
# volume = nuclei['Volume'].to_frame()
# volume_mean = volume.mean()
# volume_mad = volume.mad()
# count = discs['nuclei_count'].to_frame()
#
# ax.plot([count.min(), count.max()], [volume_mean - volume_mad, volume_mean - volume_mad], c='r')
# ax.plot([count.min(), count.max()], [volume_mean + volume_mad, volume_mean + volume_mad], c='r')
#
# print('')
#
# plt.savefig('/Users/radoslaw.ejsmont/Desktop/count_volume.pdf')
#
# plt.figure()
# ax = discs.plot(kind='scatter', x='dapi_mean', y='volume_mean', title='Nuclear volume vs DAPI intensity')
# ax.set_xlabel('Mean DAPI intensity [au]')
# ax.set_ylabel('Volume [vx]')
# plt.savefig('/Users/radoslaw.ejsmont/Desktop/dapi_volume.pdf')
#
# plt.figure()
# ax = discs.plot(kind='scatter', x='dapi_mean', y='nuclei_count', title='Nuclear count vs DAPI intensity')
# ax.set_xlabel('Mean DAPI intensity [au]')
# ax.set_ylabel('Count')
# plt.savefig('/Users/radoslaw.ejsmont/Desktop/dapi_count.pdf')
