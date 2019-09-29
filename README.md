# Single-cell resolution view of the transcriptional landscape of developing Drosophila eye.
### Data analysis scripts for the RDN-WDP project

This repository contains the Python scripts used to manipulate segmented nuclear point clouds
from the RDP-WDP project. See the [paper repository](https://github.com/HassanLab/rdn-wdp-paper)
for the license and the software description.

## Data reproduction
First, download all data from the [data repository](https://github.com/HassanLab/rdn-wdp-data).

### Data normalization
For combining the normalized datasets use the [analyse2.py](analysis/analyse2.py) script.


### Combining the normalized datasets
For combining the normalized datasets use the [compine.py](analysis/combine.py) script.

`combine.py [-h] --dir DIR [--out OUT] [--log LOG]`

* `DIR` is the directory containing the normalized point clouds and the yaml metadata,
* `OUT` is the destination file name
* `LOG` is the optional logfile


 