import pandas as pd


class CellColumns:
    _cols = {
        'cx_orig': ('Position', 'Raw', 'x'),
        'cy_orig': ('Position', 'Raw', 'y'),
        'cz_orig': ('Position', 'Raw', 'z'),
        'cx': ('Position', 'Normalized', 'x'),
        'cy': ('Position', 'Normalized', 'y'),
        'cz': ('Position', 'Normalized', 'z'),
        'cy_scaled': ('Position', 'Scaled', 'y'),
        'Volume': ('Measurements', 'Raw', 'Volume'),
        'mCherry_orig': ('Measurements', 'Raw', 'Venus'),
        'Venus_orig': ('Measurements', 'Raw', 'mCherry'),
        'DAPI_orig': ('Measurements', 'Raw', 'DAPI'),
        'mCherry': ('Measurements', 'Normalized', 'mCherry'),
        'Venus': ('Measurements', 'Normalized', 'Venus'),
        'DAPI': ('Measurements', 'Normalized', 'DAPI'),
        'ext_mCherry': ('Measurements', 'Prominence', 'mCherry'),
        'ext_Venus': ('Measurements', 'Prominence', 'Venus'),
        'ang_max_mCherry': ('Measurements', 'Angle', 'mCherry'),
        'ang_max_Venus': ('Measurements', 'Angle', 'Venus')
    }

    @classmethod
    def _multi_index(cls, index, misc=None):
        if misc is None:
            misc = {}
        tuples = cls._t_list(index, misc)

        return pd.MultiIndex.from_tuples(tuples)

    @classmethod
    def _t_list(cls, lst, misc=None):
        if misc is None:
            misc = {}
        tuples = []
        for i in lst:
            if i in cls._cols.keys():
                tuples.append(cls._cols[i])
            elif i in misc.keys():
                tuples.append(misc[i])
            else:
                tuples.append(i)

        return tuples
