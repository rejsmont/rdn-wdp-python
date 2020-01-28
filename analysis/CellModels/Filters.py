import pandas as pd
# from dask import dataframe as pd


class ColumnMappedList(list):
    """
    List that contains a column name field
    """
    def __init__(self, c, seq=()):
        super().__init__(seq)
        self._column = c

    @property
    def column(self):
        return self._column


class GenesColumnMappedList(ColumnMappedList):
    def __init__(self, seq=()):
        super().__init__('Gene', seq)


class SamplesColumnMappedList(ColumnMappedList):
    def __init__(self, seq=()):
        super().__init__('Sample', seq)


class QC:
    """
    Manually evaluated sample quality data, used in subsequent analysis
    """

    # Genes with clean Ato expression
    GENES_CLEAN = GenesColumnMappedList([
        'CG31176', 'beat-IIIc', 'king-tubby', 'lola-P', 'nmo', 'sNPF', 'Vn', 'Fas2', 'siz'
    ])

    # Genes for which wrong reporter expression pattern has been confirmed
    GENES_BAD = GenesColumnMappedList([
        'beat-IIIc', 'CG17378', 'CG31176', 'dap', 'lola-P', 'nmo', 'CG30343', 'phyl', 'siz', 'sNPF', 'spdo', 'Vn'
    ])

    # Genes with no reporter expression
    GENES_NO_TARGET = GenesColumnMappedList([
        'beat-IIIc', 'CG17378', 'CG31176', 'lola-P', 'nmo', 'CG30343', 'phyl', 'siz', 'sNPF', 'spdo', 'Vn'
    ])

    # Genes with known damage within the reporter construct
    GENES_DAMAGED = GenesColumnMappedList(['beat-IIIc', 'lola-P', 'nmo', 'siz', 'sNPF', 'Vn'])

    # Genes known to be expressed in the eye-disc, yet not reflected in reporter expression
    GENES_SHOULD = GenesColumnMappedList(['phyl', 'spdo'])

    # Samples for which segmentation has failed
    SAMPLES_BAD_SEGM = SamplesColumnMappedList([
        'J0RYWJ', '3SKX4V', '7AMINR', '4EAAEF', 'VH2DCR', 'WJ8F8M', 'ZNVOPe', 'APKoAe', 'zfroDh', 'lgxpL6', 'pcTNzE',
        '80IkVQ', 'UQZJ3K'
    ])

    # Samples for which classification has failed
    SAMPLES_BAD_CLASS = SamplesColumnMappedList([
        'hVJG7F', 't4JOuq', 'ylcNLd', '1NP0AI', '7SNHJY', 'BQ5PR8', 'DWYW27', 'I9BTQL', 'A5UIWC', 'KWGZHQ', 'M6GF25',
        'IJ5NUJ', 'WSSIG8', '3JR9EY', 'F7ZYLW', 'HDJ4XA', 'K2TCAP', 'OS7ENG', 'XC0ZUP', '61C1JS', '7RF4GF', 'AT7VB2',
        'CF4H1M', 'M946YU', 'S9PW43', '1BBD7G', '88Q70R', 'FC3922', 'QVBXHU', 'X88DWV', '05U0AU', '3LS10Z', 'CD2J9U',
        'BWLDNN', 'K0LKCY', 'KI9HFA', 'QTP049', 'SBD6VF', 'UUQTVA', 'ZL7PFF', 'CZXV1N', '2SZXIH', '4V7B84', '5VUZUB',
        'DGSCYR', 'FDMJRR', 'I0DO4Q', 'TAMCRX', '3SKX4V', '7AMINR', '80IkVQ', 'APKoAe', 'J0RYWJ', 'lgxpL6', 'VH2DCR',
        'zfroDh', 'ZNVOPe', '8CGDYO', '0F8P68', 'SD5LUQ', 'SN53VN', 'LVMMH9', 'LGNZBA', '1KAZBK', '8E3JMX', '9MCg4f',
        'T8USJQ', 'tXGki9', 'EMGTEW', 'JN0V15', 'PRCAE4', 'ergOuv', 'Ftc2VC', 'iBP4Fr', 'qqvRoI', 'V0qhzb', '1BLYXM',
        'CX8CT6', '8XHGAV', 'LMeg2F', 'R6F3J5'
    ])

    # Samples with marginally satisfactory classification results
    SAMPLES_MARGINAL_CLASS = SamplesColumnMappedList([
        'TY2COS', 'YZDFTH', 'NSOK5I', 'RVOX2M', 'D0CTSD', 'H7WOUU', 'RWPK00', '4SDPFB', 'H9OS9A', 'IU43O6', 'OHXME8',
        'TYEOK6', 'VDOERX', 'hsZ6pl', 'R4JB64', 'RKNAC2', '64EQVJ', 'lCUF2l', 'HG24ZO', 'TWP10V', '3QOEXZ', '8HTKXA',
        'ORR8VN', '5IEVIW', 'ETUQB6', 'VK3DU4', 'PV2RPF', 'QLOP4L', 'W4CQNL', 'Y7UX2N', 'Z2JP1B', '4FYXUO', 'HJ4QFN',
        'JP0OGS', '3ALXQE', 'FIQKAB', 'SZ3DCM', 'XRQVVZ', '4CNFTO', '6PC2MI', 'JLCD5B', '249YNC', '5PL9UP', 'AVTGN7',
        'QJ2QGS', '05O5MD', 'S2QZ2B', '9899TQ', '0obMbL', 'KJB859', 'OUJ0DB', '1KpdWQ', '502MKQ', '0687TE', '9473S9',
        'VOK2S9', 'J55205', '0NDb1T', '2441VU', 'HCABGW', 'OFTUDJ', 'S5CO9J', 'VD2V13',
    ])


class MinMax:
    """
    Simple class to store min and max values
    """

    def __init__(self, v_min, v_max, column):
        self._min = v_min
        self._max = v_max
        self._column = column

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def column(self):
        return self._column


class Morphology:
    """
    Morphological constraint definitions
    """

    # Broad area of the morphogenetic furrow (roughly equivalent to Ato expression zone)
    FURROW_BROAD = MinMax(-8.0, 8.0, 'cy')
    # Center of the morphogenetic furrow, an area with maximum Ato expression
    FURROW_PEAK = MinMax(-2.0, 2.0, 'cy')
    # The morphogenetic furrow
    FURROW = MinMax(-4.0, 4.0, 'cy')


class Masks:
    """
    Class defining common masks used to filter cells
    """

    def __init__(self, cells: pd.DataFrame):
        self._cells = cells
        self._genes_clean = None
        self._genes_bad = None
        self._genes_no_target = None
        self._genes_damaged = None
        self._genes_should = None
        self._samples_bad_segm = None
        self._samples_bad_class = None
        self._samples_marginal_class = None
        self._cells_mf_area = None

    def _mask_in_getter(self, p, i):
        v = getattr(self, p)
        if v is None:
            v = self._cells[i.column].isin(i)
            setattr(self, p, v)
        return v

    def _mask_min_max_getter(self, p, m: MinMax):
        v = getattr(self, p)
        if v is None:
            v = (self._cells[m.column] >= m.min) & (self._cells[m.column] <= m.max)
            setattr(self, p, v)
        return v

    # Gene-specific filters
    @property
    def genes_clean(self):
        return self._mask_in_getter('_genes_clean', QC.GENES_CLEAN)

    @property
    def genes_bad(self):
        return self._mask_in_getter('_genes_bad', QC.GENES_BAD)

    @property
    def genes_no_target(self):
        return self._mask_in_getter('_genes_no_target', QC.GENES_NO_TARGET)

    @property
    def genes_damaged(self):
        return self._mask_in_getter('_genes_damaged', QC.GENES_DAMAGED)

    @property
    def genes_should(self):
        return self._mask_in_getter('_genes_should', QC.GENES_SHOULD)

    # Sample-specific filters
    @property
    def samples_bad_segm(self):
        return self._mask_in_getter('_samples_bad_segm', QC.SAMPLES_BAD_SEGM)

    @property
    def samples_bad_class(self):
        return self._mask_in_getter('_samples_bad_class', QC.SAMPLES_BAD_CLASS)

    @property
    def samples_marginal_class(self):
        return self._mask_in_getter('_samples_marginal_class', QC.SAMPLES_MARGINAL_CLASS)

    # Morphological filters
    @property
    def cells_mf_area(self):
        return self._mask_min_max_getter('_cells_mf_area', Morphology.FURROW_BROAD)
