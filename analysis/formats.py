class GFFLikeFeature:

    _FNAMES = []
    _FALIASES = {'chromosome': 'chrom'}

    def __init__(self, fields):
        self._fields = fields

    def __getattr__(self, item):
        if item in self._FALIASES:
            item = self._FALIASES[item]
        try:
            return self._fields[self._FNAMES.index(item)]
        except ValueError:
            return None

    def __str__(self):
        string = ''
        for index, field in enumerate(self._FNAMES):
            if string != '':
                string += '\t'
            string += str(self._fields[index])

        return string


class BED6Feature(GFFLikeFeature):

    _FNAMES = ['chrom', 'start', 'end', 'name', 'score', 'strand']

    def __eq__(self, other):
        identical = getattr(self, '_FNAMES', None) == getattr(other, '_FNAMES', None)
        if not identical:
            return False

        for field in self._FNAMES:
            identical &= getattr(self, field) == getattr(other, field)

        return identical

    def __repr__(self):
        r = self.name + ':' + self.start + '..' + self.end
        if self.strand != '.':
            r += '(' + self.strand + ')'
        return r


class NarrowPeak(BED6Feature):

    _FNAMES = BED6Feature._FNAMES + ['enrichment', 'p', 'q', 'peak']
    _FALIASES = {'chromosome': 'chrom',
                 'p-value': 'p',
                 'q-value': 'q',
                 'signal': 'enrichment',
                 'value': 'enrichment',
                 'signal_value': 'enrichment'}


class GFFFeature(GFFLikeFeature):

    _FNAMES = ['chrom', 'source', 'type', 'start', 'end', 'score', 'strand', 'frame', 'group']
    _FALIASES = {'chromosome': 'chrom'}

    _attributes = None

    def __init__(self, fields):
        super().__init__(fields)
        self._fields[3] = int(fields[3]) - 1
        self._fields[4] = int(fields[4]) - 1

    def __getattr__(self, item):
        value = super().__getattr__(item)
        if value is None and self._attributes is not None:
            return getattr(self._attributes, item)
        else:
            return None

    def __str__(self):
        string = super().__str__()
        if self._attributes:
            string += '\t' + str(self._attributes)

        return string

    def __repr__(self):
        r = self.type + ':' + self.start + '..' + self.end
        if self.strand != '.':
            r += '(' + self.strand + ')'
        return r


class GFF3Feature(GFFFeature):

    _FNAMES = GFFFeature._FNAMES[0:8]

    def __init__(self, fields):
        super().__init__(fields)
        self._attributes = GFF3Attributes(fields[8])
        self._fields.pop(8)


class GTFFeature(GFFFeature):

    _FNAMES = GFFFeature._FNAMES[0:8]

    def __init__(self, fields):
        super().__init__(fields)
        self._attributes = GTFAttributes(fields[8])
        self._fields.pop(8)


class GFF3Attributes:

    _ALIASES = {}

    def __init__(self, attributes):
        self._attributes = {}
        for attr in attributes.replace('; ', ';').split(';'):
            self.append_attribute(attr)

    def __getattr__(self, item):
        if item in self._ALIASES:
            item = self._ALIASES[item]
        try:
            return self._attributes[item]
        except KeyError:
            return None

    def __str__(self):
        string = ''
        for field, value in self._attributes.items():
            if string != '':
                string += '; '
            string += field + '=\"' + str(value) + '\"'

        return string

    @staticmethod
    def parse_attribute(attribute):
        split = attribute.split('=')
        split = [x.strip('"') for x in split]
        return split.pop(0), '='.join(split)

    def append_attribute(self, attribute, value=None):
        if not attribute:
            return
        if value is None:
            attribute, value = self.parse_attribute(attribute)
        if attribute not in self._attributes.keys():
            self._attributes[attribute] = value
        elif isinstance(self._attributes[attribute], list):
            self._attributes[attribute].append(value)
        else:
            self._attributes[attribute] = [self._attributes[attribute]]


class GTFAttributes(GFF3Attributes):

    _ALIASES = {'name': 'gene_name',
                'gene': 'gene_id',
                'transcript': 'transcript_id',
                'protein': 'protein_id',
                'tss': 'tss_id'}

    @staticmethod
    def parse_attribute(attribute):
        split = attribute.split(' ')
        split = [x.strip('"') for x in split]
        return split.pop(0), ' '.join(split)

    def __str__(self):
        string = ''
        for field, value in self._attributes.items():
            if string != '':
                string += '; '
            string += field + ' \"' + str(value) + '\"'
        string += ';'

        return string
