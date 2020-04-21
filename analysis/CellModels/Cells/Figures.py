import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib import colors

from CellModels.Clustering.Data import ClusteringResult


class MultiFigurePdf:

    def __init__(self, figures):
        self._figures = figures

    def save(self, path):
        with PdfPages(path) as pdf:
            for f in self._figures:
                pdf.savefig(f.fig)
                f.close()


class Figure:

    def __init__(self, data):
        self._logger = logging.getLogger('rdn-wdp-figures')
        self._logger.debug('Initializing ' + self.name())
        self._data = data
        self._fig = None
        self._plotted = False

    def show(self):
        if not self._plotted:
            self.plot()
        self._logger.info('Showing ' + self.name())
        self._fig.show()

    def save(self, path):
        if not self._plotted:
            self.plot()
        self._logger.info('Saving ' + self.name() + ' to ' + path)
        self._fig.savefig(path)

    def plot(self):
        self._logger.info('Plotting ' + self.name())
        self._fig = plt.figure(figsize=self._size())
        ax = self.plot_axes()
        self._plotted = True
        return ax

    def plot_axes(self, ax=None):
        if ax is None:
            ax = self._get_ax(rect=[0, 0, 1, 1])
        return ax

    def name(self):
        return self.__class__.__name__

    def close(self):
        if self._plotted:
            plt.close(self._fig)
            self._plotted = False

    def _get_ax(self, *args, **kwargs):
        ax = self._fig.add_axes(plt.Axes(self._fig, *args, **kwargs))
        return ax

    def _size(self):
        return None

    @property
    def fig(self):
        if not self._plotted:
            self.plot()
        return self._fig


class LogScaleGenePlot:

    @staticmethod
    def cmap(): return 'plasma'

    @staticmethod
    def v_lim(): return [0.1, 20]

    @staticmethod
    def v_scale(): return 'log'

    @staticmethod
    def v_ticks(): return [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]

    @staticmethod
    def v_minor_ticks(): return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]

    @staticmethod
    @ticker.FuncFormatter
    def major_formatter_log(x, pos):
        return "%g" % (round(x * 10) / 10)

    def v_axis_formatter(self): return self.major_formatter_log


class LogScaleExtPlot(LogScaleGenePlot):

    @staticmethod
    def cmap(): return 'viridis'

    @staticmethod
    def v_lim(): return [0.1, 20]

    @staticmethod
    def v_minor_ticks(): return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]


class DiscPlot(Figure):
    GX_MIN = 0
    GX_MAX = 80
    GY_MIN = -10
    GY_MAX = 45
    GY_CMAX = 25

    def __init__(self, data, sample=None, cells=None):
        super().__init__(data)
        if cells is None:
            if sample is None:
                self._cells = self._data.cells
            else:
                idx = pd.IndexSlice
                self._cells = self._data.cells.loc[idx[:, sample], :]
        else:
            self._cells = cells
        self._ax = None
        self._sc = None

    def plot_axes(self, ax=None):
        if ax is None:
            ax = self._get_ax(rect=[0, 0, 1, 1])
        self._sc = ax.scatter(self._get_x(), self._get_y(),
                              c=self._get_c(), **self._get_sc_params())
        ax.set_xlim(self.GX_MIN, self.GX_MAX)
        ax.set_ylim(self.GY_MAX, self.GY_MIN)
        ax.set_ylabel('A-P position')
        ax.set_xlabel('D-V position')
        self._ax = ax
        return ax

    def _get_c(self):
        return None

    def _get_x(self):
        return self._cells[('Position', 'Normalized', 'x')]

    def _get_y(self):
        return self._cells[('Position', 'Normalized', 'y')]

    def _get_sc_params(self):
        return {}


class SamplePlot(DiscPlot, LogScaleGenePlot):

    def __init__(self, data, sample=None, field=None, cells=None):
        super().__init__(data, sample, cells)

        if field is None:

            def find_field(lst):
                for c in lst:
                    if 'Measurements' in c and 'Normalized' in c:
                        return c

            if isinstance(data, ClusteringResult):
                self._field = find_field(self._data.config.hc_features)
            else:
                self._field = find_field(self._data.cells.columns)
        else:
            self._field = field
        self._cells = self._cells.sort_values(by=[self._field])

    def _get_c(self):
        return self._cells[self._field]

    def _get_sc_params(self):
        return {
            'norm': colors.LogNorm(*self.v_lim()),
            'cmap': self.cmap(),
            's': 5
        }

    def plot(self):
        ax = super().plot()
        self.plot_colorbar()
        return ax

    def plot_colorbar(self, cax=None, orientation='vertical', fig=None):
        fig = self._fig if fig is None else fig
        ax = self._ax if cax is None else None
        cb = fig.colorbar(self._sc, cax=cax, ax=ax, orientation=orientation,
                          ticks=self.v_ticks(), format=self.major_formatter_log)
        cb.set_label(label=str(self._field[-1]) + ' expression')
