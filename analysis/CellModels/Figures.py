import logging
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import savgol_filter
from data import DiscData


class Figure:

    def __init__(self, data):
        self.logger = logging.getLogger('rdn-wdp-figures')
        self.logger.debug('Initializing ' + self.name())
        self.data = data
        self.fig = None
        self.gs = None
        self.plotted = False

    def show(self):
        if not self.plotted:
            self.plot()
        self.logger.info('Showing ' + self.name())
        self.fig.show()

    def save(self, path):
        if not self.plotted:
            self.plot()
        self.logger.info('Saving ' + self.name() + ' to ' + path)
        self.fig.savefig(path)

    def plot(self):
        self.logger.info('Plotting ' + self.name())
        self.plotted = True
        pass

    def name(self):
        return self.__class__.__name__

    def ax(self, *args, **kwargs):
        ax = plt.Axes(self.fig, *args, **kwargs)
        self.fig.add_axes(ax)
        return ax


class Plot:

    def __init__(self, fig, data):
        self.fig = fig
        self.data = data
        self.ax = None

    def plot(self, position, *args, **kwargs):
        if isinstance(position, gridspec.SubplotSpec):
            self.ax = self.fig.add_subplot(position)
        else:
            self.ax = position

    def legend(self, position, *args, **kwargs):
        pass

    @staticmethod
    def x_lim(): return False

    @staticmethod
    def y_lim(): return False

    @staticmethod
    def v_lim(): return False

    @staticmethod
    def x_scale(): return False

    @staticmethod
    def y_scale(): return False

    @staticmethod
    def v_scale(): return False

    @staticmethod
    def x_ticks(): return False

    @staticmethod
    def y_ticks(): return False

    @staticmethod
    def v_ticks(): return False

    @staticmethod
    def v_minor_ticks(): return False

    def v_axis_formatter(self): return False


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


class ProfilePlot(Plot):
    """
    Plot gene expression profile
    """

    def __init__(self, fig, data, styles=None):
        super().__init__(fig, data)
        self._styles = styles if styles is not None else False

    def plot(self, position, *args, **kwargs):
        super().plot(position, *args, **kwargs)
        self.plot_profiles()
        self.format_axis()

    def legend(self, position, ncol=1, loc='upper center', *args, **kwargs):
        if isinstance(position, gridspec.SubplotSpec):
            ax = self.fig.add_subplot(position)
        else:
            ax = position
        ax.set_axis_off()
        handles, labels = self.ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=ncol, loc=loc, frameon=False, fontsize=18)

    def plot_profile(self, profile, style=None):
        data = self.data[profile].dropna()
        x = data.index
        y = self.preprocessor(data.values)
        style = style if style is not None else {}
        self.ax.plot(x, y, label=profile, **style)

    def plot_profiles(self):
        styles = self.styles()
        for profile in self.data:
            style = styles[profile] if styles and styles[profile] else None
            self.plot_profile(profile, style=style)
        return self.ax.get_legend_handles_labels()

    def format_axis(self):
        if self.x_lim():
            self.ax.set_xlim(self.x_lim())
        if self.v_lim():
            self.ax.set_ylim(self.v_lim())
        if self.v_scale():
            self.ax.set_yscale(self.v_scale())
        if self.v_ticks():
            self.ax.set_yticks(self.v_ticks())
        if self.v_axis_formatter():
            self.ax.yaxis.set_major_formatter(self.v_axis_formatter())

    @staticmethod
    def preprocessor(x):
        return x

    def styles(self):
        return self._styles


class APProfilePlot(ProfilePlot):
    @staticmethod
    def x_lim():
        return [GY_MIN, GY_MAX]


class MFProfilePlot(ProfilePlot):
    @staticmethod
    def x_lim():
        return [DiscData.FURROW_MIN, DiscData.FURROW_MAX]


class DVProfilePlot(ProfilePlot):
    @staticmethod
    def x_lim():
        return [GX_MIN, GX_MAX]


class SmoothProfilePlot(ProfilePlot):
    @staticmethod
    def preprocessor(x):
        return savgol_filter(x, 5, 3, mode='nearest')


class DiscThumb(Plot):
    """
    Plot a disc thumbnail
    """
    def __init__(self, fig, data, title):
        super().__init__(fig, data)
        self.img = None
        self.index = self.data.index.to_frame()
        self.extent = [self.index['cx'].min(), self.index['cx'].max(),
                       self.index['cy'].max(), self.index['cy'].min()]
        self.title = title

    def plot(self, position, *args, **kwargs):
        super().plot(position, color='white', *args, **kwargs)
        matrix = self.disc_matrix()
        self.img = self.ax.imshow(matrix, extent=self.extent, norm=self.norm(), cmap=self.cmap(), aspect='auto')
        self.ax.set_facecolor('black')
        self.ax.set_xlim(self.x_lim())
        self.ax.set_ylim(self.y_lim())

    def legend(self, position, *args, **kwargs):
        if isinstance(position, gridspec.SubplotSpec):
            ax = self.fig.add_subplot(position)
        else:
            ax = position
        cb = self.fig.colorbar(self.img, cax=ax, orientation='horizontal', ticks=self.v_ticks(),
                               format=self.v_axis_formatter())
        cb.set_label(label=self.title, **kwargs)
        if self.v_scale() == 'log' and self.v_minor_ticks():
            ticks = self.img.norm(self.v_minor_ticks())
            cb.ax.xaxis.set_ticks(ticks, minor=True)

    def disc_matrix(self):
        x = self.index['cx']
        y = self.index['cy'] - self.extent[3]
        v = self.data
        matrix = np.full([self.extent[2] - self.extent[3] + 1, self.extent[1] - self.extent[0] + 1], np.NaN)
        matrix[y, x] = v
        raw = matrix.copy()
        nan = np.argwhere(np.isnan(matrix))
        x_max, y_max = matrix.shape
        for x, y in nan:
            xs = max([0, x-1])
            xe = min([x+1, x_max-1])
            ys = max([0, y - 1])
            ye = min([y + 1, y_max-1])
            kernel = raw[xs:xe, ys:ye]
            kernel = kernel[~np.isnan(kernel)]
            if kernel.size > 0:
                matrix[x, y] = np.mean(kernel)
            else:
                matrix[x, y] = 0
        return matrix

    def norm(self):
        v_scale = self.v_scale()
        v_lim = self.v_lim()
        if v_lim is not False:
            v_min, v_max = v_lim
        else:
            v_min = None
            v_max = None
        norm = None
        if v_scale is False or v_scale == 'linear':
            norm = colors.Normalize(vmin=v_min, vmax=v_max)
        elif v_scale == 'log':
            norm = colors.LogNorm(vmin=v_min, vmax=v_max)
        else:
            raise NotImplementedError("Override `norm` method to implement %s v_scale", v_scale)
        return norm

    @staticmethod
    def x_lim(): return [GX_MIN, GX_MAX]

    @staticmethod
    def y_lim(): return [GY_MAX, GY_MIN]

    @staticmethod
    def cmap(): return None


class LabeledPlot(Plot):

    def plot(self, position, text=None, color='black', *args, **kwargs):
        super().plot(position, *args, **kwargs)
        self.ax.text(0.025, 0.95, text, horizontalalignment='left', verticalalignment='top', fontsize=24,
                     color=color, transform=self.ax.transAxes)


class MultiCellPlot(Plot):

    def plot(self, position, firstrow=False, firstcol=True, lastrow=False, lastcol=False, controw=False,
             label='left', *args, **kwargs):
        super().plot(position, *args, **kwargs)
        self.ax.tick_params(bottom=(lastrow or controw),
                            top=(firstrow and not lastrow),
                            labelbottom=lastrow,
                            labeltop=(firstrow and not lastrow),
                            left=(label == 'left'),
                            right=(label == 'right'),
                            labelleft=(firstcol and (label == 'left')),
                            labelright=(lastcol and (label == 'right')))
        if self.v_scale() == 'log':
            self.ax.tick_params(axis='y', which='minor', left=(label == 'left'), right=(label == 'right'),
                                labelleft=False, labelright=False)
