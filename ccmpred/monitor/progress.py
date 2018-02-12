import ccmpred.logo
import plotly.graph_objs as go
import os
import sys
from plotly.offline import plot as plotly_plot


class Progress():
    """
    Plot the progress as plotly graph
    """

    def __init__(self , plotfile, title):

        self.optimization_log={}
        self.plotfile=plotfile
        self.title=title

    def print_header(self):

        headerline ="{0:>{1}s}".format('iter', 8)
        headerline += (" ".join("{0:>{1}s}".format(ht, 14) for ht in sorted(self.optimization_log.keys())))

        if ccmpred.logo.is_tty:
            print("\x1b[2;37m{0}\x1b[0m".format(headerline))
        else:
            print(headerline)

    def set_plot_title(self, title):
        self.title=title

    def init_log(self, **kwargs):
        for name in kwargs.keys():
            self.optimization_log[name] = []

        self.print_header()


    def log_progress(self, n_iter, **kwargs):

        if len(self.optimization_log) == 0:
            self.init_log(**kwargs)

        if (n_iter != 0) and (n_iter % 100 == 0):
            self.print_header()


        log = "{0:>{1}}".format(n_iter, '8g')
        for name, metric in sorted(kwargs.items()):
            self.optimization_log[name].append(metric)
            log += "{0:>{1}}".format(metric, '15g')
        print(log)

        # log = "{0:>{1}}".format(n_iter, '8g')
        # print(log + " ".join("{0:>{1}}".format(self.optimization_log[key][-1], '15g') for key in sorted(self.optimization_log.keys())))

        if self.plotfile is not None:
            self.plot_progress()

        sys.stdout.flush()


    def plot_progress(self, ):

        protein = os.path.basename(self.plotfile).split(".")[0]
        title = "Optimization Log for {0} ".format(protein)
        title += self.title

        data = []
        for name, metric in self.optimization_log.items():
            data.append(
                go.Scatter(
                    x=range(1, len(self.optimization_log[name]) + 1),
                    y=metric,
                    mode='lines',
                    visible="legendonly",
                    name=name
                )
            )

        plot = {
            "data": data,
            "layout": go.Layout(
                title=title,
                xaxis1=dict(
                    title="iteration",
                    exponentformat="e",
                    showexponent='All'
                ),
                yaxis1=dict(
                    title="metric",
                    exponentformat="e",
                    showexponent='All'
                ),
                font=dict(size=18),
                titlefont=dict(size=14)
            )
        }

        plotly_plot(plot, filename=self.plotfile, auto_open=False)
