import ccmpred.logo
import plotly.graph_objs as go
import os
import sys
from plotly.offline import plot as plotly_plot

class Progress():
    """

    """

    def __init__(self, plotfile=None, **kwargs):


        self.plotfile = plotfile

        self.optimization_log={}
        self.optimization_log['||x||'] = []
        self.optimization_log['||x_single||'] = []
        self.optimization_log['||x_pair||'] = []
        self.optimization_log['||g||'] = []
        self.optimization_log['||g_single||'] = []
        self.optimization_log['||g_pair||'] = []

        self.optimization_log.update(kwargs)

        self.plot_metrics = []
        self.subtitle =  ""


    def begin_process(self):

        headerline ="{0:>{1}s}".format('iter', 8)
        headerline += (" ".join("{0:>{1}s}".format(ht, 12) for ht in sorted(self.optimization_log.keys())))

        if ccmpred.logo.is_tty:
            print("\x1b[2;37m{0}\x1b[0m".format(headerline))
        else:
            print(headerline)

    def plot_options(self, plotfile, plot_metrics, subtitle):
        self.plotfile = plotfile
        self.plot_metrics = plot_metrics
        self.subtitle =  subtitle


    def log_progress(self, n_iter, xnorm_single, xnorm_pair, gnorm_single, gnorm_pair, **kwargs):


        xnorm = xnorm_single + xnorm_pair
        gnorm = gnorm_single + gnorm_pair


        self.optimization_log['||x||'].append(xnorm)
        self.optimization_log['||x_single||'].append(xnorm_single)
        self.optimization_log['||x_pair||'].append(xnorm_pair)
        self.optimization_log['||g||'].append(gnorm)
        self.optimization_log['||g_single||'].append(gnorm_single)
        self.optimization_log['||g_pair||'].append(gnorm_pair)

        for name, metric in kwargs.iteritems():
            self.optimization_log[name].append(metric)


        log = "{0:>{1}}".format(n_iter, '8g')
        print log + " ".join("{0:>{1}}".format(self.optimization_log[key][-1], '12g') for key in sorted(self.optimization_log.keys()))

        if self.plotfile is not None:
            self.plot_progress()


        sys.stdout.flush()


    def plot_progress(self):
        protein = os.path.basename(self.plotfile).split(".")[0]
        title = "Optimization Log for {0} ".format(protein)
        title += self.subtitle

        data = []
        for metric in self.plot_metrics:
            data.append(
                go.Scatter(
                    x=range(1, len(self.optimization_log[metric]) + 1),
                    y=self.optimization_log[metric],
                    mode='lines',
                    visible="legendonly",
                    name=metric
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
            )
        }

        plotly_plot(plot, filename=self.plotfile, auto_open=False)
