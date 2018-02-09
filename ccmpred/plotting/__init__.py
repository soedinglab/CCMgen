import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot
from plotly import tools
import ccmpred.io as io
import numpy as np

def plot_empirical_vs_model_statistics(
        single_freq_observed, single_freq_sampled,
        pairwise_freq_observed, pairwise_freq_sampled,
        title, plot_out, log=False):


    L = single_freq_observed.shape[0]

    ## first trace for single amino acid frequencies
    trace_single_frequencies = go.Scattergl(
        x=single_freq_observed.flatten().tolist(),
        y=single_freq_sampled.flatten().tolist(),
        mode='markers',
        name='single frequencies',
        text=["position: {0}<br>amino acid: {1}".format(i+1,io.AMINO_ACIDS[a]) for i in range(L) for a in range(20)],
        opacity=0.3,
        showlegend=True
    )
    pearson_corr_single = np.corrcoef(single_freq_observed.flatten().tolist(), single_freq_sampled.flatten().tolist())[0,1]


    ## second trace for single amino acid frequencies
    indices_upper_triangle = np.triu_indices(L, k=1)
    pair_freq_observed = pairwise_freq_observed[indices_upper_triangle[0], indices_upper_triangle[1], :, :].flatten().tolist()
    pair_freq_sampled = pairwise_freq_sampled[indices_upper_triangle[0], indices_upper_triangle[1], :, :].flatten().tolist()
    parir_freq_annotation = ["position: {0}-{1}<br>amino acid: {2}-{3}".format(i+1,j+1, io.AMINO_ACIDS[a], io.AMINO_ACIDS[b]) for i in range(L-1) for j in range(i+1, L) for a in range(20) for b in range(20)]
    trace_pairwise_frequencies = go.Scattergl(
        x=pair_freq_observed,
        y=pair_freq_sampled,
        mode='markers',
        name='pairwise frequencies',
        text=parir_freq_annotation,
        opacity=0.3,
        showlegend=True
    )
    pearson_corr_pair = np.corrcoef(pair_freq_observed, pair_freq_sampled)[0, 1]

    ## third trace for covariances
    cov_observed = [pairwise_freq_observed[i,j,a,b] - (single_freq_observed[i,a] * single_freq_observed[j,b])   for i in range(L-1) for j in range(i+1, L) for a in range(20) for b in range(20)]
    cov_sampled  = [pairwise_freq_sampled[i,j,a,b] - (single_freq_sampled[i,a] * single_freq_sampled[j,b])   for i in range(L-1) for j in range(i+1, L) for a in range(20) for b in range(20)]
    trace_cov = go.Scattergl(
        x=cov_observed,
        y=cov_sampled,
        mode='markers',
        name='covariances',
        text=parir_freq_annotation,
        opacity=0.3,
        showlegend=True
    )
    pearson_corr_cov = np.corrcoef(cov_observed, cov_sampled)[0, 1]

    ## diagonal that represents perfect correlation
    diagonal = go.Scattergl(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        showlegend=False,
        marker=dict(color='black')
    )

    diagonal_cov = go.Scattergl(
        x=[np.min(cov_observed +cov_sampled) , np.max(cov_observed +cov_sampled)],
        y=[np.min(cov_observed +cov_sampled) , np.max(cov_observed +cov_sampled)],
        mode="lines",
        showlegend=False,
        marker=dict(color='black')
    )

    ## define subplots
    fig = tools.make_subplots(rows=1, cols=3, print_grid=False)

    ## add traces as subplots
    fig.append_trace(trace_single_frequencies, 1, 1)
    fig.append_trace(diagonal, 1, 1)
    fig.append_trace(trace_pairwise_frequencies, 1, 2)
    fig.append_trace(diagonal, 1, 2)
    fig.append_trace(trace_cov, 1, 3)
    fig.append_trace(diagonal_cov, 1, 3)

    fig['layout'].update(
        font = dict(size=18),
        hovermode = 'closest',
        title = title,
        width=1500,
        height=500
    )

    fig['layout']['yaxis'].update(
            title="single model frequencies",
            exponentformat="e",
            showexponent='All',
            range=[0, 1],
            domain=[0, 1]
    )
    fig['layout']['yaxis2'].update(
            title="pairwise model frequencies",
            exponentformat="e",
            showexponent='All',
            range=[0, 1],
            domain=[0, 1]
    )
    fig['layout']['yaxis3'].update(
            title="model covariances",
            exponentformat="e",
            showexponent='All',
            domain=[0, 1]
    )

    fig['layout']['xaxis'].update(
            title="single observed frequencies",
            exponentformat="e",
            showexponent='All',
            range=[0, 1],
            domain=[0, 0.3],
            anchor='y1'
    )
    fig['layout']['xaxis2'].update(
            title="pairwise observed frequencies",
            exponentformat="e",
            showexponent='All',
            range=[0, 1],
            domain=[0.35, 0.65],
            anchor='y2'
    )
    fig['layout']['xaxis3'].update(
            title="observed covariances",
            exponentformat="e",
            showexponent='All',
            domain=[0.7, 1.0],
            anchor='y3'
    )


    #Add text to plot: Pearson correlation coefficient
    fig['layout']['annotations']=[
        dict(
            x=0.4,
            y=0.9,
            xref='x',
            yref='y',
            text='Pearson corr coeff = ' + str(np.round(pearson_corr_single, decimals=3)),
            showarrow=False
        ),
        dict(
            x=0.4,
            y=0.9,
            xref='x2',
            yref='y2',
            text='Pearson corr coeff = ' + str(np.round(pearson_corr_pair, decimals=3)),
            showarrow=False
        ),
        dict(
            x=np.min(cov_observed) + 0.4 * (np.max(cov_observed) - np.min(cov_observed)),
            y=0.9,
            xref='x3',
            yref='y2',
            text='Pearson corr coeff = ' + str(np.round(pearson_corr_cov, decimals=3)),
            showarrow=False
        )
    ]

    if log:
        fig['layout']['xaxis']['type']='log'
        fig['layout']['yaxis']['type']='log'


    plotly_plot(fig, filename=plot_out, auto_open=False, link_text='')


def plot_alignment(aa_counts_single, title, plot_file, freq=True):

    Neff = np.sum(aa_counts_single[0,:])
    L = aa_counts_single.shape[0]

    #create plot
    data = []

    if freq:
        aa_counts_single /= Neff

    #add bar for each amino acid for each position
    for aa in range(20):
        data.append(
            go.Bar(
                x= list(range(1,L+1)),
                y=aa_counts_single[:, aa].tolist(),
                showlegend=True,
                name=io.AMINO_ACIDS[aa]
              )
        )


    layout = go.Layout(
        barmode='stack',
        title=title,
        xaxis=dict(title="Alignment Position"),
        yaxis=dict(
            title="Amino Acid Distribution",
            exponentformat='e',
            showexponent='All'),
        font=dict(size=18)
    )

    plot = {'data': data, 'layout': layout}


    plotly_plot(plot, filename=plot_file, auto_open=False, link_text='')
