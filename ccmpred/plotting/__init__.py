import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot
from plotly import tools
import ccmpred.io as io
import numpy as np
import scipy.stats
import colorlover as cl


def plot_percentage_gaps_per_position(single_freq, plot_file=None):

    L = single_freq.shape[0]

    #compute percentage of gaps
    gaps = single_freq[:, 20] / single_freq.sum(1)

    #normalized entropy
    entropy_per_position = scipy.stats.entropy(single_freq.transpose(),base=2)
    entropy_per_position /= np.max(entropy_per_position)


    #create plot
    data = []
    data.append(
        go.Scatter(
            x=[x for x in range(1,L+1)],
            y=gaps,
            name = "percentage of gaps",
            mode="Lines",
            line=dict(width=3)
        )
    )

    data.append(
        go.Scatter(
            x=[x for x in range(1,L+1)],
            y=entropy_per_position,
            name = "relative Entropy",
            mode="Lines",
            line=dict(width=3)
        )
    )

    layout = {
        'title':"Percentage of gaps and Entropy in alignment",
        'xaxis':{'title':"Alignment Position"},
        'yaxis':{'title':"Percentage of Gaps/Entropy"},
        'font':{'size':18}
    }

    plot = {'data': data, 'layout': layout}
    if plot_file is None:
        return plot
    else:
        plotly_plot(plot, filename=plot_file, auto_open=False)

def plot_contact_map_someScore_plotly(plot_matrix, title, seqsep, gaps_percentage_plot=None, plot_file=None):

    # sort matrix by confidence score
    plot_matrix.sort_values(by='confidence', ascending=False, inplace=True)
    L = np.max(plot_matrix[['residue_i', 'residue_j']].values)

    data = []

    # add predicted contact map
    data.append(
        go.Heatmap(
            x=plot_matrix.residue_i.tolist(),
            y=plot_matrix.residue_j.tolist(),
            z=plot_matrix.confidence.tolist(),
            name='predicted',
            colorscale='Greys', reversescale=True,
            colorbar=go.ColorBar(
                x=1.02,
                y=0.4,
                yanchor='bottom',
                len=0.4,
                title="Score"
            )
        )
    )
    data.append(
        go.Heatmap(
            x=plot_matrix.residue_j.tolist(),
            y=plot_matrix.residue_i.tolist(),
            z=plot_matrix.confidence.tolist(),
            name='predicted'
        )
    )


    # if distances and class are available
    if 'contact' in plot_matrix and 'distance' in plot_matrix:

        # define true and false positives among the L/5 highest scores
        sub_L5_true = plot_matrix.query('distance > 0').head(int(L / 5)).query('contact > 0')
        sub_L5_false = plot_matrix.query('distance > 0').head(int(L / 5)).query('contact < 1')

        if len(sub_L5_true) > 0:
            # Mark TP and FP in the plot with little crosses
            tp = go.Scatter(
                x=sub_L5_true['residue_i'].tolist() + sub_L5_true['residue_j'].tolist(),
                y=sub_L5_true['residue_j'].tolist() + sub_L5_true['residue_i'].tolist(),
                mode='markers',
                marker=dict(
                    symbol=134,
                    color="green",
                    line=dict(width=2),
                    size=12
                ),  # size_tp, sizeref=np.max([size_tp + size_fp])/15, sizemode = 'diameter'),
                name="TP (L/5)",
                hoverinfo="none"
            )

        # 'rgb(255,247,188)', 'rgb(254,196,79)'
        green_yello_red = ['rgb(254,196,79)', 'rgb(222,45,38)']
        max_tp = 8
        max_fp = np.max(plot_matrix[plot_matrix.contact < 1]['distance'])
        fp_distance_range = int(np.ceil((max_fp - max_tp) / 10.0) * 10)
        green_yello_red_interpolated = cl.interp(green_yello_red, fp_distance_range)
        data_color = [green_yello_red_interpolated[int(x - max_tp)] for x in sub_L5_false['distance']]

        if len(sub_L5_false) > 0:
            fp = go.Scatter(
                x=sub_L5_false['residue_i'].tolist() + sub_L5_false['residue_j'].tolist(),
                y=sub_L5_false['residue_j'].tolist() + sub_L5_false['residue_i'].tolist(),
                mode='markers',
                marker=dict(
                    symbol=134,
                    # color="red",
                    color=data_color * 2,
                    colorscale=green_yello_red_interpolated,
                    line=dict(width=2),
                    size=12
                ),  # size_fp, sizeref=np.max([size_tp + size_fp])/15, sizemode = 'diameter'),
                name="FP (L/5)",
                hoverinfo="none"
            )

        # colorscale from red (small distance) to blue(large distance)
        zmax = np.max(plot_matrix.distance)
        percent_at_contact_thr = 8 / zmax
        distance_colorscale = [[0, 'rgb(128, 0, 0)'], [percent_at_contact_thr, 'rgb(255, 255, 255)'],
                               [1, 'rgb(22, 96, 167)']]

        # define triangle on opposite site of Predictions
        heatmap_observed = go.Heatmap(
            x=plot_matrix.residue_j.tolist(),
            y=plot_matrix.residue_i.tolist(),
            z=plot_matrix.distance.tolist(),
            name='observed',
            zmin=0,
            zmax=zmax,
            colorscale=distance_colorscale,
            # colorscale='Greys', reversescale=True,
            # colorscale=distance_colors_interpol, reversescale=True,
            colorbar=go.ColorBar(
                x=1.02,
                y=0,
                yanchor='bottom',
                len=0.4,
                title="Distance [A]")
        )

        # put all plot elements in data list
        data[1] = heatmap_observed

        if len(sub_L5_true) > 0:
            data.append(tp)
        if len(sub_L5_false) > 0:
            data.append(fp)


    # add diagonal and diagonals marking sequence separation
    data.append(go.Scatter(x=[0, L], y=[0, L], mode='lines', line=dict(color=('rgb(0, 0, 0)'), width=4), hoverinfo=None,
                           showlegend=False))
    data.append(
        go.Scatter(x=[0, L - seqsep + 1], y=[seqsep - 1, L], mode='lines', line=dict(color=('rgb(0, 0, 0)'), width=2),
                   showlegend=False))
    data.append(
        go.Scatter(x=[seqsep - 1, L], y=[0, L - seqsep + 1], mode='lines', line=dict(color=('rgb(0, 0, 0)'), width=2),
                   showlegend=False))


    fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, print_grid=False)


    if gaps_percentage_plot is not None:
        for trace in gaps_percentage_plot['data']:
            fig.append_trace(trace, 1, 1)


    for trace in data:
        fig.append_trace(trace, 2, 1)

    fig['layout']['title'] = title
    fig['layout']['legend'] = {'x': 1.02, 'y': 1}  # places legend to the right of plot

    fig['layout']['xaxis1']['title'] = 'j'
    fig['layout']['xaxis1']['range'] = [0.5, L + 0.5]
    fig['layout']['xaxis1']['domain'] = [0.0, 1.0]
    fig['layout']['xaxis1']['zeroline'] = False

    fig['layout']['yaxis2']['title'] = 'i'
    fig['layout']['yaxis2']['range'] = [0.5, L + 0.5]
    fig['layout']['yaxis2']['domain'] = [0.0, 1.0]
    fig['layout']['yaxis2']['scaleanchor'] = "x"
    fig['layout']['yaxis2']['scaleratio'] = 1.0
    fig['layout']['yaxis2']['zeroline'] = False

    fig['layout']['font']['size'] = 18

    #percentage gaps and entropy plot
    if gaps_percentage_plot is not None:
        fig['layout']['yaxis2']['domain'] = [0.0, 0.9]
        fig['layout']['xaxis1']['domain'] = [0.0, 0.9]
        fig['layout']['yaxis1']['domain'] = [0.9, 1.0]



    if plot_file:
        plotly_plot(fig, filename=plot_file, auto_open=False)
    else:
        return fig

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
