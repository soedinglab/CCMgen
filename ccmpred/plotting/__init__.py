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
            name = "fraction gaps",
            mode="Lines",
            line=dict(width=3)
        )
    )

    data.append(
        go.Scatter(
            x=[x for x in range(1,L+1)],
            y=entropy_per_position,
            name = "entropy",
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


    hover_text = ["residue i: {0}<br>residue j: {1}<br>score: {2}".format(
                plot_matrix.residue_i.tolist()[i],
                plot_matrix.residue_j.tolist()[i],
                np.round(plot_matrix.confidence.tolist()[i], decimals=3))
                for i in range(len(plot_matrix.residue_i.tolist()))]

    hover_text  += ["residue i: {0}<br>residue j: {1}<br>score: {2}".format(
                plot_matrix.residue_j.tolist()[i],
                plot_matrix.residue_i.tolist()[i],
                np.round(plot_matrix.confidence.tolist()[i], decimals=3))
                for i in range(len(plot_matrix.residue_i.tolist()))]

    # add predicted contact map
    data.append(
        go.Heatmap(
            x=plot_matrix.residue_i.tolist() + plot_matrix.residue_j.tolist(),
            y=plot_matrix.residue_j.tolist() + plot_matrix.residue_i.tolist(),
            z=plot_matrix.confidence.tolist() + plot_matrix.confidence.tolist(),
            name='predicted',
            hoverinfo="text",
            text=hover_text,
            colorscale='Greys',
            reversescale=True,
            colorbar=go.ColorBar(
                x=1,
                y=0.4,
                yanchor='bottom',
                len=0.4,
                title="Score"
            )
        )
    )


    # if distances and class are available
    if 'contact' in plot_matrix and 'distance' in plot_matrix:

        # colorscale from red (small distance) to blue(large distance)
        zmax = np.max(plot_matrix.distance)
        percent_at_contact_thr = 8 / zmax
        distance_colorscale = [[0, 'rgb(128, 0, 0)'], [percent_at_contact_thr, 'rgb(255, 255, 255)'],
                               [1, 'rgb(22, 96, 167)']]

        hover_text = ["residue i: {0}<br>residue j: {1}<br>score: {2}<br>distance: {3}".format(
                plot_matrix.residue_j.tolist()[i],
                plot_matrix.residue_i.tolist()[i],
                np.round(plot_matrix.confidence.tolist()[i], decimals=3),
                np.round(plot_matrix.distance.tolist()[i], decimals=3))
                for i in range(len(plot_matrix.residue_i.tolist()))]


        hover_text += ["residue i: {0}<br>residue j: {1}<br>score: {2}<br>distance: {3}".format(
                plot_matrix.residue_i.tolist()[i],
                plot_matrix.residue_j.tolist()[i],
                np.round(plot_matrix.confidence.tolist()[i], decimals=3),
                np.round(plot_matrix.distance.tolist()[i], decimals=3))
                for i in range(len(plot_matrix.residue_i.tolist()))]


        # define triangle on opposite site of Predictions
        data.append(
            go.Heatmap(
                x=plot_matrix.residue_j.tolist(),
                y=plot_matrix.residue_i.tolist(),
                z=plot_matrix.distance.tolist(),
                name='observed',
                hoverinfo="text",
                text=hover_text,
                zmin=0,
                zmax=zmax,
                colorscale=distance_colorscale,
                colorbar=go.ColorBar(
                    x=1,
                    y=0,
                    yanchor='bottom',
                    len=0.4,
                    title="Distance [A]")
            )
        )


        # define true and false positives among the L/5 highest scores
        sub_L5_true = plot_matrix.query('distance > 0').head(int(L / 5)).query('contact > 0')
        sub_L5_false = plot_matrix.query('distance > 0').head(int(L / 5)).query('contact < 1')

        tp_text = ["residue i: {0}<br>residue j: {1}<br>score: {2}<br>distance: {3}".format(
                sub_L5_true.residue_i.tolist()[i],
                sub_L5_true.residue_j.tolist()[i],
                np.round(plot_matrix.confidence.tolist()[i], decimals=3),
                np.round(plot_matrix.distance.tolist()[i], decimals=3))
                for i in range(len(sub_L5_true.residue_i.tolist()))]

        tp_text += ["residue i: {0}<br>residue j: {1}<br>score: {2}<br>distance: {3}".format(
                sub_L5_true.residue_j.tolist()[i],
                sub_L5_true.residue_i.tolist()[i],
                np.round(plot_matrix.confidence.tolist()[i], decimals=3),
                np.round(plot_matrix.distance.tolist()[i], decimals=3))
                for i in range(len(sub_L5_true.residue_i.tolist()))]

        if len(sub_L5_true) > 0:
            # Mark TP and FP in the plot with little crosses
            data.append(
                go.Scatter(
                    x=sub_L5_true['residue_i'].tolist() + sub_L5_true['residue_j'].tolist(),
                    y=sub_L5_true['residue_j'].tolist() + sub_L5_true['residue_i'].tolist(),
                    mode='markers',
                    text=tp_text,
                    hoverinfo="text",
                    marker=dict(
                        symbol=134,
                        color="green",
                        line=dict(width=2),
                        size=12
                    ),
                    name="TP (L/5)"
                )
            )

        # 'rgb(255,247,188)', 'rgb(254,196,79)'
        green_yello_red = ['rgb(254,196,79)', 'rgb(222,45,38)']
        max_tp = 8
        max_fp = np.max(plot_matrix[plot_matrix.contact < 1]['distance'])
        fp_distance_range = int(np.ceil((max_fp - max_tp) / 10.0) * 10)
        green_yello_red_interpolated = cl.interp(green_yello_red, fp_distance_range)
        data_color = [green_yello_red_interpolated[int(x - max_tp)] for x in sub_L5_false['distance']]

        fp_text = ["residue i: {0}<br>residue j: {1}<br>score: {2}<br>distance: {3}".format(
                sub_L5_false.residue_i.tolist()[i],
                sub_L5_false.residue_j.tolist()[i],
                np.round(plot_matrix.confidence.tolist()[i], decimals=3),
                np.round(plot_matrix.distance.tolist()[i], decimals=3))
                for i in range(len(sub_L5_false.residue_i.tolist()))]

        fp_text += ["residue i: {0}<br>residue j: {1}<br>score: {2}<br>distance: {3}".format(
                sub_L5_false.residue_j.tolist()[i],
                sub_L5_false.residue_i.tolist()[i],
                np.round(plot_matrix.confidence.tolist()[i], decimals=3),
                np.round(plot_matrix.distance.tolist()[i], decimals=3))
                for i in range(len(sub_L5_false.residue_i.tolist()))]


        if len(sub_L5_false) > 0:
            data.append(
                go.Scatter(
                    x=sub_L5_false['residue_i'].tolist() + sub_L5_false['residue_j'].tolist(),
                    y=sub_L5_false['residue_j'].tolist() + sub_L5_false['residue_i'].tolist(),
                    mode='markers',
                    text=fp_text,
                    hoverinfo="text",
                    marker=dict(
                        symbol=134,
                        color=data_color * 2,
                        colorscale=green_yello_red_interpolated,
                        line=dict(width=2),
                        size=12
                    ),
                    name="FP (L/5)"

                )
            )




    # add diagonal and diagonals marking sequence separation
    data.append(go.Scatter(
        x=[0, L], y=[0, L],
        mode='lines',
        line=dict(color=('rgb(0, 0, 0)'), width=4),
        hoverinfo=None,
        showlegend=False)
    )
    data.append(
        go.Scatter(
            x=[0, L - seqsep + 1], y=[seqsep - 1, L],
            mode='lines',
            line=dict(color=('rgb(0, 0, 0)'), width=2),
            showlegend=False)
    )
    data.append(
        go.Scatter(
            x=[seqsep - 1, L], y=[0, L - seqsep + 1],
            mode='lines',
            line=dict(color=('rgb(0, 0, 0)'), width=2),
            showlegend=False)
    )


    fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, print_grid=False)


    if gaps_percentage_plot is not None:
        for trace in gaps_percentage_plot['data']:
            fig.append_trace(trace, 1, 1)

    for trace in data:
        fig.append_trace(trace, 2, 1)

    fig['layout']['title'] = title
    fig['layout']['width'] = 1000
    fig['layout']['height'] = 1000
    fig['layout']['legend'] = {'x': 1, 'y': 1}  # places legend to the right of plot
    fig['layout']['hovermode'] = "closest"

    fig['layout']['xaxis1']['title'] = 'i'
    fig['layout']['xaxis1']['range'] = [0.5, L + 0.5]
    fig['layout']['xaxis1']['domain'] = [0.0, 1.0]
    fig['layout']['xaxis1']['zeroline'] = False

    fig['layout']['yaxis2']['title'] = 'j'
    fig['layout']['yaxis2']['range'] = [0.5, L + 0.5]
    fig['layout']['yaxis2']['domain'] = [0.0, 1.0]
    fig['layout']['yaxis2']['scaleanchor'] = "x"
    fig['layout']['yaxis2']['scaleratio'] = 1.0
    fig['layout']['yaxis2']['zeroline'] = False

    fig['layout']['font']['size'] = 18

    #percentage gaps and entropy plot
    if gaps_percentage_plot is not None:
        fig['layout']['yaxis2']['domain'] = [0.0, 0.9]
        #fig['layout']['xaxis1']['domain'] = [0.0, 0.9]
        fig['layout']['yaxis1']['domain'] = [0.9, 1.0]



    if plot_file:
        plotly_plot(fig, filename=plot_file, auto_open=False)
    else:
        return fig

def plot_empirical_vs_model_statistics(
        single_freq_observed, single_freq_sampled,
        pairwise_freq_observed, pairwise_freq_sampled,
        plot_out):

    L = single_freq_observed.shape[0]
    indices_upper_triangle_i, indices_upper_triangle_j = np.triu_indices(L, k=1)

    x_single = single_freq_observed.flatten().tolist()
    y_single = single_freq_sampled.flatten().tolist()
    pair_freq_observed = pairwise_freq_observed[
                         indices_upper_triangle_i,
                         indices_upper_triangle_j, :, :].flatten().tolist()
    pair_freq_sampled = pairwise_freq_sampled[
                        indices_upper_triangle_i,
                        indices_upper_triangle_j, :, :].flatten().tolist()
    cov_observed = [pairwise_freq_observed[i, j, a, b] - (single_freq_observed[i, a] * single_freq_observed[j, b])
                    for i in range(L - 1) for j in range(i + 1, L) for a in range(20) for b in range(20)]
    cov_sampled = [pairwise_freq_sampled[i, j, a, b] - (single_freq_sampled[i, a] * single_freq_sampled[j, b])
                   for i in range(L - 1) for j in range(i + 1, L) for a in range(20) for b in range(20)]


    ## first trace: single amino acid frequencies
    trace_single_frequencies = go.Scattergl(
        x=x_single,
        y=y_single,
        mode='markers',
        name='single frequencies',
        text=["position: {0}<br>amino acid: {1}".format(i+1,io.AMINO_ACIDS[a]) for i in range(L) for a in range(20)],
        marker=dict(color='black'),
        opacity=0.1,
        showlegend=False
    )
    pearson_corr_single = np.corrcoef(x_single, y_single)[0,1]

    ## second trace: pairwise amino acid frequencies
    parir_freq_annotation = ["position: {0}-{1}<br>amino acid: {2}-{3}".format(
        i+1,
        j+1,
        io.AMINO_ACIDS[a],
        io.AMINO_ACIDS[b]) for i in range(L-1) for j in range(i+1, L) for a in range(20) for b in range(20)]
    trace_pairwise_frequencies = go.Scattergl(
        x=pair_freq_observed,
        y=pair_freq_sampled,
        mode='markers',
        name='pairwise frequencies',
        text=parir_freq_annotation,
        marker=dict(color='black'),
        opacity=0.1,
        showlegend=False
    )
    pearson_corr_pair = np.corrcoef(pair_freq_observed, pair_freq_sampled)[0, 1]

    ## third trace: covariances
    trace_cov = go.Scattergl(
        x=cov_observed,
        y=cov_sampled,
        mode='markers',
        name='covariances',
        text=parir_freq_annotation,
        marker=dict(color='black'),
        opacity=0.1,
        showlegend=False
    )
    pearson_corr_cov = np.corrcoef(cov_observed, cov_sampled)[0, 1]


    #define diagonals
    diag_single = [np.min(x_single  + y_single), np.max(x_single  + y_single)]
    diag_pair = [np.min(pair_freq_observed + pair_freq_sampled), np.max(pair_freq_observed  + pair_freq_sampled)]
    diag_cov = [np.min(cov_observed + cov_sampled), np.max(cov_observed+ cov_sampled)]


    diagonal_single = go.Scattergl(
        x=diag_single,
        y=diag_single,
        mode="lines",
        showlegend=False,
        marker=dict(color='rgb(153, 204, 255)')
    )

    diagonal_pair = go.Scattergl(
        x=diag_pair,
        y=diag_pair,
        mode="lines",
        showlegend=False,
        marker=dict(color='rgb(153, 204, 255)')
    )

    diagonal_cov = go.Scattergl(
        x=diag_cov,
        y=diag_cov,
        mode="lines",
        showlegend=False,
        marker=dict(color='rgb(153, 204, 255)')
    )



    ## define subplots
    fig = tools.make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["single site amino acid frequencies", "pairwise amino acid frequencies", "covariances"],
        horizontal_spacing = 0.05,
        print_grid=False
    )

    ## add traces as subplots
    fig.append_trace(trace_single_frequencies, 1, 1)
    fig.append_trace(diagonal_single, 1, 1)
    fig.append_trace(trace_pairwise_frequencies, 1, 2)
    fig.append_trace(diagonal_pair, 1, 2)
    fig.append_trace(trace_cov, 1, 3)
    fig.append_trace(diagonal_cov, 1, 3)

    #incresae size of subplot titles
    fig['layout']['annotations'][0]['font']['size'] = 20
    fig['layout']['annotations'][1]['font']['size'] = 20
    fig['layout']['annotations'][2]['font']['size'] = 20

    # # add text to plot: Pearson correlation coefficient
    fig['layout']['annotations'].extend(
        [
            dict(
                x=0.13,#0.02,
                y=0.04,#0.95,
                xanchor="left",
                xref='paper',
                yref='paper',
                text='Pearson r = ' + str(np.round(pearson_corr_single, decimals=3)),
                bgcolor = "white",
                showarrow=False
            ),
            dict(
                x=0.48,#0.37,
                y=0.04,#0.95,
                xanchor="left",
                xref='paper',
                yref='paper',
                text='Pearson r = ' + str(np.round(pearson_corr_pair, decimals=3)),
                bgcolor="white",
                showarrow=False
            ),
            dict(
                x=0.85,#0.71,
                y=0.04,#0.95,
                xanchor="left",
                xref='paper',
                yref='paper',
                text='Pearson r = ' + str(np.round(pearson_corr_cov, decimals=3)),
                bgcolor="white",
                showarrow=False
            )
        ]
    )



    #define layout
    fig['layout'].update(
        font = dict(size=20),
        hovermode = 'closest',
        width=1500,
        height=500,
        margin=dict(t=40)

    )


    #specify axis layout details
    fig['layout']['yaxis1'].update(
            title="statistics from MCMC sample",
            exponentformat="e",
            showexponent='All',
            scaleanchor="x1",
            scaleratio=1
    )
    fig['layout']['yaxis2'].update(
            exponentformat="e",
            showexponent='All',
            scaleanchor="x2",
            scaleratio=1
    )
    fig['layout']['yaxis3'].update(
            exponentformat="e",
            showexponent='All',
            scaleanchor="x3",
            scaleratio=1
    )
    fig['layout']['xaxis1'].update(
            exponentformat="e",
            showexponent='All',
            scaleanchor="y1",
            scaleratio=1,
            showspikes=True
    )
    fig['layout']['xaxis2'].update(
            title="statistics from natural sequences",
            exponentformat="e",
            showexponent='All',
            scaleanchor="y2",
            scaleratio=1
    )
    fig['layout']['xaxis3'].update(
            exponentformat="e",
            showexponent='All',
            scaleanchor="y3",
            scaleratio=1
    )

    fig['layout']['xaxis1']['range'] = [0, 1]
    fig['layout']['xaxis2']['range'] = [0, 1]
    fig['layout']['yaxis1']['range'] = [0, 1]
    fig['layout']['yaxis2']['range'] = [0, 1]



    plotly_plot(fig, filename=plot_out, auto_open=False, link_text='', image_filename=plot_out.replace("html", ""))



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
