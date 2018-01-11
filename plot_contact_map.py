#!/usr/bin/env python

# ===============================================================================
###     Plot a contact map
###
###     when pdb file is specified, observed distances will be in upper left
###     and contact map will be in lower right
# ===============================================================================

import argparse
import numpy as np
import pandas as pd
import ccmpred.raw as raw
from ccmpred.weighting import SequenceWeights
from ccmpred.pseudocounts import PseudoCounts
import ccmpred.io.contactmatrix as io
from ccmpred.io.alignment import read_msa
import ccmpred.io.pdb as pdb
import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot
from plotly import tools
import scipy.stats
import colorlover as cl


def plot_percentage_gaps_per_position(alignment, plot_file=None):

    N = float(len(alignment))
    L = len(alignment[0])

    weighting = SequenceWeights(False, 0.8)
    weights = weighting.weights_simple(alignment)

    #compute counts and frequencies
    pseudocounts = PseudoCounts(alignment, weights)
    pseudocounts.calculate_frequencies(
        'uniform_pseudocounts', 1, 1, remove_gaps=False
    )

    #compute percentage of gaps
    gaps = pseudocounts.counts[0][:, 20] / pseudocounts.counts[0].sum(1)

    #normalized entropy
    entropy_per_position = scipy.stats.entropy(pseudocounts.counts[0].transpose(),base=2)
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
        'title':"Percentage of gaps and Entropy in alignment <br> N="+str(N) + ", L="+str(L),
        'xaxis':{'title':"Alignment Position"},
        'yaxis':{'title':"Percentage of Gaps/Entropy"},
        'font':{'size':18}
    }

    plot = {'data': data, 'layout': layout}
    if plot_file is None:
        return plot
    else:
        plotly_plot(plot, filename=plot_file, auto_open=False)

def plot_contact_map(mat, seqsep, contact_threshold, plot_file, title, alignment=None, pdb_file=None):
    L = len(mat)
    indices_upper_tri = np.triu_indices(L, seqsep)

    ### if alignment file is specified, compute Ni
    gaps_percentage_plot = None
    if alignment is not None:
        gaps_percentage_plot = plot_percentage_gaps_per_position(alignment, plot_file=None)

    plot_matrix = pd.DataFrame()

    ###compute distance map from pdb file
    if (pdb_file):
        pdb_file = pdb_file
        observed_distances = pdb.distance_map(pdb_file, L)
        plot_matrix['distance'] = observed_distances[indices_upper_tri]
        plot_matrix['contact'] = ((plot_matrix.distance < contact_threshold) * 1).tolist()

    # add scores
    plot_matrix['residue_i'] = indices_upper_tri[0] + 1
    plot_matrix['residue_j'] = indices_upper_tri[1] + 1
    plot_matrix['confidence'] = mat[indices_upper_tri]

    ### Plot Contact Map
    plot_contact_map_someScore_plotly(plot_matrix, title, seqsep, gaps_percentage_plot, plot_file)

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

    # add diagonal and diagonals marking sequence separation
    data.append(go.Scatter(x=[0, L], y=[0, L], mode='lines', line=dict(color=('rgb(0, 0, 0)'), width=4), hoverinfo=None,
                           showlegend=False))
    data.append(
        go.Scatter(x=[0, L - seqsep + 1], y=[seqsep - 1, L], mode='lines', line=dict(color=('rgb(0, 0, 0)'), width=2),
                   showlegend=False))
    data.append(
        go.Scatter(x=[seqsep - 1, L], y=[0, L - seqsep + 1], mode='lines', line=dict(color=('rgb(0, 0, 0)'), width=2),
                   showlegend=False))

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

    fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, print_grid=False)

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

    if gaps_percentage_plot is not None:
        for trace in gaps_percentage_plot['data']:
            fig.append_trace(trace, 1, 1)

        # contact map domain 0-0.9
        fig['layout']['yaxis2']['domain'] = [0.0, 0.9]

        # xaxis range only to 0.9 so that contact map is square
        fig['layout']['xaxis1']['domain'] = [0.0, 0.9]

        # line plot domain 0.9-1.0
        fig['layout']['yaxis1']['title'] = 'Percentage of Gaps'
        fig['layout']['yaxis1']['domain'] = [0.9, 1.0]

    if plot_file:
        plotly_plot(fig, filename=plot_file, auto_open=False)
    else:
        return fig

def main():

    parser = argparse.ArgumentParser(description='Plotting a contact map.')

    group_append = parser.add_mutually_exclusive_group(required=True)
    group_append.add_argument('-m', '--mat-file', dest='mat_file', type=str, help='path to mat file')
    group_append.add_argument('-b', '--braw-file', dest='braw_file', type=str,help='path to binary raw coupling file')

    parser.add_argument('-o', '--plot-out', dest='plot_out', type=str, help='Output directory for plot')

    parser.add_argument('--seq-sep', dest='seqsep', type=int, default=6, help='Minimal sequence separation')
    parser.add_argument('--contact-threshold', dest='contact_threshold', type=int, default=8,  help='Contact definition as maximal C_beta distance between residue pairs.')
    parser.add_argument('--pdb-file', dest='pdb_file', type=str, help='Optional PDB file (renumbered starting from 1) for distance matrix.')
    parser.add_argument('--alignment-file', dest='alignment_file', type=str, help='Optional alignment file for gap percentage and entropy subplot.')
    parser.add_argument("--aln-format", dest="aln_format", default="psicov", help="File format for MSAs [default: \"%(default)s\"]")
    parser.add_argument("--apc", action="store_true", default=False, help="Apply average product correction")
    parser.add_argument("--entropy-correction", dest='entropy_correction', action="store_true", default=False, help="Apply entropy correction")

    args = parser.parse_args()


    if args.mat_file is None and args.braw_file is None:
        print("Either mat_file or braw_file need to be set.")

    mat_file    = args.mat_file
    braw_file   = args.braw_file
    alignment_file = args.alignment_file
    aln_format = args.aln_format
    pdb_file    = args.pdb_file
    plot_out    = args.plot_out
    seqsep      = args.seqsep
    contact_threshold = args.contact_threshold

    apc = args.apc
    entropy_correction = args.entropy_correction

    alignment=None
    if alignment_file is not None:
        alignment = read_msa(alignment_file, aln_format)

        #compute sequence weights
        weighting = SequenceWeights(False, 0.8)
        weights = weighting.weights_simple(alignment)

        #compute frequencies
        pseudocounts = PseudoCounts(alignment, weights)
        pseudocounts.calculate_frequencies(
            'uniform_pseudocounts', 1, 1, remove_gaps=False
        )

    if braw_file is not None:

        braw = raw.parse_msgpack(braw_file)
        meta_info = braw.meta

        #compute frobenius score from couplings
        mat = io.frobenius_score(braw.x_pair)

        if entropy_correction:
            if alignment is None:
                print("Alignment file is necessary to compute entropy correction!")
            else:
                scaling_factor_eta, mat = io.compute_local_correction(
                    pseudocounts.freqs[0],
                    braw.x_pair,
                    meta_info['workflow'][0]['msafile']['neff'],
                    meta_info['workflow'][0]['regularization']['lambda_pair'],
                    mat,
                    squared=False, entropy=True
                )
        elif apc:
            mat = io.apc(mat)

    if mat_file is not None:
        mat, meta_info = io.read_matrix(mat_file)

        if entropy_correction:
            print("Binary Raw file is necessary to compute entropy correction!")
        elif apc:
            mat = io.apc(mat)


    plot_file = plot_out + "/contact_map_seqsep{0}_contacthr{1}.html".format(seqsep, contact_threshold)
    plot_contact_map(mat, seqsep, contact_threshold, plot_file, "", alignment=alignment, pdb_file=pdb_file)





if __name__ == '__main__':
    main()