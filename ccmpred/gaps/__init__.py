import numpy as np
import ccmpred.counts

from ccmpred.gaps.cext import remove_gaps_probs, remove_gaps_consensus


def remove_gaps_col_freqs(msa):
    counts = ccmpred.counts.single_counts(msa)
    counts[:, 20] = 0

    counts /= np.sum(counts, axis=1)[:, np.newaxis]

    return remove_gaps_probs(msa, counts)


def backinsert_gapped_positions(res, gapped_positions):

    res.ncol = res.ncol + len(gapped_positions)

    for position in gapped_positions:
        res.x_single = np.insert(res.x_single,position, 0, axis=0)
        res.x_pair = np.insert(res.x_pair,position, 0, axis=0)
        res.x_pair = np.insert(res.x_pair,position, 0, axis=1)

    return res


def remove_gapped_positions(msa, max_gap_percentage):

    if max_gap_percentage >= 100:
        return msa, []

    msa_gap_counts = (msa == 20).sum(0)
    max_gap_count = (max_gap_percentage/100.0 * msa.shape[0])

    ungapped_positions  = np.where(msa_gap_counts <  max_gap_count)
    gapped_positions    = np.where(msa_gap_counts >=  max_gap_count)

    print("Removed " + str(len(gapped_positions[0])) + " alignment positions with > " +
          str(max_gap_percentage/100.0) + " percent gaps. ")

    return np.ascontiguousarray(msa[:, ungapped_positions[0]]), gapped_positions[0]