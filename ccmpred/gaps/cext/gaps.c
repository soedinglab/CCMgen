#include <stdlib.h>

#include "gaps.h"

int pick_random_weighted(flt *probs, int n) {
	int a;
	double p = (double)rand() / (double)RAND_MAX;
	for (a = 0; a < n; a++) {
		flt p_curr = probs[a];
		if (p < p_curr) {
			return a;
		}
		p -= p_curr;
	}
	return n - 1;
}


/**
 * substitute gaps in the sequence according to probability
 *
 * @param[in] p The MSA probabilities
 * @param[inout] msa The MSA to clean
 * @param[in] nrow The number of rows
 * @param[in] ncol The number of columns
 */
void remove_gaps_probs(
	const flt *const p,
	unsigned char *const msa,
	int nrow,
	int ncol
) {
	int i, j;
	for(i = 0; i < nrow; i++) {
		for (j = 0; j < ncol; j++) {
			if (msa[i * ncol + j] != GAP) continue;

			msa[i * ncol + j] = pick_random_weighted((flt *)&p[j * N_ALPHA], N_ALPHA);
		}
	}
}

/**
 * remove gaps according to consensus sequence
 *
 * @param[inout] msa the MSA to clean (nrow x ncol)
 * @param[in] The consensus sequence to use as a replacement (ncol)
 * @param[in] nrow The number of rows
 * @param[in] ncol The number of columns
 */
void remove_gaps_consensus(
	unsigned char *const msa,
	unsigned char *const consensus,
	int nrow,
	int ncol
) {
	int i, j;
	for(i = 0; i < nrow; i++) {
		for(j = 0; j < ncol; j++) {
			if(msa[i * ncol + j] != GAP) continue;
			msa[i * ncol + j] = consensus[j];
		}
	}
}
