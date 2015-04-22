#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "cd.h"
#include "cdutil.h"


/**
 * Compute conditional probabilities
 * $P(X_i = a |  X^n_0, ... X^n_L \setminus X^n_i)$
 *
 * @param[in] i Index of the column to compute probabilities for
 * @param[out] cond_probs Returns a 20-field array of conditional probabilities
 * @param[in] x The current potentials
 * @param[in] last_seq The current sequence to condition on
 * @param[in] ncol The number of columns in the MSA
 */
void compute_conditional_probs(
	const int i,
	flt *const cond_probs,
	const flt *const x,
	const unsigned char *const last_seq,
	const int ncol
) {
	int a, j;
	int nsingle = ncol * (N_ALPHA - 1);

	for (a = 0; a < N_ALPHA - 1; a++) {
		cond_probs[a] = E1(i,a);
	}

	for (a = 0; a < N_ALPHA - 1; a++) {
		for (j = 0; j < ncol; j++) {
			cond_probs[a] += E2(i, a, j, last_seq[j]);
		}

		// don't add up the case i = j
		cond_probs[a] -= E2(i, a, i, last_seq[i]);
	}

	cond_probs[GAP] = F0;

	flt denom = F0;
	for (a = 0; a < N_ALPHA - 1; a++) {
		cond_probs[a] = fexp(cond_probs[a]);
		denom += cond_probs[a];
	}

	for (a = 0; a < N_ALPHA - 1; a++) {
		cond_probs[a] /= denom;
	}
}

/**
 * Resample a multiple sequence alignment
 * 
 * @param[inout] seq The MSA to work on
 * @param[in] x The current potentials
 * @param[in] ncol The number of columns in the MSA
 * @param[in] n_samples The number of samples to generate (also the number of rows in the MSA)
 */
void sample_sequences(
	unsigned char *seq,
	const flt *const x,
	const int n_samples,
	const int ncol
) {
	#pragma omp parallel
	{
		int k, i;
		flt *pcondcurr = fl_malloc(N_ALPHA);

		#pragma omp for
		for (k = 0; k < n_samples; k++) {
			i = pick_random_uniform(ncol - 1);

			compute_conditional_probs(i, pcondcurr, x, &seq[k * ncol], ncol);
			seq[k * ncol + i] = pick_random_weighted(pcondcurr, N_ALPHA - 1);
		}
		fl_free(pcondcurr);
	}
}

