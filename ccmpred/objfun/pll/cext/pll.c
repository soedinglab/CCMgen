#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "pll.h"

float evaluate_pll(
	const float *x,
	float *g,
	float *g2,
	float *v_centering,
	float *weights,
	unsigned char *msa,
	const uint32_t ncol,
	const uint32_t nrow,
	float lambda_single,
	float lambda_pair
) {
	int i, j, k, s, v, a, b;
	uint32_t nsingle = ncol * (N_ALPHA - 1);
	uint32_t nsingle_padded = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD);
	uint64_t nvar_padded = nsingle_padded + ncol * ncol * N_ALPHA * N_ALPHA_PAD;

	const float *x1 = x;
	const float *x2 = &x[nsingle_padded];

	float *g1 = g;
	float *g2l = &g[nsingle_padded];

	// set fx and gradient to 0 initially
	float fx = 0.0;

	memset(g, 0, sizeof(float) * nvar_padded);
	memset(g2, 0, sizeof(float) * (nvar_padded - nsingle_padded));

	for(i = 0; i < nrow; i++) {
		float weight = weights[i];

		float precomp[N_ALPHA * ncol] __attribute__ ((aligned (32)));	// aka PC(a,s)
		float precomp_sum[ncol] __attribute__ ((aligned (32)));
		float precomp_norm[N_ALPHA * ncol] __attribute__ ((aligned (32)));	// aka PCN(a,s)

		// compute PC(a,s) = V_s(a) + sum(k \in V_s) w_{sk}(a, X^i_k)
		for(a = 0; a < N_ALPHA-1; a++) {
			for(s = 0; s < ncol; s++) {
				PC(a,s) = V(s,a);
			}
		}
		
		for(k = 0; k < ncol; k++) {
			unsigned char xik = X(i,k);

			for(a = 0; a < N_ALPHA - 1; a++) {
				for(j = 0; j < ncol; j++) {
					PC(a, j) += W(xik, k, a, j);
				}

			}
		}

		for(s = 0; s < ncol; s++) {
			PC(N_ALPHA - 1, s) = 0;
		}

		// compute precomp_sum(s) = log( sum(a=1..21) exp(PC(a,s)) )
		memset(precomp_sum, 0, sizeof(float) * ncol);
		for(a = 0; a < N_ALPHA - 1; a++) {
			for(s = 0; s < ncol; s++) {
				precomp_sum[s] += expf(PC(a,s));
			}
		}

		for(s = 0; s < ncol; s++) {
			precomp_sum[s] = logf(precomp_sum[s]);
		}

		for(a = 0; a < N_ALPHA - 1; a++) {
			for(s = 0; s < ncol; s++) {
				PCN(a,s) = expf(PC(a, s) - precomp_sum[s]);
			}
		}

		for(s = 0; s < ncol; s++) {
			PCN(N_ALPHA - 1, s) = 0.0;
		}

		// actually compute fx and gradient
		for(k = 0; k < ncol; k++) {

			unsigned char xik = X(i,k);

			fx += weight * (-PC( xik, k ) + precomp_sum[k]);

			if(xik < N_ALPHA - 1) {
				G1(k, xik) -= weight;
			} else {
				for(a = 0; a < N_ALPHA; a++) {
					PCN(a, k) = 0;
				}

			}


			for(a = 0; a < N_ALPHA - 1; a++) {
				G1(k, a) += weight * PCN(a, k);
			}

		}
		for(k = 0; k < ncol; k++) {

			unsigned char xik = X(i,k);

			for(j = 0; j < ncol; j++) {
				unsigned char xij = X(i,j);
				G2(xik, k, xij, j) -= weight;
			}

			for(a = 0; a < N_ALPHA - 1; a++) {
				for(j = 0; j < ncol; j++) {
					G2(xik, k, a, j) += weight * PCN(a, j);
				}
			}
		}

	} // i



	// add transposed onto un-transposed
	for(b = 0; b < N_ALPHA; b++) {
		for(k = 0; k < ncol; k++) {
			for(a = 0; a < N_ALPHA; a++) {
				for(j = 0; j < ncol; j++) {
					G2L(b, k, a, j) = G2(b, k, a, j) + G2(a, j, b, k);
				}
			}
		}
	}

	// set gradients to zero for self-edges
	for(b = 0; b < N_ALPHA; b++) {
		for(k = 0; k < ncol; k++) {
			for(a = 0; a < N_ALPHA; a++) {
				G2L(b, k, a, k) = 0;
			}
		}
	}

	for(k = 0; k < ncol; k++) {
		for(j = 0; j < ncol; j++) {
			for(a = 0; a < N_ALPHA; a++) {
				G2L(a, k, N_ALPHA - 1, j) = 0;
				G2L(N_ALPHA - 1, k, a, j) = 0;
			}
		}
	}

	// regularization
	float reg = 0.0; // 0.0
	for(v = 0; v < nsingle; v++) {

		float xdelta = x[v] - v_centering[v];

		reg += lambda_single * xdelta * xdelta;
		g[v] += 2 * lambda_single * xdelta; // F2 is 2.0
	}

	for(v = nsingle_padded; v < nvar_padded; v++) {
		reg += 0.5 * lambda_pair * x[v] * x[v]; // F05 is 0.5
		g[v] += 2 * lambda_pair * x[v]; // F2 is 2.0
	}

	fx += reg;

	return fx;
}
