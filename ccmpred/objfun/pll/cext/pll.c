#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "pll.h"

double evaluate_pll(
	const double *x,
	double *g,
	double *g2,
	double *v_centering,
	double *weights,
	unsigned char *msa,
	const uint32_t ncol,
	const uint32_t nrow,
	double lambda_single,
	double lambda_pair
) {
	uint32_t nsingle = ncol * (N_ALPHA - 1);
	uint32_t nsingle_padded = nsingle + N_ALPHA_PAD - (nsingle % N_ALPHA_PAD);
	uint64_t nvar_padded = nsingle_padded + ncol * ncol * N_ALPHA * N_ALPHA_PAD;

	const double *x1 = x;
	const double *x2 = &x[nsingle_padded];

	double *g1 = g;
	double *g2l = &g[nsingle_padded];

	// set fx and gradient to 0 initially
	double fx = 0.0;

	memset(g, 0, sizeof(double) * nvar_padded);
	memset(g2, 0, sizeof(double) * (nvar_padded - nsingle_padded));

	for(int n = 0; n < nrow; n++) {
		double weight = weights[n];

		double precomp[N_ALPHA * ncol] __attribute__ ((aligned (32)));	// aka PC(a,s)
		double precomp_sum[ncol] __attribute__ ((aligned (32)));
		double precomp_norm[N_ALPHA * ncol] __attribute__ ((aligned (32)));	// aka PCN(a,s)

		// compute PC(a,s) = V_s(a) + sum(k \in V_s) w_{sk}(a, X^i_k)
		for(int a = 0; a < N_ALPHA-1; a++) {
			for(int j = 0; j < ncol; j++) {
				PC(a,j) = V(j,a);
			}
		}
		
		for(int j = 0; j < ncol; j++) {
			unsigned char xnj = X(n,j);

			for(int a = 0; a < N_ALPHA - 1; a++) {
				for(int i = 0; i < ncol; i++) {
					PC(a, i) += W(xnj, j, a, i);
				}

			}
		}

		for(int j = 0; j < ncol; j++) {
			PC(N_ALPHA - 1, j) = 0;
		}

		// compute precomp_sum(s) = log( sum(a=1..21) exp(PC(a,s)) )
		memset(precomp_sum, 0, sizeof(double) * ncol);
		for(int a = 0; a < N_ALPHA - 1; a++) {
			for(int j = 0; j < ncol; j++) {
				precomp_sum[j] += expf(PC(a,j));
			}
		}

		for(int j = 0; j < ncol; j++) {
			precomp_sum[j] = logf(precomp_sum[j]);
		}

		for(int a = 0; a < N_ALPHA - 1; a++) {
			for(int j = 0; j < ncol; j++) {
				PCN(a,j) = expf(PC(a, j) - precomp_sum[j]);
			}
		}

		for(int j = 0; j < ncol; j++) {
			PCN(N_ALPHA - 1, j) = 0.0;
		}

		// actually compute fx and gradient
		for(int j = 0; j < ncol; j++) {

			unsigned char xnj = X(n,j);

			fx += weight * (-PC( xnj, j ) + precomp_sum[j]);

			if(xnj < N_ALPHA - 1) {
				G1(j, xnj) -= weight;
			} else {
				for(int a = 0; a < N_ALPHA; a++) {
					PCN(a, j) = 0;
				}

			}


			for(int a = 0; a < N_ALPHA - 1; a++) {
				G1(j, a) += weight * PCN(a, j);
			}

		}

		for(int j = 0; j < ncol; j++) {

			unsigned char xnj = X(n,j);

			for(int i = 0; i < ncol; i++) {
				unsigned char xni = X(n,i);
				G2(xnj, j, xni, i) -= weight;
			}

			for(int a = 0; a < N_ALPHA - 1; a++) {
				for(int i = 0; i < ncol; i++) {
					G2(xnj, j, a, i) += weight * PCN(a, i);
				}
			}
		}

	} // n



	// add transposed onto un-transposed
	for(int b = 0; b < N_ALPHA; b++) {
		for(int j = 0; j < ncol; j++) {
			for(int a = 0; a < N_ALPHA; a++) {
				for(int i = 0; i < ncol; i++) {
					G2L(b, j, a, i) = G2(b, j, a, i) + G2(a, i, b, j);
				}
			}
		}
	}

	// set gradients to zero for self-edges
	for(int b = 0; b < N_ALPHA; b++) {
		for(int j = 0; j < ncol; j++) {
			for(int a = 0; a < N_ALPHA; a++) {
				G2L(b, j, a, j) = 0;
			}
		}
	}

	for(int j = 0; j < ncol; j++) {
		for(int i = 0; i < ncol; i++) {
			for(int a = 0; a < N_ALPHA; a++) {
				G2L(a, j, N_ALPHA - 1, i) = 0;
				G2L(N_ALPHA - 1, j, a, i) = 0;
			}
		}
	}

	// regularization
	double reg = 0.0; // 0.0
	for(int v = 0; v < nsingle; v++) {

		double xdelta = x[v] - v_centering[v];

		reg += lambda_single * xdelta * xdelta;
		g[v] += 2 * lambda_single * xdelta; // F2 is 2.0
	}

	for(int v = nsingle_padded; v < nvar_padded; v++) {
		reg += 0.5 * lambda_pair * x[v] * x[v]; // F05 is 0.5
		g[v] += 2 * lambda_pair * x[v]; // F2 is 2.0
	}

	fx += reg;

	return fx;
}
