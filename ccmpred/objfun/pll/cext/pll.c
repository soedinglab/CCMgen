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

	double *precomp_norm = malloc(sizeof(double) * N_ALPHA * nrow * ncol);

	#pragma omp parallel for reduction(+:fx)
	for(int nj = 0; nj < nrow * ncol; nj++) {
		int n = nj / ncol;
		int j = nj % ncol;
		double weight = weights[n];
		double precomp[N_ALPHA];
		double precomp_sum = 0;
		for(int a = 0; a < N_ALPHA - 1; a++) {
			precomp[a] = V(j, a);

			for(int i = 0; i < ncol; i++) {
				unsigned char xni = X(n, i);
				precomp[a] += W(a, j, xni, i);
			}

			precomp_sum += expf(precomp[a]);
		}
		precomp[N_ALPHA - 1] = 0;
		precomp_sum = logf(precomp_sum);

		for(int a = 0; a < N_ALPHA - 1; a++) {
			precomp_norm[(n * N_ALPHA + a) * ncol + j] = expf(precomp[a] - precomp_sum);
		}
		precomp_norm[(n * N_ALPHA + N_ALPHA - 1) * ncol + j] = 0;

		unsigned char xnj = X(n,j);

		fx += weight * (-precomp[xnj] + precomp_sum);

	} // nj

	#pragma omp parallel for
	for(int nj = 0; nj < nrow * ncol; nj++) {
		int n = nj / ncol;
		int j = nj % ncol;
		unsigned char xnj = X(n,j);
		double weight = weights[n];

		if(xnj < N_ALPHA - 1) {
			#pragma omp atomic
			G1(j, xnj) -= weight;

			for(int a = 0; a < N_ALPHA - 1; a++) {
				#pragma omp atomic
				G1(j, a) += weight * precomp_norm[(n * N_ALPHA + a) * ncol + j];
			}
		} else {
			for(int a = 0; a < N_ALPHA; a++) {
				precomp_norm[(n * N_ALPHA + a) * ncol + j] = 0;
			}
		}

		for(int i = 0; i < ncol; i++) {
			unsigned char xni = X(n,i);
			#pragma omp atomic
			G2(xnj, j, xni, i) -= weight;
		}

	} // nj

	#pragma omp parallel for
	for(int nj = 0; nj < nrow * ncol; nj++) {

		int n = nj / ncol;
		int j = nj % ncol;
		double weight = weights[n];
		unsigned char xnj = X(n,j);

		for(int ai = 0; ai < (N_ALPHA - 1) * ncol; ai++) {
				#pragma omp atomic
				g2[((xnj * ncol + j) * N_ALPHA_PAD * ncol + ai)] += weight * precomp_norm[(n * N_ALPHA * ncol) + ai];
		}

	} // nj

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

	free(precomp_norm);

	return fx;
}
