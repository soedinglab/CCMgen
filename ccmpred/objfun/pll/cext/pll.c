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
	double *weights,
	unsigned char *msa,
	const uint32_t ncol,
	const uint32_t nrow
) {
	uint32_t nsingle = ncol * N_ALPHA;
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

	//#pragma omp parallel for reduction(+:fx)
	for(uint32_t nj = 0; nj < nrow * ncol; nj++) {
		uint32_t n = nj / ncol;
		uint32_t j = nj % ncol;
		double weight = weights[n];
		double precomp[N_ALPHA];
		double precomp_sum = 0;
		for(int a = 0; a < N_ALPHA - 1; a++) {
			precomp[a] = V(a, j);

			for(uint32_t i = 0; i < ncol; i++) {
				unsigned char xni = X(n, i);

				if (xni < N_ALPHA - 1) {
					precomp[a] += W(a, j, xni, i);
				}
			}

			precomp_sum += exp(precomp[a]);
		}
		precomp[N_ALPHA - 1] = 0;
		precomp_sum = log(precomp_sum);


		for(int a = 0; a < N_ALPHA - 1; a++) {
			precomp_norm[(n * N_ALPHA + a) * ncol + j] = exp(precomp[a] - precomp_sum);
		}
		precomp_norm[(n * N_ALPHA + N_ALPHA - 1) * ncol + j] = 0;

		unsigned char xnj = X(n,j);

		if(xnj < N_ALPHA - 1) {
			fx += weight * (precomp_sum - precomp[xnj]);
		}

	} // nj

	#pragma omp parallel for
	for(uint32_t nj = 0; nj < nrow * ncol; nj++) {
		uint32_t n = nj / ncol;
		uint32_t j = nj % ncol;
		unsigned char xnj = X(n,j);
		double weight = weights[n];

		if(xnj < N_ALPHA - 1) {

			for(uint32_t a = 0; a < N_ALPHA - 1; a++) {
				#pragma omp atomic
				G1(a, j) += weight * precomp_norm[(n * N_ALPHA + a) * ncol + j];
			}
		} else {
			for(uint32_t a = 0; a < N_ALPHA; a++) {
				precomp_norm[(n * N_ALPHA + a) * ncol + j] = 0;
			}
		}

	} // nj

	#pragma omp parallel for
	for(uint32_t nj = 0; nj < nrow * ncol; nj++) {

		uint32_t n = nj / ncol;
		uint32_t j = nj % ncol;
		double weight = weights[n];
		unsigned char xnj = X(n,j);

		for(uint8_t a = 0; a < N_ALPHA - 1; a++) {
			for(uint32_t i = 0; i < ncol; i++) {
				#pragma omp atomic
				g2[((xnj * ncol + j) * N_ALPHA_PAD + a) * ncol + i ] += weight * precomp_norm[(n * N_ALPHA + a) * ncol + i];

			}
		}

	} // nj

	// add transposed onto un-transposed
	for(uint32_t b = 0; b < N_ALPHA; b++) {
		for(uint32_t j = 0; j < ncol; j++) {
			for(uint32_t a = 0; a < N_ALPHA; a++) {
				for(uint32_t i = 0; i < ncol; i++) {
					G2L(b, j, a, i) = G2(b, j, a, i) + G2(a, i, b, j);
				}
			}
		}
	}

	// set gradients to zero for self-edges
	for(uint32_t b = 0; b < N_ALPHA; b++) {
		for(uint32_t j = 0; j < ncol; j++) {
			for(uint32_t a = 0; a < N_ALPHA; a++) {
				G2L(b, j, a, j) = 0;
			}
		}
	}

	for(uint32_t j = 0; j < ncol; j++) {
		for(uint32_t i = 0; i < ncol; i++) {
			for(uint32_t a = 0; a < N_ALPHA; a++) {
				G2L(a, j, N_ALPHA - 1, i) = 0;
				G2L(N_ALPHA - 1, j, a, i) = 0;
			}
		}
	}

	free(precomp_norm);

	return fx;
}
