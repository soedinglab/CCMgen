#ifndef CD_H
#define CD_H

#define N_ALPHA 21
#define GAP 20

typedef double flt;
#define F0 0.0
#define F1 1.0
#define F2 2.0
#define fexp exp
#define flog log


#define X1_INDEX(i,a) (i) * (N_ALPHA - 1) + (a)
#define X2_INDEX(i,a,j,b) (((i) * N_ALPHA + (a)) * ncol + (j)) * N_ALPHA + (b)

#define G1(i,a) g[X1_INDEX(i,a)]
#define G2(i,a,j,b) g[nsingle + X2_INDEX(i,a,j,b)]
#define E1(i,a) x[X1_INDEX(i,a)]
#define E2(i,a,j,b) x[nsingle + X2_INDEX(i,a,j,b)]
#define H1(i,a) h[X1_INDEX(i,a)]
#define H2(i,a,j,b) h[nsingle + X2_INDEX(i,a,j,b)]


#define MSA(n,i) msa[MSA_INDEX(n,i)]

#define MSA_INDEX(n,i) (n) * ncol + (i)

void compute_conditional_probs(
	const int i,
	flt *const cond_probs,
	const flt *const x,
	const unsigned char *const last_seq,
	const int ncol
);

#endif
