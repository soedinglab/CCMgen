#ifndef GAP_H
#define GAP_H

#define GAP 20
#define N_ALPHA 21
typedef double flt;

void remove_gaps_probs(
	const flt *const p,
	unsigned char *const msa,
	int nrow,
	int ncol
);

void remove_gaps_consensus(
	unsigned char *const msa,
	unsigned char *const consensus,
	int nrow,
	int ncol
);

#endif
