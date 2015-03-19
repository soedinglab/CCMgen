#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "treecd.h"
#include "cd.h"
#include "cdutil.h"

/**
 * Mutate a sequence seq nmut times according to potentials in x
 *
 * @param[inout] seq The sequence to work on
 * @param[in] x The single and pairwise emission potentials for computing conditional probabilities
 * @param[in] nmut The number of substitutions to perform
 * @param[in] ncol The length of the sequence
 */
void mutate_sequence(uint8_t *seq, flt *x, uint16_t nmut, int ncol) {

	flt* pcond = fl_malloc(N_ALPHA);

	for(int m = 0; m < nmut; m++) {
		int i = pick_random_uniform(ncol);

		compute_conditional_probs(i, pcond, x, seq, ncol);
		seq[i] = pick_random_weighted(pcond, N_ALPHA - 1);
	}

	fl_free(pcond);
}


void swap(void *a, void *b) {
	void *temp = a;
	a = b;
	b = temp;
}

/**
 * Mutate an ancestral sequence along a tree
 *
 * @param[in] n_children At index i, stores the number of child vertices for vertex i
 * @param[in] branch_lengths At index i, stores the length of the branch leading to vertex i
 * @param[in] x The single and pairwise emission potentials for computing conditional probabilities
 * @param[in] nvert The total number of vertices in the tree
 * @param[in] nleaves The total number of leaves in the tree
 * @param[inout] seqs The ancestral sequence at the beginning of the array. After this method returns, stores all leaf sequences.
 * @param[in] ncol The length of individual sequences
 * @param[in] mutation_rate Coefficient to tune the number of substitutions to make per evolutionary time unit
 */
void mutate_along_tree(
	int32_t *n_children, 
	flt *branch_lengths,
	flt *x,
	uint32_t nvert,
	uint8_t *seqs,
	uint32_t ncol,
	flt mutation_rate
) {

	// Preprocessing: Count number of leaves and compute index of first children
	uint32_t *first_child_index = (uint32_t *)malloc(sizeof(uint32_t) * nvert);
	uint32_t fci = 0;
	uint32_t nleaves = 0;
	for(int i = 0; i < nvert; i++){
		if(n_children[i] == 0) { nleaves++; }
		first_child_index[i] = fci;
		fci += n_children[i];
	}

	// nc: number of children for vertex at index i of current BFS level
	uint32_t *nc_in = (uint32_t *)malloc(sizeof(uint32_t) * nleaves);
	uint32_t *nc_out = (uint32_t *)malloc(sizeof(uint32_t) * nleaves);

	// ni: index of vertex at index i of current BFS level
	uint32_t *ni_in = (uint32_t *)malloc(sizeof(uint32_t) * nleaves);
	uint32_t *ni_out = (uint32_t *)malloc(sizeof(uint32_t) * nleaves);

	uint8_t *seqs_in = (uint8_t *)malloc(sizeof(uint8_t) * ncol * nleaves);
	uint8_t *seqs_out = (uint8_t *)malloc(sizeof(uint8_t) * ncol * nleaves);

	// bl: branch length at index i of current BFS level
	flt *bl = fl_malloc(nleaves);

	// initial level has one vertex with ancestral sequence
	uint32_t nn = 1;
	nc_in[0] = n_children[0];
	ni_in[0] = 0;
	memcpy(seqs_in, seqs, sizeof(uint8_t) * ncol);

	// BFS over tree levels
	while(nn < nleaves) {

		// Phase 1: grow nc_out, ni_out, bl and seqs_out
		int pos = 0;
		for(uint32_t i = 0; i < nn; i++) {

			uint32_t nci = nc_in[i];

			if(nci == 0) {

				// keep a leaf in the subsequent tree level
				nc_out[pos] = nc_in[i];
				ni_out[pos] = ni_in[i];
				bl[pos] = 0;
				memcpy(&seqs_out[pos * ncol], &seqs_in[i * ncol], sizeof(uint8_t) * ncol);

				pos++;

			} else {

				// grow to one or more descendants
				for(uint32_t j = 0; j < nci; j++) {
					uint32_t inew = first_child_index[ni_in[i]] + j;

					nc_out[pos] = n_children[inew];
					ni_out[pos] = inew;
					bl[pos] = branch_lengths[inew];
					memcpy(&seqs_out[pos * ncol], &seqs_in[i * ncol], sizeof(uint8_t) * ncol);

					pos++;
				}

			}

		}

		// Phase 2: evolve seq according to bl
		for(int i = 0; i < pos; i++) {
			int nmut = bl[i] * mutation_rate;
			mutate_sequence(&seqs_out[i * ncol], x, nmut, ncol);
		}

		nn = pos;
		swap(nc_in, nc_out);
		swap(ni_in, ni_out);
		swap(seqs_in, seqs_out);
	}

	memcpy(seqs, seqs_in, sizeof(uint8_t) * ncol * nleaves);

	free(first_child_index);
	free(nc_in);
	free(nc_out);
	free(ni_in);
	free(ni_out);
	free(seqs_in);
	free(seqs_out);
	fl_free(bl);
}
