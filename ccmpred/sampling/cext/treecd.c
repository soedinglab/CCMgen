#include <stdint.h>
#include <stdio.h>
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
		int i = pick_random_uniform(ncol - 1);

		compute_conditional_probs(i, pcond, x, seq, ncol);
		seq[i] = pick_random_weighted(pcond, N_ALPHA - 1);
//		sample gaps as well (need to adjust E2 and X1 in cd.h but single potentials only have dim 20:
//		compute_conditional_probs_gaps(i, pcond, x, seq, ncol);
//		seq[i] = pick_random_weighted(pcond, N_ALPHA);
	}

	fl_free(pcond);
}

/**
 * Mutate a sequence seq nmut times according to potentials in x
 *
 * @param[inout] seq The sequence to work on
 * @param[in] x The single and pairwise emission potentials for computing conditional probabilities
 * @param[in] nmut The number of substitutions to perform
 * @param[in] ncol The length of the sequence
 */
void mutate_sequence_gibbs(uint8_t *seq, flt *x, uint16_t nmut, int ncol) {

	flt* pcond = fl_malloc(N_ALPHA);

	//int array with elements 1..L
	unsigned int sequence_position_vector[ncol];
	for (unsigned int p=0; p < ncol; p++) sequence_position_vector[p] = p;

	for(int m = 0; m < nmut; m++) {

		shuffle(sequence_position_vector, ncol);

		for (int i=0; i < ncol; i++){
			compute_conditional_probs(sequence_position_vector[i], pcond, x, seq, ncol);
			seq[sequence_position_vector[i]] = pick_random_weighted(pcond, N_ALPHA - 1);
		}
	}

	fl_free(pcond);
}



void swap(void **a, void **b) {
	void *temp = *a;
	*a = *b;
	*b = temp;
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
	uint64_t *n_children,
	flt *branch_lengths,
	flt *x,
	uint64_t nvert,
	uint8_t *seqs,
	uint32_t ncol,
	flt mutation_rate
) {

	seed_rng();

	// Preprocessing: Count number of leaves and compute index of first children
	uint64_t *first_child_index = (uint64_t *)malloc(sizeof(uint64_t) * nvert);
	uint64_t fci = 1;
	uint64_t nleaves = 0;

	for(uint64_t i = 0; i < nvert; i++) {
		if(n_children[i] == 0) { nleaves++; }
		first_child_index[i] = fci;
		fci += n_children[i];
	}

	// nc: number of children for vertex at index i of current BFS level
	uint64_t *nc_in = (uint64_t *)malloc(sizeof(uint64_t) * nleaves);
	uint64_t *nc_out = (uint64_t *)malloc(sizeof(uint64_t) * nleaves);

	// ni: index of vertex at index i of current BFS level
	uint64_t *ni_in = (uint64_t *)malloc(sizeof(uint64_t) * nleaves);
	uint64_t *ni_out = (uint64_t *)malloc(sizeof(uint64_t) * nleaves);

	// seqs: sequences at index i of current BFS level
	uint8_t *seqs_in = (uint8_t *)malloc(sizeof(uint8_t) * ncol * nleaves);
	uint8_t *seqs_out = (uint8_t *)malloc(sizeof(uint8_t) * ncol * nleaves);

	// bl: branch length at index i of current BFS level
	flt *bl = fl_malloc(nleaves);

	// fill initial level with root nodes and ancestral sequences
	uint64_t nn = n_children[0];
	memcpy(nc_in, &n_children[1], sizeof(uint64_t) * nn);
	memcpy(seqs_in, seqs, sizeof(uint8_t) * ncol * nn);
	for(uint64_t i = 0; i < nn; i++) {
		ni_in[i] = i + 1;
	}

	// BFS over tree levels
	while(nn < nleaves) {

		// Phase 1: grow nc_out, ni_out, bl and seqs_out
		uint64_t pos = 0;
		for(uint64_t i = 0; i < nn; i++) {

			uint64_t nci = nc_in[i];

			if(nci == 0) {
				// we have no children - copy the leaf node to keep it in next level
				nc_out[pos] = nc_in[i];
				ni_out[pos] = ni_in[i];
				bl[pos] = 0;
				memcpy(&seqs_out[pos * ncol], &seqs_in[i * ncol], sizeof(uint8_t) * ncol);

				pos++;

			} else {

				// we have one or more children - grow out arrays to make room for descendants
				// mutation to descendant sequences will be handled in phase 2
				for(uint64_t j = 0; j < nci; j++) {
					uint64_t inew = first_child_index[ni_in[i]] + j;

					nc_out[pos] = n_children[inew];
					ni_out[pos] = inew;
					bl[pos] = branch_lengths[inew];
					memcpy(&seqs_out[pos * ncol], &seqs_in[i * ncol], sizeof(uint8_t) * ncol);

					pos++;
				}

			}

		}

		// Phase 2: evolve seq according to bl
		#pragma omp parallel for
		for(uint64_t i = 0; i < pos; i++) {
			int nmut = bl[i] * mutation_rate * ncol;
			//printf("nn = %i, i = %i, nmut = %i, bl[i]=%f\n", nn, i, nmut, bl[i]);
			mutate_sequence(&seqs_out[i * ncol], x, nmut, ncol);
		}

		nn = pos;
		//printf("nn = %i.\n", nn);
		swap((void **)&nc_in, (void **)&nc_out);
		swap((void **)&ni_in, (void **)&ni_out);
		swap((void **)&seqs_in, (void **)&seqs_out);

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
