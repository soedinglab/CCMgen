#ifndef TREECD_H
#define TREECD_H

#include <stdint.h>
#include "cd.h"

void mutate_along_tree(
	uint64_t *n_children,
	flt *branch_lengths,
	flt *x,
	uint64_t nvert,
	uint8_t *seqs,
	uint32_t ncol,
	flt mutation_rate
);

#endif
