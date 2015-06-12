#ifndef TREECD_H
#define TREECD_H

#include <stdint.h>
#include "cd.h"

void mutate_along_tree(
	int32_t *n_children,
	flt *branch_lengths,
	flt *x,
	uint32_t nvert,
	uint8_t *seqs,
	uint32_t ncol,
	flt mutation_rate
);

#endif
