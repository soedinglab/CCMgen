#ifndef WEIGHTING_H
#define WEIGHTING_H

#include <stdint.h>

void count_ids(
	const uint8_t *msa,
	uint16_t *ids,
	const uint32_t nrow,
	const uint32_t ncol
);

#endif
