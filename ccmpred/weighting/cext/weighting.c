#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "weighting.h"

/**
 * Count the number of sequence identities for all rows in an MSA
 *
 * @param[in] seq The MSA to work on
 * @param[out] counts The number of sequence identities
 * @param[in] nrow The number of columns in the MSA
 * @param[in] ncol The number of rows in the MSA
 */
void count_ids(
	const uint8_t *msa,
	uint16_t *ids,
	const uint32_t nrow,
	const uint32_t ncol
) {
	uint32_t nij = nrow * (nrow + 1) / 2;

	for(uint32_t ij = 0; ij < nij; ij++) {

		// compute i and j from ij
		// http://stackoverflow.com/a/244550/1181102
		uint32_t i, j;
		{
			uint32_t ii = nrow * (nrow + 1) / 2 - 1 - ij;
			uint32_t K = floor((sqrt(8 * ii + 1) - 1) / 2);
			i = nrow - 1 - K;
			j = ij - nrow * i + i * (i + 1) / 2;
		}

		uint16_t my_ids = 0;
		for(uint32_t k = 0; k < ncol; k++) {
			if(msa[i * ncol + k] == msa[j * ncol + k]) {
				my_ids++;
			}
		}

		ids[i * nrow + j] = my_ids;
	}

}
