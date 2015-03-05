#ifndef MSACOUNTS_H
#define MSACOUNTS_H

#define N_ALPHA 21

extern void msa_count_single(float *counts, uint8_t *msa, float *weights, uint32_t nrow, uint32_t ncol);
extern void msa_count_pairs(float *counts, uint8_t *msa, float *weights, uint32_t nrow, uint32_t ncol);
extern void msa_char_to_index(uint8_t *msa, uint32_t nrow, uint32_t ncol);

#endif

