#ifndef CDUTIL_H
#define CDUTIL_H

void seed_rng();

int pick_random_uniform(int max);
int pick_random_weighted(flt *probs, int n);

flt* fl_malloc(int n);
void fl_free(flt *dest);
void fl_memcpy(flt *dest, flt *src, int n);

void vecimulc(flt *dst, flt f, int n);

#endif
