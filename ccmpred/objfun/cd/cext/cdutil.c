#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include "cd.h"
#include "cdutil.h"


void seed_rng() {
	int pid;
	struct timeval t;
	gettimeofday(&t, NULL);
	pid = getpid();
	srand(t.tv_usec * t.tv_sec * pid);
}

int pick_random_uniform(int max) {
    int div = RAND_MAX / (max + 1);
    int retval;

    do {
        retval = rand() / div;
    } while (retval > max);

    return retval;
}

int pick_random_weighted(flt *probs, int n) {
	int a;
	double p = (double)rand() / (double)RAND_MAX;
	for (a = 0; a < n; a++) {
		flt p_curr = probs[a];
		if (p < p_curr) {
			return a;
		}
		p -= p_curr;
	}
	return n - 1;
}

flt* fl_malloc(int n) {
	return (flt *)malloc(sizeof(flt) * n);
}

void fl_free(flt *dest) {
	free(dest);
}

void fl_memcpy(flt *dest, flt *src, int n) {
	memcpy(dest, src, sizeof(flt) * n);
}

void vecimulc(flt *dst, flt f, int n) {
	int i;
	for(i = 0; i < n; i++) {
		dst[i] *= f;
	}
}
