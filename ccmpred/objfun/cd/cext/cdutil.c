#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "cd.h"
#include "cdutil.h"


void seed_rng() {
	int pid;
	struct timeval t;
	gettimeofday(&t, NULL);
	pid = getpid();
	srand(t.tv_usec * t.tv_sec * pid);
}


/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator. */
void shuffle(unsigned int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}


int pick_random_uniform(int max) {
    int div = RAND_MAX / (max + 1);
    int retval;

    do {
        retval = rand() / div;
    } while (retval > max);

    return retval;
}

//    A      B                        C
//   0.1    0.2                      0.7
// |----|--------|------------------------------------------|
// |    0.1      0.3                                        1

//p<0.1 --> A
//0.1 < p < 0.3 --> p - 0.1 < 0.2 --> B
//p>=0.3 --> p - 0.1 - 0.2 < 0.7 --> C
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
