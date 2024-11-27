#include "stdio.h"
#include <inttypes.h>
#include <stdint.h>

typedef struct {
    int32_t a;
    float b;
} Out;

extern Out mymain();

int main(void) {
    Out out = mymain();
    /*
    printf("val = 0x%" PRIx64 "\n", (uint64_t) out.val);
    printf("tag: %i\n", out.tag);

    double x = 0x7ffc501069c8;
    printf("tag: %f\n", x);
    */
    //printf("a: %" PRId64 "\n", out.a);
    printf("a: %i\n", out.a);
    //double *p = (double *) &out.b;
    printf("b: %f\n", out.b);
}
