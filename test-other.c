#include "stdio.h"
#include <inttypes.h>
#include <stdint.h>

typedef struct {
    uint8_t tag;
    int64_t val;
} Out;

extern Out mymain();

int main(void) {
    Out out = mymain();
    printf("val = 0x%" PRIx64 "\n", (uint64_t) out.val);
    printf("tag: %i\n", out.tag);

    double x = 0x7ffc501069c8;
    printf("tag: %f\n", x);
}
