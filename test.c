#include "stdio.h"

/*
extern double mymain();

int main(void) {
    printf("output of mymain(): %f\n", mymain());
}
*/

extern int mymain();

int main(void) {
    printf("output of mymain(): %d\n", mymain());
}
