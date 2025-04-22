#include <stdint.h>

extern void take_arr(uint8_t arr[4]);

int main() {
  uint8_t arr[4] = {1, 2, 3, 4};
  take_arr(arr);
}
