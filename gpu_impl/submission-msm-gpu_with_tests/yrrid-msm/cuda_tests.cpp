#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "Curve.cu"

void cuda_swap_test() {
  printf("\n");
  BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  FILE *file = fopen("tests/cuda_swap_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.swap(r, b);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}