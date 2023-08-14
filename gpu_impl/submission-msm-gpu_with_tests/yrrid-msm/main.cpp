#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "HostCurve.cpp"
#include "HostReduce.cpp"
#include "cuda_tests.cpp"

int main() { 
  int a=100;
  printf("qwe %u\n", a);

  setZero_test();
  setOne_test();
  setR_test();
  set_test();
  load_test();
  store_test();
  exportField_test();
  isZero_test();
  isGE_test();

  addN_test();
  subN_test();

  add_test();
  sub_test();
  mul_test();

  shiftRight_test();

  swap_test();
  reduce_test();
  inverse_test();

  reduce_PointXYZZ_test();
  load_PointXYZZ_test();
  store_PointXYZZ_test();
  normalize_PointXYZZ_test();

  set_AccumulatorXYZZ_test();
  setZero_AccumulatorXYZZ_test();
  dbl_AccumulatorXYZZ_test();
  add_AccumulatorXYZZ_test();

  cuda_swap_test();

  printf("\n");

  printf("ready\n");
}