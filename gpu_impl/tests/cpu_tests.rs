#![allow(non_snake_case)]
use cudatowgsl::*;

use std::{fs::File, io::Read, vec};

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct TestStruct<I, O> {
  input: I,
  output: O,
}

#[test]
fn cpu_setZero_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setZero_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  setZero(&mut r);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_setOne_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setOne_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  setOne(&mut r);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_setR_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setR_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  setR(&mut r);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_set_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/set_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];

  set(&mut r, field);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_load_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/load_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];

  load(&mut r, &field);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_store_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/store_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];

  store(&mut r, field);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_isZero_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/isZero_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], u32> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let k = isZero( r) as u32;

  assert_eq!(ts.output, k);
}

#[test]
fn cpu_isGE_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/isGE_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], u32> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];

  let k = isGE( r, field) as u32;

  assert_eq!(ts.output, k);
}

#[test]
fn cpu_addN_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/addN_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];

  addN(&mut r, field);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_subN_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/subN_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];

  subN(&mut r, field);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_add_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/add_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];
  let c = [120,110,100,90,80,70,60,50,40,30,20,10];

  add(&mut r, field, c);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_sub_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/sub_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];
  let c = [120,110,100,90,80,70,60,50,40,30,20,10];

  sub(&mut r, field, c);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_mul_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/mul_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];
  let c = [120,110,100,90,80,70,60,50,40,30,20,10];

  mul(&mut r, field, c);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_shiftRight_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/shiftRight_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];

  shiftRight(&mut r, field, 1);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_swap_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/swap_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let mut field = [12,11,10,9,8,7,6,5,4,3,2,1];

  swap(&mut r, &mut field);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_reduce_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/reduce_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];

  reduce(&mut r, field);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_inverse_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/inverse_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let field = [12,11,10,9,8,7,6,5,4,3,2,1];

  inverse(&mut r, field);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_reduce_PointXYZZ_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/reduce_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, PointXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  r.reduce();

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_load_PointXYZZ_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/load_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, PointXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let mut ptr = vec![0; 48];
  for i in 0..48{
    ptr[i] = i as u32;
  }

  r.load(&ptr);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_store_PointXYZZ_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/store_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, Vec<u32>> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let mut ptr = vec![0; 48];
  for i in 0..48{
    ptr[i] = i as u32;
  }

  r.store(&mut ptr);

  assert_eq!(ts.output, ptr);
}

#[test]
fn cpu_normalize_PointXYZZ_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/normalize_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, PointXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  r.normalize();

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_set_AccumulatorXYZZ_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/set_AccumulatorXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<AccumulatorXYZZ, AccumulatorXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  let x = [1,2,3,4,5,6,7,87,8,9,11,12];
  let y = [12,11,10,9,8,7,6,5,4,3,2,1];
  let zz = [12,11,10,9,8,7,6,5,4,3,2,1];
  let zzz = [12,11,10,9,8,7,6,5,4,3,2,1];

  r.set(x, y, zz, zzz);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_setZero_AccumulatorXYZZ_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setZero_AccumulatorXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<AccumulatorXYZZ, AccumulatorXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  r.setZero();

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_dbl_AccumulatorXYZZ_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/dbl_AccumulatorXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<AccumulatorXYZZ, AccumulatorXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  let x = [1,2,3,4,5,6,7,87,8,9,11,12];
  let y = [12,11,10,9,8,7,6,5,4,3,2,1];
  let zz = [12,11,10,9,8,7,6,5,4,3,2,1];
  let zzz = [12,11,10,9,8,7,6,5,4,3,2,1];

  let p = PointXYZZ {
    x,
    y,
    zz,
    zzz,
  };

  r.dbl(p);

  assert_eq!(ts.output, r);
}

#[test]
fn cpu_add_AccumulatorXYZZ_test() {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/add_AccumulatorXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<AccumulatorXYZZ, AccumulatorXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  let x = [1,2,3,4,5,6,7,87,8,9,11,12];
  let y = [12,11,10,9,8,7,6,5,4,3,2,1];
  let zz = [12,11,10,9,8,7,6,5,4,3,2,1];
  let zzz = [12,11,10,9,8,7,6,5,4,3,2,1];

  let p = PointXYZZ {
    x,
    y,
    zz,
    zzz,
  };

  r.add(p);

  assert_eq!(ts.output, r);
}
