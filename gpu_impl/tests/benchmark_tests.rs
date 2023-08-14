#![allow(non_snake_case)]
use cudatowgsl::*;

use criterion::{criterion_group, criterion_main, Criterion};

use std::{fs::File, io::Read, vec};

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct TestStruct<I, O> {
  input: I,
  output: O,
}

fn bench_cpu_setZero(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setZero_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  c.bench_function("setZero_cpu", |b| b.iter(|| setZero(&mut r)));
}

fn bench_gpu_setZero(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setZero_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r.to_vec(), shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "setZero_call", 2);

  c.bench_function("setZero_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_setOne(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setOne_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  c.bench_function("setOne_cpu", |b| b.iter(|| setOne(&mut r)));
}

fn bench_gpu_setOne(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setOne_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r.to_vec(), shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "setOne_call", 2);

  c.bench_function("setOne_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_setR(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setR_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  c.bench_function("setR_cpu", |b| b.iter(|| setR(&mut r)));
}

fn bench_gpu_setR(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setR_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r.to_vec(), shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "setRSquared_call", 2);

  c.bench_function("setR_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_set(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/set_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let f = [12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("set_cpu", |b| b.iter(|| set(&mut r, f)));
}

fn bench_gpu_set(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/set_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let mut bindings: Bindings = Bindings::initialize_one(r.to_vec());

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "set_call", 1);

  c.bench_function("set_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_load(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/load_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let f = [12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("load_cpu", |b| b.iter(|| load(&mut r, &f)));
}

fn bench_gpu_load(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/load_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let mut bindings: Bindings = Bindings::initialize_one(r.to_vec());

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "load_call", 1);

  c.bench_function("load_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_store(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/store_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let f = [12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("store_cpu", |b| b.iter(|| store(&mut r, f)));
}

fn bench_gpu_store(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/store_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let mut bindings: Bindings = Bindings::initialize_one(r.to_vec());

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "store_call", 1);

  c.bench_function("store_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_isZero(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/isZero_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], u32> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  c.bench_function("isZero_cpu", |b| b.iter(|| isZero(r)));
}

fn bench_gpu_isZero(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/isZero_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], u32> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r.to_vec(), shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "isZero_call", 2);

  c.bench_function("isZero_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_addN(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/addN_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let f = [12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("addN_cpu", |b| b.iter(|| addN(&mut r, f)));
}

fn bench_gpu_addN(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/addN_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let mut bindings: Bindings = Bindings::initialize_one(r.to_vec());

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "addN_call", 1);

  c.bench_function("addN_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_add(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/add_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let f = [12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("add_cpu", |b| b.iter(|| add(&mut r, f, f)));
}

fn bench_gpu_add(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/add_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let mut bindings: Bindings = Bindings::initialize_one(r.to_vec());

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "add_call", 1);

  c.bench_function("add_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_sub(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/sub_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let f = [12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("sub_cpu", |b| b.iter(|| sub(&mut r, f, f)));
}

fn bench_gpu_sub(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/sub_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let mut bindings: Bindings = Bindings::initialize_one(r.to_vec());

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "sub_call", 1);


  c.bench_function("sub_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_mul(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/mul_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let f = [12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("mul_cpu", |b| b.iter(|| mul(&mut r, f, f)));
}

fn bench_gpu_mul(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/mul_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let mut bindings: Bindings = Bindings::initialize_one(r.to_vec());

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "mul_call", 1);

  c.bench_function("mul_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_swap(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/swap_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let mut f = [12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("swap_cpu", |b| b.iter(|| swap(&mut r, &mut f)));
}

fn bench_gpu_swap(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/swap_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let mut bindings: Bindings = Bindings::initialize_one(r.to_vec());

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "swap_call", 1);

  c.bench_function("swap_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_reduce(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/reduce_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let f = [12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("reduce_cpu", |b| b.iter(|| reduce(&mut r, f)));
}

fn bench_gpu_reduce(c: &mut Criterion) { // +
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/reduce_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<[u32;12], [u32;12]> = serde_json::from_str(&contents).unwrap();

  let mut r = vec![0; 24];
  let f = [12,11,10,9,8,7,6,5,4,3,2,1];
  for i in 0..12 {
    r[i] = ts.input[i];
    r[i+12] = f[i];
  }
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "reduce_call", 2);

  c.bench_function("reduce_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_reduce_PointXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/reduce_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, PointXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  c.bench_function("reduce_PointXYZZ_cpu", |b| b.iter(|| r.reduce()));
}

fn bench_gpu_reduce_PointXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/reduce_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, PointXYZZ> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let shared = vec![0; 300];

  let v = IntoIterator::into_iter(r.x)
  .chain(IntoIterator::into_iter(r.y))
  .chain(IntoIterator::into_iter(r.zz))
  .chain(IntoIterator::into_iter(r.zzz))
  .collect();

  let mut bindings: Bindings = Bindings::initialize_two(v, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "reduce_PointXYZZ_call", 2);

  c.bench_function("reduce_PointXYZZ_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_load_PointXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/load_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, PointXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let f = [12,11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("load_PointXYZZ_cpu", |b| b.iter(|| r.load(&f)));
}

fn bench_gpu_load_PointXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/load_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, PointXYZZ> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let v = IntoIterator::into_iter(r.x)
  .chain(IntoIterator::into_iter(r.y))
  .chain(IntoIterator::into_iter(r.zz))
  .chain(IntoIterator::into_iter(r.zzz))
  .collect();

  let mut bindings: Bindings = Bindings::initialize_one(v);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "load_PointXYZZ_call", 1);

  c.bench_function("load_PointXYZZ_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_store_PointXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/store_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, Vec<u32>> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let mut f = [12,11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1];

  c.bench_function("store_PointXYZZ_cpu", |b| b.iter(|| r.store(&mut f)));
}

fn bench_gpu_store_PointXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/store_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, Vec<u32>> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let v = IntoIterator::into_iter(r.x)
  .chain(IntoIterator::into_iter(r.y))
  .chain(IntoIterator::into_iter(r.zz))
  .chain(IntoIterator::into_iter(r.zzz))
  .collect();

  let mut bindings: Bindings = Bindings::initialize_one(v);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "store_PointXYZZ_call", 1);

  c.bench_function("store_PointXYZZ_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_normalize_PointXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/normalize_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, PointXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  c.bench_function("normalize_PointXYZZ_cpu", |b| b.iter(|| r.normalize()));
}

fn bench_gpu_normalize_PointXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/normalize_PointXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<PointXYZZ, PointXYZZ> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let shared = vec![0; 300];

  let v = IntoIterator::into_iter(r.x)
  .chain(IntoIterator::into_iter(r.y))
  .chain(IntoIterator::into_iter(r.zz))
  .chain(IntoIterator::into_iter(r.zzz))
  .collect();

  let mut bindings: Bindings = Bindings::initialize_two(v, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "normalize_call", 2);

  c.bench_function("normalize_PointXYZZ_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_setZero_AccumulatorXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setZero_AccumulatorXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<AccumulatorXYZZ, AccumulatorXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;

  c.bench_function("setZero_AccumulatorXYZZ_cpu", |b| b.iter(|| r.setZero()));
}

fn bench_gpu_setZero_AccumulatorXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/setZero_AccumulatorXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<AccumulatorXYZZ, AccumulatorXYZZ> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;

  let v = IntoIterator::into_iter(r.xyzz.x)
  .chain(IntoIterator::into_iter(r.xyzz.y))
  .chain(IntoIterator::into_iter(r.xyzz.zz))
  .chain(IntoIterator::into_iter(r.xyzz.zzz))
  .collect();

  let mut bindings: Bindings = Bindings::initialize_one(v);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "setZero_HighThroughput_call", 1);

  c.bench_function("setZero_AccumulatorXYZZ_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_dbl(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/dbl_AccumulatorXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<AccumulatorXYZZ, AccumulatorXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let p = PointXYZZ::default();

  c.bench_function("dbl_AccumulatorXYZZ_cpu", |b| b.iter(|| r.dbl(p)));
}

fn bench_gpu_dbl(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/dbl_AccumulatorXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<AccumulatorXYZZ, AccumulatorXYZZ> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let shared = vec![0; 300];

  let v = IntoIterator::into_iter(r.xyzz.x)
  .chain(IntoIterator::into_iter(r.xyzz.y))
  .chain(IntoIterator::into_iter(r.xyzz.zz))
  .chain(IntoIterator::into_iter(r.xyzz.zzz))
  .collect();

  let mut bindings: Bindings = Bindings::initialize_two(v, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "dbl_call", 2);

  c.bench_function("dbl_AccumulatorXYZZ_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_add_AccumulatorXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/add_AccumulatorXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<AccumulatorXYZZ, AccumulatorXYZZ> = serde_json::from_str(&contents).unwrap();

  let mut r = ts.input;
  let p = PointXYZZ::default();

  c.bench_function("add_AccumulatorXYZZ_cpu", |b| b.iter(|| r.add(p)));
}

fn bench_gpu_add_AccumulatorXYZZ(c: &mut Criterion) {
  let mut file = File::open("./submission-msm-gpu_with_tests/yrrid-msm/tests/add_AccumulatorXYZZ_test.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();

  let ts: TestStruct::<AccumulatorXYZZ, AccumulatorXYZZ> = serde_json::from_str(&contents).unwrap();

  let r = ts.input;
  let shared = vec![0; 300];

  let v = IntoIterator::into_iter(r.xyzz.x)
  .chain(IntoIterator::into_iter(r.xyzz.y))
  .chain(IntoIterator::into_iter(r.xyzz.zz))
  .chain(IntoIterator::into_iter(r.xyzz.zzz))
  .collect();

  let mut bindings: Bindings = Bindings::initialize_two(v, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "add2_HighThroughput_call", 2);

  c.bench_function("add_AccumulatorXYZZ_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn bench_cpu_msmPreprocessPoints(c: &mut Criterion) {
  let mut msm: MSMContext = MSMContext::new(100000, 100000);
  let affinePointsPtr: [u32; 48] = [0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11]; 

  c.bench_function("cpu_msmPreprocessPoints", |b| b.iter(|| msmPreprocessPoints_cpu(&mut msm, &affinePointsPtr, 65536)));
}

fn bench_msmPreprocessPoints(c: &mut Criterion) {
  let mut msm: MSMContext = MSMContext::new(100000, 100000);
  let affinePointsPtr: [u32; 24] = [0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11]; 

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();

  c.bench_function("msmPreprocessPoints", |b| b.iter(|| msm.msmPreprocessPoints(&affinePointsPtr, 65536, &gpu)));
}

fn bench_cpu_msmRun(c: &mut Criterion) {
  let mut msm: MSMContext = MSMContext::new(100000, 100000);

  let affinePointsPtr: [u32; 48] = [0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11]; 

  msmPreprocessPoints_cpu(&mut msm, &affinePointsPtr, 65536);

  let mut points = [7952349086523354480, 16914211206557533, 1804455504723096531, 0, 0, 0, 7952349086523354480, 16914211206557533, 1804455504723096531, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  let scalars = [3697582722, 2119299629, 2515941055, 2806193226, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

  c.bench_function("cpu_msmRun", |b| b.iter(|| MSM_run_cpu(&mut msm, &mut points, &scalars, 65536)));
}

fn bench_msmRun(c: &mut Criterion) {
  let mut msm: MSMContext = MSMContext::new(100000, 100000);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();

  let affinePointsPtr: [u32; 24] = [0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11]; 
	msm.msmPreprocessPoints(&affinePointsPtr, 65536, &gpu);

  let mut points = [7952349086523354480, 16914211206557533, 1804455504723096531, 0, 0, 0, 7952349086523354480, 16914211206557533, 1804455504723096531, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  let scalars = [3697582722, 2119299629, 2515941055, 2806193226, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

  c.bench_function("msmRun", |b| b.iter(|| msm.msmRun(&mut points, &scalars, 65536, &gpu)));
}

criterion_group!{
  name = benches;
  config = Criterion::default().sample_size(10);
  targets = 
    bench_cpu_setZero, bench_gpu_setZero, bench_cpu_setOne, bench_gpu_setOne,
    bench_cpu_setR, bench_gpu_setR, bench_cpu_set, bench_gpu_set,
    bench_cpu_load, bench_gpu_load, bench_cpu_store, bench_gpu_store,
    bench_cpu_isZero, bench_gpu_isZero, bench_cpu_addN, bench_gpu_addN,
    bench_cpu_add, bench_gpu_add, bench_cpu_sub, bench_gpu_sub,
    bench_cpu_mul, bench_gpu_mul, bench_cpu_swap, bench_gpu_swap,
    bench_cpu_reduce, bench_gpu_reduce, bench_cpu_reduce_PointXYZZ, bench_gpu_reduce_PointXYZZ,
    bench_cpu_load_PointXYZZ, bench_gpu_load_PointXYZZ, bench_cpu_store_PointXYZZ, bench_gpu_store_PointXYZZ,
    bench_cpu_normalize_PointXYZZ, bench_gpu_normalize_PointXYZZ, bench_cpu_setZero_AccumulatorXYZZ, bench_gpu_setZero_AccumulatorXYZZ,
    bench_cpu_dbl, bench_gpu_dbl, bench_cpu_add_AccumulatorXYZZ, bench_gpu_add_AccumulatorXYZZ,
    bench_msmPreprocessPoints, bench_msmRun, 
    bench_cpu_msmPreprocessPoints, bench_cpu_msmRun
}

fn batch_cpu_setZero(c: &mut Criterion) {
  c.bench_function("setZero_batch_cpu", |b| b.iter(|| setZero_batch()));
}

fn batch_gpu_setZero(c: &mut Criterion) {
  let r = vec![1; 60000];
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "setZero_batch_call", 2);

  c.bench_function("setZero_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_setOne(c: &mut Criterion) {
  c.bench_function("setOne_batch_cpu", |b| b.iter(|| setOne_batch()));
}

fn batch_gpu_setOne(c: &mut Criterion) {
  let r = vec![1; 60000];
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "setOne_batch_call", 2);

  c.bench_function("setOne_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_setR(c: &mut Criterion) {
  c.bench_function("setR_batch_cpu", |b| b.iter(|| setR_batch()));
}

fn batch_gpu_setR(c: &mut Criterion) {
  let r = vec![1; 60000];
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "setR_batch_call", 2);

  c.bench_function("setR_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_set(c: &mut Criterion) {
  c.bench_function("set_batch_cpu", |b| b.iter(|| set_batch()));
}

fn batch_gpu_set(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "set_batch_call", 1);

  c.bench_function("set_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_load(c: &mut Criterion) {
  c.bench_function("load_batch_cpu", |b| b.iter(|| load_batch()));
}

fn batch_gpu_load(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "load_batch_call", 1);

  c.bench_function("load_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_store(c: &mut Criterion) {
  c.bench_function("store_batch_cpu", |b| b.iter(|| store_batch()));
}

fn batch_gpu_store(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "store_batch_call", 1);

  c.bench_function("store_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_isZero(c: &mut Criterion) {
  c.bench_function("isZero_batch_cpu", |b| b.iter(|| isZero_batch()));
}

fn batch_gpu_isZero(c: &mut Criterion) {
  let r = vec![1; 60000];
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "isZero_batch_call", 2);

  c.bench_function("isZero_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_addN(c: &mut Criterion) {
  c.bench_function("addN_batch_cpu", |b| b.iter(|| addN_batch()));
}

fn batch_gpu_addN(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "addN_batch_call", 1);

  c.bench_function("addN_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_add(c: &mut Criterion) {
  c.bench_function("add_batch_cpu", |b| b.iter(|| add_batch()));
}

fn batch_gpu_add(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "add_batch_call", 1);

  c.bench_function("add_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_sub(c: &mut Criterion) {
  c.bench_function("sub_batch_cpu", |b| b.iter(|| sub_batch()));
}

fn batch_gpu_sub(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "sub_batch_call", 1);

  c.bench_function("sub_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_mul(c: &mut Criterion) {
  c.bench_function("mul_batch_cpu", |b| b.iter(|| mul_batch()));
}

fn batch_gpu_mul(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "mul_batch_call", 1);

  c.bench_function("mul_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_swap(c: &mut Criterion) {
  c.bench_function("swap_batch_cpu", |b| b.iter(|| swap_batch()));
}

fn batch_gpu_swap(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "swap_batch_call", 1);

  c.bench_function("swap_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_reduce(c: &mut Criterion) {
  c.bench_function("reduce_batch_cpu", |b| b.iter(|| reduce_batch()));
}

fn batch_gpu_reduce(c: &mut Criterion) { // +
  let r = vec![1; 60000];
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "reduce_batch_call", 2);

  c.bench_function("reduce_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_reduce_PointXYZZ(c: &mut Criterion) {
  c.bench_function("reduce_PointXYZZ_batch_cpu", |b| b.iter(|| reduce_PointXYZZ_batch()));
}

fn batch_gpu_reduce_PointXYZZ(c: &mut Criterion) {
  let r = vec![1; 60000];
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "reduce_PointXYZZ_batch_call", 2);

  c.bench_function("reduce_PointXYZZ_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_load_PointXYZZ(c: &mut Criterion) {
  c.bench_function("load_PointXYZZ_batch_cpu", |b| b.iter(|| load_PointXYZZ_batch()));
}

fn batch_gpu_load_PointXYZZ(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "load_PointXYZZ_batch_call", 1);

  c.bench_function("load_PointXYZZ_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_store_PointXYZZ(c: &mut Criterion) {
  c.bench_function("store_PointXYZZ_batch_cpu", |b| b.iter(|| store_PointXYZZ_batch()));
}

fn batch_gpu_store_PointXYZZ(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "store_PointXYZZ_batch_call", 1);

  c.bench_function("store_PointXYZZ_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_normalize_PointXYZZ(c: &mut Criterion) {
  c.bench_function("normalize_PointXYZZ_batch_cpu", |b| b.iter(|| normalize_PointXYZZ_batch()));
}

fn batch_gpu_normalize_PointXYZZ(c: &mut Criterion) {
  let r = vec![1; 60000];
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "normalize_batch_call", 2);

  c.bench_function("normalize_PointXYZZ_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_setZero_AccumulatorXYZZ(c: &mut Criterion) {
  c.bench_function("setZero_AccumulatorXYZZ_batch_cpu", |b| b.iter(|| setZero_AccumulatorXYZZ_batch()));
}

fn batch_gpu_setZero_AccumulatorXYZZ(c: &mut Criterion) {
  let r = vec![1; 60000];

  let mut bindings: Bindings = Bindings::initialize_one(r);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "setZero_AccumulatorXYZZ_batch_call", 1);

  c.bench_function("setZero_AccumulatorXYZZ_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_dbl_AccumulatorXYZZ(c: &mut Criterion) {
  c.bench_function("dbl_AccumulatorXYZZ_batch_cpu", |b| b.iter(|| dbl_AccumulatorXYZZ_batch()));
}

fn batch_gpu_dbl_AccumulatorXYZZ(c: &mut Criterion) {
  let r = vec![1; 60000];
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "dbl_AccumulatorXYZZ_batch_call", 2);

  c.bench_function("dbl_AccumulatorXYZZ_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_add_AccumulatorXYZZ(c: &mut Criterion) {
  c.bench_function("add_AccumulatorXYZZ_batch_cpu", |b| b.iter(|| add_AccumulatorXYZZ_batch()));
}

fn batch_add_dbl_AccumulatorXYZZ(c: &mut Criterion) {
  let r = vec![1; 60000];
  let shared = vec![0; 300];

  let mut bindings: Bindings = Bindings::initialize_two(r, shared);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "add_AccumulatorXYZZ_batch_call", 2);

  c.bench_function("add_AccumulatorXYZZ_batch_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn batch_cpu_msmPreprocessPoints(c: &mut Criterion) {
  c.bench_function("batch_cpu_msmPreprocessPoints", |b| b.iter(|| msmPreprocessPoints_cpu_batch()));
}

fn batch_gpu_msmPreprocessPoints(c: &mut Criterion) {
  c.bench_function("batch_msmPreprocessPoints", |b| b.iter(|| msmPreprocessPoints_gpu_batch()));
}

fn batch_msmRun(c: &mut Criterion) {
  let mut msm: MSMContext = MSMContext::new(100000, 1000000);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();

  let affinePointsPtr: [u32; 24] = [0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11]; 
	msm.msmPreprocessPoints(&affinePointsPtr, 65536, &gpu);

  let mut points = [7952349086523354480, 16914211206557533, 1804455504723096531, 0, 0, 0, 7952349086523354480, 16914211206557533, 1804455504723096531, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  let scalars = [3697582722, 2119299629, 2515941055, 2806193226, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

  c.bench_function("batch_msmRun", |b| b.iter(|| msm.msmRun(&mut points, &scalars, 65536*1500, &gpu)));
}

fn batch_cpu_msmRun(c: &mut Criterion) {
  let mut msm: MSMContext = MSMContext::new(100000, 10000000);

  let affinePointsPtr: [u32; 48] = [0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11, 0, 1,2,3,4,5,6,70,8,9,10,11]; 

  msmPreprocessPoints_cpu(&mut msm, &affinePointsPtr, 65536);

  let mut points = [7952349086523354480, 16914211206557533, 1804455504723096531, 0, 0, 0, 7952349086523354480, 16914211206557533, 1804455504723096531, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  let scalars = [3697582722, 2119299629, 2515941055, 2806193226, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

  c.bench_function("cpu_msmRun", |b| b.iter(|| MSM_run_cpu(&mut msm, &mut points, &scalars, 65536*1500)));
}

criterion_group!{
  name = batches;
  config = Criterion::default().sample_size(10);
  targets = 
    batch_cpu_setZero, batch_gpu_setZero, batch_cpu_setOne, batch_gpu_setOne,
    batch_cpu_setR, batch_gpu_setR, batch_cpu_set, batch_gpu_set,
    batch_cpu_load, batch_gpu_load, batch_cpu_store, batch_gpu_store,
    batch_cpu_isZero, batch_gpu_isZero, batch_cpu_addN, batch_gpu_addN,
    batch_cpu_add, batch_gpu_add, batch_cpu_sub, batch_gpu_sub,
    batch_cpu_mul, batch_gpu_mul, batch_cpu_swap, batch_gpu_swap,
    batch_cpu_reduce, batch_gpu_reduce, batch_cpu_reduce_PointXYZZ, batch_gpu_reduce_PointXYZZ,
    batch_cpu_load_PointXYZZ, batch_gpu_load_PointXYZZ, batch_cpu_store_PointXYZZ, batch_gpu_store_PointXYZZ,
    batch_cpu_normalize_PointXYZZ, batch_gpu_normalize_PointXYZZ, batch_cpu_setZero_AccumulatorXYZZ, batch_gpu_setZero_AccumulatorXYZZ,
    batch_cpu_dbl_AccumulatorXYZZ, batch_gpu_dbl_AccumulatorXYZZ, batch_cpu_add_AccumulatorXYZZ, batch_add_dbl_AccumulatorXYZZ,
    batch_cpu_msmPreprocessPoints, batch_gpu_msmPreprocessPoints,
    batch_cpu_msmRun, batch_msmRun
}

fn add_two(a: u32, b: u32) -> u32 {
  return a + b;
}

fn bench_cpu_add_two(c: &mut Criterion) {
  let a = 100;
  let a2 = 200;

  c.bench_function("add_two_cpu", |b| b.iter(|| add_two(a, a2)));
}

fn bench_gpu_add_two(c: &mut Criterion) {
  let v = vec![100, 200];

  let mut bindings: Bindings = Bindings::initialize_one(v);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "add_two_call", 1);

  c.bench_function("add_two_gpu", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

fn add_two_vec(a: &[u32], b: &[u32]) -> Vec<u32> {
  let mut res = Vec::with_capacity(10000000);

  for i in 0..10000000 {
    res.push(a[i] + b[i]);
  }

  return res;
}

fn bench_cpu_add_two_vec(c: &mut Criterion) {
  let v = vec![1; 10000000];
  let v2 = vec![2; 10000000];

  c.bench_function("add_two_vec_cpu", |b| b.iter(|| add_two_vec(&v, &v2)));
}

fn bench_gpu_add_two_vec(c: &mut Criterion) {
  let mut v = Vec::new();

  for _ in 0..1000000 {
    v.push(1);
  }

  let mut bindings: Bindings = Bindings::initialize_one(v);

  let gpu = pollster::block_on(GpuConsts::initialaze()).unwrap();
  let bc = BufCoder::initialize(&gpu, &mut bindings, "add_two_vec_call", 1);

  c.bench_function("add_add_two_vec", |b| b.iter(|| pollster::block_on(gpu.run(&bc)).unwrap()));
}

criterion_group!{
  name = tests;
  config = Criterion::default().sample_size(10);
  targets = 
    bench_cpu_add_two, bench_gpu_add_two, 
    bench_cpu_add_two_vec, bench_gpu_add_two_vec
}

criterion_main!(
  benches,
  batches,
  tests
);
