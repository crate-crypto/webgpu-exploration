#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

use std::{fs::File, path::PathBuf, io::{Seek, SeekFrom, Read}, time::Instant};

use crate::{Bindings, GpuConsts, BufCoder};

use crate::host_reduce::HostReduce;

const NBUCKETS: u32 = 0x00400000;
const PAGE_SIZE: u32 = 31744;

const SCRATCH_MAX_COUNT: u32 = 9126;
const SIZE_LIMIT: u32 = (SCRATCH_MAX_COUNT-256)*32; // (SCRATCH_MAX_COUNT-256)*32
const SCRATCH_REQUIRED: u32 = SCRATCH_MAX_COUNT*160; // SCRATCH_MAX_COUNT*160

fn advanceBytes(current: &[u32], bytes: u32) -> &[u32] {
  &current[(bytes) as usize..]
}

pub fn advanceScalars(scalars: &[u32], count: u32) -> &[u32] {
  &scalars[(32 * count) as usize..]
}

pub fn advanceFields(results: &[u32], fieldCount: u32) -> &[u32] {
  &results[(48 * fieldCount) as usize..]
}

pub fn MSMReadHexPoints(mut pointsPtr: &mut [u8], count: u32, path: PathBuf) -> i32 {
  let mut f = File::open(path).expect("Unable to open points.hex file");

  for _ in 0..count {
    for j in 0..2 {
      if !parseHex(&mut pointsPtr[j*48..], &mut f, 48) {
        println!("Points file parse failed\n");
        return -1;
      }
    }

    for j in 0..8 {
      pointsPtr[j + 96]=0;
    }
    pointsPtr=&mut pointsPtr[104..];
  }

  return 0;
}

pub fn MSMReadHexScalars(mut scalarsPtr: &mut [u8], count: u32, path: PathBuf) -> i32 {
  let mut f = File::open(path).expect("Unable to open points.hex file");

  for _ in 0..count {
    if !parseHex(scalarsPtr, &mut f, 32) {
      println!("Points file parse failed\n");
      return -1;
    }
    scalarsPtr = &mut scalarsPtr[32..];
  }

  return 0;
}

fn parseHex(buffer: &mut [u8], file: &mut File, bytes: i32) -> bool {
  file.seek(SeekFrom::Start(0)).unwrap();

  let mut next: i32 = 0;
  let count: i32;
  let mut nibble: i32;
  let mut current: i32;
  let mut nibbles = vec![0; (bytes * 2) as usize];

  for i in 0..bytes as usize {
    buffer[i] = 0;
    nibbles[2 * i] = 0;
    nibbles[2 * i + 1] = 0;
  }

  let mut buffer1 = Vec::new();
  file.read_to_end(&mut buffer1).expect("Could not read file");

  let mut number = 0; 
  // let mut buffer1 = vec![0; 2048];
  // file.take(1).read(&mut buffer1).unwrap() as i32;
  current = buffer1[number] as i32;

  while current as u8 == b' ' || current as u8 == b'\r' || current as u8 == b'\n' || current as u8 == b'\t' || current==-1 {
    if current == -1
    {
      return false;
    }
    number += 1; 
    file.take(1).read(&mut buffer1).unwrap() as i32;
    current = buffer1[number] as i32;
  }

  for _ in 0..bytes * 2 {
    if current as u8 >= b'0'  && current as u8 <= b'9' {
      nibble = current - '0' as i32;
    } 
    else if current as u8 >= b'a' && current as u8 <= b'f' {
      nibble = current - 'a' as i32 + 10;
    } 
    else if current as u8 >= b'A' && current as u8 <= b'F' {
      nibble = current - 'A' as i32 + 10;
    } 
    else {
      break;
    }
    nibbles[next as usize] = nibble;
    next += 1;
    number += 1;
    file.take(1).read(&mut buffer1).unwrap() as i32;
    current = buffer1[number] as i32;
  }

  if current as u8 != b'\r' && current as u8 != b'\n' && current as u8 != b'\t' && current as u8 != b' ' && current != -1 {
    println!("Bad hex value in input file");
    std::process::exit(1);
  }

  count = next;
  for i in 0..count {
    if i % 2 == 0 {
      buffer[(i / 2) as usize] += nibbles[(count - 1 - i) as usize] as u8;
    } 
    else {
      buffer[(i / 2) as usize] += (nibbles[(count - 1 - i) as usize] * 16) as u8; 
    }
  }

  return true;
}

fn MAX128_2(a: u32, b: u32) -> u32 {
  if a>=b {
    return a + 127 & 0xFFFFFF80;
  }
  else {
    return b + 127 & 0xFFFFFF80;
  }
}

fn MAX128_3(a: u32, b: u32, c: u32) -> u32 {
  let mut max: u32;

  if a>=b {
    max = a;
  }
  else {
    max = b;
  }

  if max>=c {
    max = max;
  }
  else {
    max = c;
  }
  return max + 127 & 0xFFFFFF80;
}

#[derive(Default)]
pub struct MemoryLayout {
  overlay1: u32,
  overlay2: u32,
  overlay3: u32,
  overlay4: u32,
  overlay5: u32,

  pub scalars: Vec<u32>,
  pub processedScalars: Vec<u32>,
  pub pages: Vec<u32>,
  pub points: Vec<u32>,
  pub unsortedTriple: Vec<u32>,
  pub sortedTriple: Vec<u32>,
  pub scratch: Vec<u32>,

  pub buckets: Vec<u32>,
  pub results: Vec<u32>,

  pub atomics: Vec<u32>,
  pub sizes: Vec<u32>,
  pub prefixSumSizes: Vec<u32>,
  pub counters: Vec<u64>,
  pub histogram: Vec<u32>,
}

pub struct MSMContext {
  pub ml: MemoryLayout,
  pub errorState: i32,
  pub maxPoints: u32,
  pub maxBatches: u32,
  pub smCount: u32,
  pub preprocessedPoints: u32,

  pub gpuPlanningMemory: Vec<u32>,
  pub gpuPointsMemory: Vec<u32>,
  pub cpuReduceResults: Vec<u32>,

  // cudaStream_t runStream, memoryStream;

  // cudaEvent_t  planningComplete, lastRoundPlanningComplete, writeComplete;
  pub planningComplete: bool,
  pub lastRoundPlanningComplete: bool,
  pub writeComplete: bool,

  // cudaEvent_t  timer0, timer1, timer2, timer3, timer4;
  timer0: Instant,
  timer1: Instant,
  timer2: Instant,
}

impl MSMContext {
  pub fn new(_maxPoints: u32, _maxBatches: u32) -> Self {
    return Self { 
      ml: MemoryLayout::default(), 
      errorState: 0, 
      maxPoints: _maxPoints + 255 & 0xFFFFFF00, 
      maxBatches: _maxBatches, 
      smCount: 0,
      preprocessedPoints: 0, 
      gpuPlanningMemory: Vec::default(), 
      gpuPointsMemory: Vec::default(), 
      cpuReduceResults: Vec::default(), 
      planningComplete: false, 
      lastRoundPlanningComplete: false, 
      writeComplete: false, 
      timer0: Instant::now(), 
      timer1: Instant::now(), 
      timer2: Instant::now(), 
    };
  }

  fn memoryLayoutSize(&mut self) -> usize {
    let mut totalBytes: usize;
    let counters = 11*1024 + 128;
    let pointsPerPage = (PAGE_SIZE-4)/5;
    let pageCount = (self.maxPoints*11 + pointsPerPage - 1)/pointsPerPage + 11*1024;
    let sizeCount = 11*1024;

    let overlay1a: u32;
    let overlay1b: u32;
    let mut overlay2a: u32;
    let overlay3a: u32;
    let overlay3b: u32;
    let overlay3c: u32;

    overlay1a=self.maxPoints*32;
    overlay1b=pageCount*PAGE_SIZE;
    self.ml.overlay1=MAX128_2(overlay1a, overlay1b);

    overlay2a=self.maxPoints*44;
    overlay2a+=NBUCKETS*(11+11)*4;
    self.ml.overlay2=overlay2a;

    overlay3a = self.maxPoints*33;
    overlay3b = self.smCount*(SCRATCH_REQUIRED + 127 & 0xFFFFFF80);
    overlay3c = NBUCKETS*(2+12+12)*4 + 32*(1+6+6)*4;
    self.ml.overlay3 = MAX128_3(overlay3a, overlay3b, overlay3c);

    self.ml.overlay4 = (NBUCKETS+NBUCKETS+32)*192;

    self.ml.overlay5 = (128*4 + counters*8 + sizeCount*4 + sizeCount*4 + 1024*4 + self.smCount*8*3*192*self.maxBatches) + 127 & 0xFFFFFF80;

    totalBytes=self.ml.overlay1 as usize; 
    totalBytes+=self.ml.overlay2 as usize;
    totalBytes+=self.ml.overlay3 as usize;
    totalBytes+=self.ml.overlay4 as usize;
    totalBytes+=self.ml.overlay5 as usize;

    if totalBytes<(104 * self.maxPoints) as usize{
      totalBytes = (104 * self.maxPoints) as usize;
    }
    
    return totalBytes / (1024*2);
  }

  fn initializeMemoryLayout(&mut self) -> i32 {
    // let counters: u32;
    let pointsPerPage: u32;
    let pageCount: u32;
    let sizeCount: u32;

    let overlay = self.gpuPlanningMemory.clone();
    let mut current: Vec<u32>;

    pointsPerPage= (PAGE_SIZE-4)/5;
    // counters=11*1024 + 128;
    sizeCount=11*1024;
    pageCount=(self.maxPoints*11 + pointsPerPage - 1)/pointsPerPage + 11*1024; 

    // OVERLAY 1
    // current = overlay.clone();
    // self.ml.scalars=advanceBytes(&current, self.maxPoints*32).to_vec();

    current = overlay.clone();
    if current.len() < (pageCount*PAGE_SIZE) as usize {
      self.ml.pages = current
    }
    else {
      self.ml.pages=advanceBytes(&current, pageCount*PAGE_SIZE).to_vec();
    }

    // advanceBytes(&overlay, self.ml.overlay1);

    // // OVERLAY 2
    current=overlay.clone();
    if current.len() < (self.maxPoints*44) as usize {
      self.ml.points = current[0..12].to_vec();
    }
    else {
      self.ml.points=advanceBytes(&current, self.maxPoints*44).to_vec();
    }
    if current.len() < (NBUCKETS*(11+11)*4) as usize {
      self.ml.unsortedTriple = current[0..48].to_vec();
    }
    else {
      self.ml.unsortedTriple=advanceBytes(&current, NBUCKETS*(11+11)*4).to_vec();
    }
    
    // advanceBytes(&overlay, self.ml.overlay2);

    // // OVERLAY 3
    current=overlay.clone();
    if current.len() < (self.maxPoints*33) as usize {
      self.ml.processedScalars = current;
    }
    else {
      self.ml.processedScalars=advanceBytes(&current, self.maxPoints*33).to_vec();
    }
  
    current=overlay.clone();
    if current.len() < (self.smCount*(SCRATCH_REQUIRED + 127 & 0xFFFFFF80)) as usize {
      self.ml.scratch = current.clone();
    }
    else {
      self.ml.scratch=advanceBytes(&current, self.smCount*(SCRATCH_REQUIRED + 127 & 0xFFFFFF80)).to_vec();
    }
     
    current=overlay.clone();
    if current.len() < (NBUCKETS*(2+12+12)*4 + 32*(1+6+6)*4) as usize {
      self.ml.sortedTriple=current[0..12].to_vec();
    }
    else {
      self.ml.sortedTriple=advanceBytes(&current, NBUCKETS*(2+12+12)*4 + 32*(1+6+6)*4).to_vec();
    }

    // advanceBytes(&overlay, self.ml.overlay3);

    // OVERLAY 4 
    current=overlay.clone();
    if current.len() < ((NBUCKETS+NBUCKETS+32)*192) as usize {
      self.ml.buckets=current[0..48].to_vec();
    }
    else {
      self.ml.buckets=advanceBytes(&current, (NBUCKETS+NBUCKETS+32)*192).to_vec();
    }
    
    // advanceBytes(&overlay, self.ml.overlay4);

    // // OVERLAY 5 
    current=overlay.clone();
    if current.len() < (128*4) as usize {
      self.ml.atomics = current.clone();
    }
    else {
      self.ml.atomics=advanceBytes(&current, 128*4).to_vec();
    }
    self.ml.counters=current.iter().map(|x| *x as u64).take(12).collect::<Vec<_>>();
    if current.len() < (sizeCount*4) as usize {
      self.ml.sizes = current.clone();
    }
    else {
      self.ml.sizes=advanceBytes(&current, sizeCount*4).to_vec();
    }
    if current.len() < (sizeCount*4) as usize {
      self.ml.prefixSumSizes = current.clone();
    }
    else {
      self.ml.prefixSumSizes=advanceBytes(&current, sizeCount*4).to_vec();
    }
    if current.len() < (1024*4) as usize {
      self.ml.histogram = current.clone();
    }
    else {
      self.ml.histogram=advanceBytes(&current, 1024*4).to_vec();
    }
    
    if current.len() < (self.smCount*8*3*192*self.maxBatches) as usize {
      self.ml.results=current[0..48].to_vec().clone();
    }
    else {
      self.ml.results=advanceBytes(&current, self.smCount*8*3*192*self.maxBatches).to_vec();
    }

    // advanceBytes(&overlay, self.ml.overlay5);

    return 0;  
  }

  pub fn initializeGPU(&mut self) -> i32 {
    if self.errorState!=0 {
      return self.errorState;
    }

    if self.smCount!=0 {
      // we're already initialized
      return 0;
    }

    self.smCount = 1;
    if self.memoryLayoutSize() > 256 {
      self.gpuPlanningMemory = vec![0; 256];
    }
    else {
      self.gpuPlanningMemory = vec![0; self.memoryLayoutSize()];
    }

    self.gpuPointsMemory = vec![0; (96 * 6) as usize];

    if self.initializeMemoryLayout()!=0 {
      return self.errorState;
    }

    //self.cpuReduceResults = vec![0; (self.maxBatches*self.smCount*8*3*192) as usize];

    return 0;
  }

  pub fn hostReduce(&self, projectiveResultsPtr: &mut [u64], batch: u32) {
    let mut hr: HostReduce = HostReduce::default();

    hr.initialize(2, 23, self.smCount*8);

    if projectiveResultsPtr.len() < (batch*6*3) as usize {
      hr.reduce(projectiveResultsPtr, &self.cpuReduceResults);
    }
    else {
      hr.reduce(&mut projectiveResultsPtr[(batch*6*3) as usize ..], &self.cpuReduceResults);
    }
  }

  pub fn batch_msmPreprocessPoints(&mut self, affinePointsPtr: &[u32], points: u32, gpu: &GpuConsts) -> i32 {
    let basePoints: u32;

    self.timer0 = Instant::now();

    if self.errorState!=0 {
      return self.errorState;
    }

    if self.initializeGPU()<0 {
      return self.errorState;
    }
    if points>self.maxPoints {
      println!("Point count exceeded max points");
      return -1;
    }

    if points%65536!=0 {
      println!("Point count must be evenly divisible by 65536");
      return -1;
    }

    basePoints=points;

    if affinePointsPtr.len() < 104*basePoints as usize {
      self.gpuPlanningMemory = affinePointsPtr.to_vec();
    }
    else {
      self.gpuPlanningMemory = affinePointsPtr[0..104*basePoints as usize].to_vec();
    }

    let mut input = vec![0; 100];
    let shared = vec![0; 300];

    input[0] = self.smCount;
    input[1] = 256;
    input[2] = 1536;
    input[3] = 1;

    for i in 0..24 {
      input[i + 4] = self.gpuPointsMemory[i];
    }
    for i in 0..24 {
      input[i + 28] = self.gpuPlanningMemory[i];
    }
    input[52] = basePoints;

    let mut bindings: Bindings = Bindings::initialize_two(input.clone(), shared.clone());

    let bc = BufCoder::initialize(&gpu, &mut bindings, "precomputePointsKernel_batch_call", 2);
       
    let res = pollster::block_on(gpu.run(&bc)).unwrap();

    //let res = pollster::block_on(run(&mut bindings, "precomputePointsKernel_call", 2)); // precomputePointsKernel<<<smCount, 256, 1536>>>(gpuPointsMemory, gpuPlanningMemory, basePoints);

    self.gpuPointsMemory = res.iter().cloned().take(24).collect::<Vec<_>>();
    println!("gpuPointsMemory: {:?}", self.gpuPointsMemory);

    self.preprocessedPoints=points;
    return 0;
  }

  pub fn msmPreprocessPoints(&mut self, affinePointsPtr: &[u32], points: u32, gpu: &GpuConsts) -> i32 {
    let basePoints: u32;

    self.timer0 = Instant::now();

    if self.errorState!=0 {
      return self.errorState;
    }

    if self.initializeGPU()<0 {
      return self.errorState;
    }
    if points>self.maxPoints {
      println!("Point count exceeded max points");
      return -1;
    }

    if points%65536!=0 {
      println!("Point count must be evenly divisible by 65536");
      return -1;
    }

    basePoints=points;

    if affinePointsPtr.len() < 104*basePoints as usize {
      self.gpuPlanningMemory = affinePointsPtr.to_vec();
    }
    else {
      self.gpuPlanningMemory = affinePointsPtr[0..104*basePoints as usize].to_vec();
    }

    let mut input = vec![0; 100];
    let shared = vec![0; 300];

    input[0] = self.smCount;
    input[1] = 256;
    input[2] = 1536;
    input[3] = 1;

    for i in 0..24 {
      input[i + 4] = self.gpuPointsMemory[i];
    }
    for i in 0..24 {
      input[i + 28] = self.gpuPlanningMemory[i];
    }
    input[52] = basePoints;

    let mut bindings: Bindings = Bindings::initialize_two(input.clone(), shared.clone());

    let bc = BufCoder::initialize(&gpu, &mut bindings, "precomputePointsKernel_call", 2);
       
    let res = pollster::block_on(gpu.run(&bc)).unwrap();

    //let res = pollster::block_on(run(&mut bindings, "precomputePointsKernel_call", 2)); // precomputePointsKernel<<<smCount, 256, 1536>>>(gpuPointsMemory, gpuPlanningMemory, basePoints);

    self.gpuPointsMemory = res.iter().cloned().take(24).collect::<Vec<_>>();
    //println!("gpuPointsMemory: {:?}", self.gpuPointsMemory);

    self.preprocessedPoints=points;
    return 0;
  }

  /// number of batches in this function is a result of division a scalars value on a points value
  /// the number of a scalars value can be set using an environment variable
  pub fn msmRun(&mut self, projectiveResultsPtr: &mut [u64], scalarsPtr: &[u32], scalars: u32, gpu: &GpuConsts) -> i32 {
    let points: u32 = self.preprocessedPoints;
    let batches: u32 = scalars/points;
    let mut nextScalarsPtr = scalarsPtr;
    let mut nextResultsPtr = self.ml.results.clone();

    if self.errorState != 0 {
      return self.errorState;
    }

    if scalars == 0 {
      println!("Scalar number must be not zero");
      return -1;
    }
    if scalars%points!=0 {
      println!("Scalar count must be a multiply of point count");
      return -1;
    }

    if batches>self.maxBatches {
      println!("Batch count exceed max batches");
      return -1;
    }

    if self.preprocessedPoints!=points {
      println!("Points count does not match preprocessed points");
      return -1;
    }

    self.ml.scalars = nextScalarsPtr.to_vec(); // CUDA_CHECK(cudaMemcpy(ml.scalars, nextScalarsPtr, points*32u, cudaMemcpyHostToDevice));

    self.timer1 = Instant::now(); // CUDA_CHECK(cudaEventRecord(timer1, runStream));

    let mut input = vec![0; 324];
    let shared = vec![0; 300];
    let mut res = Vec::new();

    input[0] = points/256;
    input[1] = 256;
    input[2] = 8928;
    input[3] = 1;
    for i in 4..16 {
      input[i] = self.ml.processedScalars[i - 4];
    }
    for i in 16..28 {
      input[i] = self.ml.scalars[i - 16];
    }
    input[28] = points;
    

    for i in 0..12 {
      input[(i << 1) + 29] = self.ml.counters[i] as u32; // low 32 bits
      input[(i << 1) + 30] = (self.ml.counters[i] >> 32) as u32; // high 32 bits
    }
    for i in 0..12 {
      input[i + 53] = self.ml.sizes[i];
    }
    for i in 0..12 {
      input[i + 65] = self.ml.atomics[i];
    }
    for i in 0..12 {
      input[i + 77] = self.ml.histogram[i];
    }


    for i in 0..12 {
      input[i + 89] = self.ml.pages[i];
    }
    for i in 0..12 {
      input[i + 101] = self.ml.processedScalars[i];
    }
    input[113] = points;


    for i in 0..12 {
      input[i + 114] = self.ml.prefixSumSizes[i];
    }


    for i in 0..12 {
      input[i + 126] = self.ml.points[i];
    }
    for i in 0..48 {
      input[i + 138] = self.ml.unsortedTriple[i];
    }
    for i in 0..12 {
      input[i + 186] = self.ml.scratch[i];
    }


    for i in 0..12 {
      input[i + 198] = self.ml.unsortedTriple[i];
    }


    for i in 0..12 {
      input[i + 210] = self.ml.sortedTriple[i];
    }


    for i in 0..48 {
      input[i + 222] = self.ml.buckets[i];
    }
    for i in 0..12 {
      input[i + 270] = self.gpuPointsMemory[i];
    }

    let mut bindings: Bindings = Bindings::initialize_two(input.clone(), shared.clone());

    let bc = BufCoder::initialize(&gpu, &mut bindings, "MSM_run_call", 2);

    for batch in 1..=batches {
      if batch>0 {
        self.writeComplete = false;
      }

      // input[0] = points/256;
      // input[1] = 256;
      // input[2] = 8928;
      // input[3] = 1;
      // for i in 4..16 {
      //   input[i] = self.ml.processedScalars[i - 4];
      // }
      // for i in 16..28 {
      //   input[i] = self.ml.scalars[i - 16];
      // }
      
      // input[28] = points;

      // let mut bindings: Bindings = Bindings::initialize_two(input.clone(), shared.clone());

      // let bc = BufCoder::initialize(&gpu, &mut bindings, "processSignedDigitsKernel_call", 2);
      // shared = pollster::block_on(gpu.run(&bc)).unwrap();

      // // shared = pollster::block_on(run(&mut bindings, "processSignedDigitsKernel_call", 2)); // processSignedDigitsKernel<<<points/256, 256, 8928, runStream>>>(ml.processedScalars, ml.scalars, points);

      // input[0] = self.smCount;
      // input[1] = 256;
      // input[2] = 0;
      // input[3] = 1;
      // for i in 0..12 {
      //   input[(i << 1) + 4] = self.ml.counters[i] as u32; // low 32 bits
      //   input[(i << 1) + 5] = (self.ml.counters[i] >> 32) as u32; // high 32 bits
      // }
      // for i in 0..12 {
      //   input[i + 28] = self.ml.sizes[i];
      // }
      // for i in 0..12 {
      //   input[i + 40] = self.ml.atomics[i];
      // }
      // for i in 0..12 {
      //   input[i + 52] = self.ml.histogram[i];
      // }

      // bindings = Bindings::initialize_one(input.clone());

      // let bc = BufCoder::initialize(&gpu, &mut bindings, "initializeCountersSizesAtomicsHistogramKernel_call", 1);
      // let res = pollster::block_on(gpu.run(&bc)).unwrap();

      // //let res = pollster::block_on(run(&mut bindings, "initializeCountersSizesAtomicsHistogramKernel_call", 1)); // initializeCountersSizesAtomicsHistogramKernel<<<smCount, 256, 0, runStream>>>(ml.counters, ml.sizes, ml.atomics, ml.histogram);
      
      // for i in 0..12 {
      //   self.ml.counters[i] = (res[i] as u64) << 32 | (res[i + 12] as u64);
      // }
      // self.ml.sizes = res.iter().cloned().skip(24).take(12).collect::<Vec<_>>();
      // self.ml.atomics = res.iter().cloned().skip(36).take(12).collect::<Vec<_>>();
      // self.ml.histogram = res.iter().cloned().skip(48).take(12).collect::<Vec<_>>();
      
      // input[0] = self.smCount;
      // input[1] = 1024;
      // input[2] = 64*1024;
      // input[3] = 1;
      // for i in 0..12 {
      //   input[i + 4] = self.ml.pages[i];
      // }
      // for i in 0..12 {
      //   input[i + 16] = self.ml.sizes[i];
      // }
      // for i in 0..12 {
      //   input[(i << 1) + 28] = self.ml.counters[i] as u32; // low 32 bits
      //   input[(i << 1) + 29] = (self.ml.counters[i] >> 32) as u32; // high 32 bits
      // }
      // for i in 0..12 {
      //   input[i + 52] = self.ml.processedScalars[i];
      // }
      // input[64] = points;

      // bindings = Bindings::initialize_two(input.clone(), shared.clone());
    
      // let bc = BufCoder::initialize(&gpu, &mut bindings, "partition1024Kernel_call", 2);
      // shared = pollster::block_on(gpu.run(&bc)).unwrap();
      
      // //shared = pollster::block_on(run(&mut bindings, "partition1024Kernel_call", 2)); // CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)partition1024Kernel, dim3(smCount), dim3(1024), partition1024Args, 64*1024, runStream));

      // input[0] = 11;
      // input[1] = 1024;
      // input[2] = 0;
      // input[3] = 1;
      // for i in 0..12 {
      //   input[i + 4] = self.ml.pages[i];
      // }
      // for i in 0..12 {
      //   input[i + 16] = self.ml.prefixSumSizes[i];
      // }
      // for i in 0..12 {
      //   input[i + 28] = self.ml.sizes[i];
      // }
      // for i in 0..12 {
      //   input[(i << 1) + 40] = self.ml.counters[i] as u32; // low 32 bits
      //   input[(i << 1) + 41] = (self.ml.counters[i] >> 32) as u32; // high 32 bits
      // }
      // for i in 0..12 {
      //   input[i + 64] = self.ml.atomics[i];
      // }

      // bindings = Bindings::initialize_two(input.clone(), shared.clone());

      // let bc = BufCoder::initialize(&gpu, &mut bindings, "sizesPrefixSumKernel_call", 2);
      // let res = pollster::block_on(gpu.run(&bc)).unwrap();

      // //let res = pollster::block_on(run(&mut bindings, "sizesPrefixSumKernel_call", 2)); // CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)sizesPrefixSumKernel, dim3(11), dim3(1024), sizesPrefixSumArgs, 0, runStream));
      // self.ml.prefixSumSizes = res.iter().cloned().take(12).collect::<Vec<_>>();
      // self.ml.sizes = res.iter().cloned().skip(12).take(12).collect::<Vec<_>>();
      // shared = res.iter().cloned().skip(24).take(300).collect::<Vec<_>>();

      // input[0] = self.smCount;
      // input[1] = 1024;
      // input[2] = 64*1024;
      // input[3] = 1;

      // for i in 0..12 {
      //   input[i + 4] = self.ml.points[i];
      // }
      // for i in 0..48 {
      //   input[i + 16] = self.ml.unsortedTriple[i];
      // }
      // for i in 0..12 {
      //   input[i + 64] = self.ml.scratch[i];
      // }
      // for i in 0..12 {
      //   input[i + 76] = self.ml.prefixSumSizes[i];
      // }
      // for i in 0..12 {
      //   input[i + 88] = self.ml.sizes[i];
      // }
      // for i in 0..12 {
      //   input[i + 100] = self.ml.pages[i];
      // }
      // for i in 0..12 {
      //   input[i + 112] = self.ml.atomics[i];
      // }
      // input[124] = points;

      // bindings = Bindings::initialize_two(input.clone(), shared.clone());

      // let bc = BufCoder::initialize(&gpu, &mut bindings, "partition4096Kernel_call", 2);
      // let res = pollster::block_on(gpu.run(&bc)).unwrap();

      // //let res = pollster::block_on(run(&mut bindings, "partition4096Kernel_call", 2)); //partition4096Kernel<<<smCount, 1024, 64*1024, runStream>>>(ml.points, ml.unsortedTriple, ml.scratch, ml.prefixSumSizes, ml.sizes, ml.pages, ml.atomics, points);
      // self.ml.points = res.iter().cloned().take(12).collect::<Vec<_>>();
      // self.ml.scratch = res.iter().cloned().skip(12).take(12).collect::<Vec<_>>();
      // shared = res.iter().cloned().skip(24).take(300).collect::<Vec<_>>();

      // input[0] = self.smCount;
      // input[1] = 1024;
      // input[2] = 0;
      // input[3] = 1;

      // for i in 0..12 {
      //   input[i + 4] = self.ml.histogram[i];
      // }
      // for i in 0..12 {
      //   input[i + 16] = self.ml.unsortedTriple[i];
      // }

      // bindings = Bindings::initialize_one(input.clone());

      // let bc = BufCoder::initialize(&gpu, &mut bindings, "histogramPrefixSumKernel_call", 1);
      // let res = pollster::block_on(gpu.run(&bc)).unwrap();

      // //let res = pollster::block_on(run(&mut bindings, "histogramPrefixSumKernel_call", 1)); //histogramPrefixSumKernel<<<smCount, 1024, 0, runStream>>>(ml.histogram, ml.unsortedTriple);

      // self.ml.histogram = res.iter().cloned().take(12).collect::<Vec<_>>();

      // input[0] = self.smCount;
      // input[1] = 1024;
      // input[2] = 96*1024;
      // input[3] = 1;
      // for i in 0..12 {
      //   input[i + 4] = self.ml.sortedTriple[i];
      // }
      // for i in 0..12 {
      //   input[i + 16] = self.ml.histogram[i];
      // }
      // for i in 0..12 {
      //   input[i + 28] = self.ml.unsortedTriple[i];
      // }

      // bindings = Bindings::initialize_two(input.clone(), shared.clone());

      // let bc = BufCoder::initialize(&gpu, &mut bindings, "sortCountsKernel_call", 2);
      // let res = pollster::block_on(gpu.run(&bc)).unwrap();

      // //let res = pollster::block_on(run(&mut bindings, "sortCountsKernel_call", 2)); //sortCountsKernel<<<smCount, 1024, 96*1024, runStream>>>(ml.sortedTriple, ml.histogram, ml.unsortedTriple);
      // self.ml.sortedTriple = res.iter().cloned().take(12).collect::<Vec<_>>();
      // shared = res.iter().cloned().skip(12).take(300).collect::<Vec<_>>();

      if batch!=batches {
        self.planningComplete = true;
      }
      else {
        self.lastRoundPlanningComplete = true
      }

      // input[0] = self.smCount;
      // input[1] = 384;
      // input[2] = 96*1024;
      // input[3] = 1;

      // for i in 0..48 {
      //   input[i + 4] = self.ml.buckets[i];
      // }
      // for i in 0..12 {
      //   input[i + 52] = self.gpuPointsMemory[i];
      // }
      // for i in 0..12 {
      //   input[i + 64] = self.ml.sortedTriple[i];
      // }
      // for i in 0..12 {
      //   input[i + 76] = self.ml.points[i];
      // }
      // for i in 0..12 {
      //   input[i + 88] = self.ml.atomics[i];
      // }

      // bindings = Bindings::initialize_two(input.clone(), shared.clone());

      // let bc = BufCoder::initialize(&gpu, &mut bindings, "computeBucketSums_call", 2);
      // let res = pollster::block_on(gpu.run(&bc)).unwrap();

      // //let res = pollster::block_on(run(&mut bindings, "computeBucketSums_call", 2)); //computeBucketSums<<<smCount, 384, 96*1024, runStream>>>(ml.buckets, gpuPointsMemory, ml.sortedTriple, ml.points, ml.atomics);
      // self.ml.buckets = res.iter().cloned().take(48).collect::<Vec<_>>();
      // shared = res.iter().cloned().skip(48).take(300).collect::<Vec<_>>();

      // // input[0] = self.smCount;
      // // input[1] = 256;
      // // input[2] = 256*96 + 1536;
      // // input[3] = 1;

      // // for i in 0..48 {
      // //   input[i + 4] = nextResultsPtr[i];
      // // }
      // // for i in 0..48 {
      // //   input[i + 52] = self.ml.buckets[i];
      // // }

      // // bindings = Bindings::initialize_two(input.clone(), shared.clone());

      // //let res = pollster::block_on(run(&mut bindings, "reduceBuckets_call", 2)); //reduceBuckets<<<smCount, 256, 256*96 + 1536, runStream>>>(nextResultsPtr, ml.buckets);
      // // nextResultsPtr = res.iter().cloned().take(48).collect::<Vec<_>>();
      // // shared = res.iter().cloned().skip(48).take(300).collect::<Vec<_>>();


      res = pollster::block_on(gpu.run(&bc)).unwrap();




      if nextResultsPtr.len() < (self.smCount*8*3*4) as usize {
        nextResultsPtr=nextResultsPtr;
      }
      else {
        nextResultsPtr=advanceFields(&nextResultsPtr, self.smCount*8*3*4).to_vec();
      }

      if batch!=batches {
        self.planningComplete = true;
        if batch==0 {
          if nextScalarsPtr.len() < ((points-points/4)*32) as usize {
            self.ml.scalars = nextScalarsPtr.to_vec();
          }
          else {
            self.ml.scalars = nextScalarsPtr[0..((points-points/4)*32) as usize].to_vec();
          }
          if nextScalarsPtr.len() > (points-points/4) as usize {
            nextScalarsPtr=advanceScalars(nextScalarsPtr, points-points/4);
          }
        }
        else {
          if nextScalarsPtr.len() < (points*32) as usize {
            self.ml.scalars = nextScalarsPtr.to_vec();
          }
          else {
            self.ml.scalars = nextScalarsPtr[0..(points*32) as usize].to_vec();
          }
          if nextScalarsPtr.len() > points as usize {
            nextScalarsPtr=advanceScalars(nextScalarsPtr, points);
          }
        }
        self.writeComplete = true;
      }
    }

    self.ml.results = res.iter().cloned().take(48).collect::<Vec<_>>();

    self.lastRoundPlanningComplete = true;
    if self.ml.results.len() < ((batches-1)*self.smCount*8*3*192) as usize || (batches-1)*self.smCount*8*3*192 == 0 {
      self.cpuReduceResults = self.ml.results.clone();
    }
    else {
      self.cpuReduceResults = self.ml.results[0..((batches-1)*self.smCount*8*3*192) as usize].to_vec();
    }

    for batch in 0..batches-1 {
      self.hostReduce(projectiveResultsPtr, batch);
    }

    self.timer2 = Instant::now();
  
    self.hostReduce(projectiveResultsPtr, batches-1);

    let ms = self.timer1 - self.timer0;
    println!("Initial copy: {:?}", ms);
    let ms = self.timer2 - self.timer0;
    println!("Total time: {:?}", ms);
    
    return 0;
  }
}
