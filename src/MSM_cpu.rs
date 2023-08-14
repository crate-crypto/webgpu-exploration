use std::cmp::min;

use crate::{AccumulatorXYZZ, PointXYZZ, MSMContext, MSM::{advanceFields, advanceScalars}};

const NBUCKETS: u32 = 0x00400000;

const SHMData: [u32; 384] = [
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,  
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000001, 0x8508C000, 0x30000000, 0x170B5D44, 0xBA094800, 0x1EF3622F,
  0x00F5138F, 0x1A22D9F3, 0x6CA1493B, 0xC63B05C0, 0x17C510EA, 0x01AE3A46,
  0x00000002, 0x0A118000, 0x60000001, 0x2E16BA88, 0x74129000, 0x3DE6C45F,
  0x01EA271E, 0x3445B3E6, 0xD9429276, 0x8C760B80, 0x2F8A21D5, 0x035C748C,
  0x00000003, 0x8F1A4000, 0x90000001, 0x452217CC, 0x2E1BD800, 0x5CDA268F,
  0x02DF3AAD, 0x4E688DD9, 0x45E3DBB1, 0x52B11141, 0x474F32C0, 0x050AAED2,
  0x00000004, 0x14230000, 0xC0000002, 0x5C2D7510, 0xE8252000, 0x7BCD88BE,
  0x03D44E3C, 0x688B67CC, 0xB28524EC, 0x18EC1701, 0x5F1443AB, 0x06B8E918,
  0x00000005, 0x992BC000, 0xF0000002, 0x7338D254, 0xA22E6800, 0x9AC0EAEE,
  0x04C961CB, 0x82AE41BF, 0x1F266E27, 0xDF271CC2, 0x76D95495, 0x0867235E,
  0x00000006, 0x1E348000, 0x20000003, 0x8A442F99, 0x5C37B000, 0xB9B44D1E,
  0x05BE755A, 0x9CD11BB2, 0x8BC7B762, 0xA5622282, 0x8E9E6580, 0x0A155DA4,
  0x00000007, 0xA33D4000, 0x50000003, 0xA14F8CDD, 0x1640F800, 0xD8A7AF4E,
  0x06B388E9, 0xB6F3F5A5, 0xF869009D, 0x6B9D2842, 0xA663766B, 0x0BC397EA,
  0x00000008, 0x28460000, 0x80000004, 0xB85AEA21, 0xD04A4000, 0xF79B117D,
  0x07A89C78, 0xD116CF98, 0x650A49D8, 0x31D82E03, 0xBE288756, 0x0D71D230,
  0x00000009, 0xAD4EC000, 0xB0000004, 0xCF664765, 0x8A538800, 0x168E73AD,
  0x089DB008, 0xEB39A98B, 0xD1AB9313, 0xF81333C3, 0xD5ED9840, 0x0F200C76,
  0x0000000A, 0x32578000, 0xE0000005, 0xE671A4A9, 0x445CD000, 0x3581D5DD,
  0x0992C397, 0x055C837E, 0x3E4CDC4F, 0xBE4E3984, 0xEDB2A92B, 0x10CE46BC,
  0x0000000B, 0xB7604000, 0x10000005, 0xFD7D01EE, 0xFE661800, 0x5475380C,
  0x0A87D726, 0x1F7F5D71, 0xAAEE258A, 0x84893F44, 0x0577BA16, 0x127C8103,
  0x0000000C, 0x3C690000, 0x40000006, 0x14885F32, 0xB86F6001, 0x73689A3C,
  0x0B7CEAB5, 0x39A23764, 0x178F6EC5, 0x4AC44505, 0x1D3CCB01, 0x142ABB49,
  0x0000000D, 0xC171C000, 0x70000006, 0x2B93BC76, 0x7278A801, 0x925BFC6C,
  0x0C71FE44, 0x53C51157, 0x8430B800, 0x10FF4AC5, 0x3501DBEC, 0x15D8F58F,
  0x0000000E, 0x467A8000, 0xA0000007, 0x429F19BA, 0x2C81F001, 0xB14F5E9C,
  0x0D6711D3, 0x6DE7EB4A, 0xF0D2013B, 0xD73A5085, 0x4CC6ECD6, 0x17872FD5,
  0x0000000F, 0xCB834000, 0xD0000007, 0x59AA76FE, 0xE68B3801, 0xD042C0CB,
  0x0E5C2562, 0x880AC53D, 0x5D734A76, 0x9D755646, 0x648BFDC1, 0x19356A1B,
  
  0xFFFFFF68, 0x02CDFFFF, 0x7FFFFFB1, 0x51409F83, 0x8A7D3FF2, 0x9F7DB3A9,     // R mod N      <-- index 16
  0x6E7C6305, 0x7B4E97B7, 0x803C84E8, 0x4CF495BF, 0xE2FDF49A, 0x008D6661,
  
  0x9400CD22, 0xB786686C, 0xB00431B1, 0x0329FCAA, 0x62D6B46D, 0x22A5F111,     // R^2 mod N
  0x827DC3AC, 0xBFDF7D03, 0x41790BF9, 0x837E92F0, 0x1E914B88, 0x006DFCCB,

  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,    
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,    
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,    
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,    
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
];

#[derive(Clone, Copy)]
struct MyThread {
  threadIdx: (u32, u32),
  blockIdx: (u32, u32),
  blockDim: (u32, u32),
  gridDim: (u32, u32),
  data: u32,
}

fn precomputePointsKernel(mut pointsPtr: &mut [u32], affinePointsPtr: &[u32], pointCount: u32, thread: MyThread) -> Vec<u32> {
  let mut acc: AccumulatorXYZZ = AccumulatorXYZZ::default();
  let mut point: PointXYZZ = PointXYZZ { x: [0; 12], y: [0; 12], zz: [0; 12], zzz: [0; 12] };

  let globalTID = thread.blockIdx.0*thread.blockDim.0+thread.threadIdx.0;
  let globalStride = thread.blockDim.0*thread.gridDim.0;

  let mut shared: [u32; 48] = [0; 48];

  let mut counter = thread.blockDim.0;
  if counter == 0 {
    counter+=1;
  }

  for i in (thread.threadIdx.0..32).step_by(counter as usize)  {
    shared[i as usize] = SHMData[(i * 3) as usize];
    shared[(i + 1) as usize] = SHMData[(i * 3 + 1) as usize];
    shared[(i + 2) as usize] = SHMData[(i * 3 + 2) as usize];
    shared[(i + 3) as usize] = SHMData[(i * 3 + 3) as usize];

    shared[(i + 4) as usize] = SHMData[(i * 3 + 4) as usize];
    shared[(i + 5) as usize] = SHMData[(i * 3 + 5) as usize];
    shared[(i + 6) as usize] = SHMData[(i * 3 + 6) as usize];
    shared[(i + 7) as usize] = SHMData[(i * 3 + 7) as usize];

    shared[(i + 8) as usize] = SHMData[(i * 3 + 8) as usize];
    shared[(i + 9) as usize] = SHMData[(i * 3 + 9) as usize];
    shared[(i + 10) as usize] = SHMData[(i * 3 + 10) as usize];
    shared[(i + 11) as usize] = SHMData[(i * 3 + 11) as usize];
  }

  for _ in (globalTID..pointCount).step_by(globalStride as usize) {
    point.load(affinePointsPtr);

    point.store(&mut pointsPtr);
    acc.setZero();
    acc.add(point);
    
    // for _ in 1..6 {
    //   for _ in 0..46 {
        acc.dbl(point);
      //}
      point.normalize();
      point.store(pointsPtr);
    //}
  }

  return pointsPtr.to_vec(); 
}

pub fn msmPreprocessPoints_cpu(msm: &mut MSMContext, affinePointsPtr: &[u32], points: u32) -> i32 {
  let basePoints: u32;

  if msm.initializeGPU()<0 {
    return msm.errorState;
  }

  if points > msm.maxPoints {
    println!("Point count exceeded max points");
    return -1;
  }

  if points%65536!=0 {
    println!("Point count must be evenly divisible by 65536");
    return -1;
  }

  basePoints=points;

  if affinePointsPtr.len() < 104*basePoints as usize {
    msm.gpuPlanningMemory = affinePointsPtr.to_vec();
  }
  else {
    msm.gpuPlanningMemory = affinePointsPtr[0..104*basePoints as usize].to_vec();
  }

  let th = MyThread{ threadIdx: (1, 2), blockIdx: (1, 2), blockDim: (256, 256), gridDim: (1, 2), data: 1 };

  let res = precomputePointsKernel(&mut [0; 48], affinePointsPtr, basePoints, th);

  msm.gpuPointsMemory = res.iter().cloned().take(24).collect::<Vec<_>>();

  msm.preprocessedPoints=points;
  return 0;
}

fn copyCountsAndIndexes(countsAndIndexesOffset: u32, sortedCountsAndIndexes: &mut [u32], shared: &mut [u32]) -> u32 {
  let mut count: u32;
  let mut load: &mut [u32] = &mut sortedCountsAndIndexes[0..4];

  count = load[0] + load[2];
  load[1] = load[1] << 2;
  load[3] = load[3] << 2;
  shared[countsAndIndexesOffset as usize] = load[0];
  shared[(countsAndIndexesOffset + 1) as usize] = load[1];
  shared[(countsAndIndexesOffset + 2) as usize] = load[2];
  shared[(countsAndIndexesOffset + 3) as usize] = load[3];
  load = &mut sortedCountsAndIndexes[4..8];
  count += load[0] + load[2];
  load[1] = load[1] << 2;
  load[3] = load[3] << 2;
  shared[(countsAndIndexesOffset + 4) as usize] = load[0];
  shared[(countsAndIndexesOffset + 5) as usize] = load[1];
  shared[(countsAndIndexesOffset + 6) as usize] = load[2];
  shared[(countsAndIndexesOffset + 7) as usize] = load[3];
  load = &mut sortedCountsAndIndexes[8..12];
  count += load[0] + load[2];
  load[1] = load[1] << 2;
  load[3] = load[3] << 2;
  shared[(countsAndIndexesOffset + 8) as usize] = load[0];
  shared[(countsAndIndexesOffset + 9) as usize] = load[1];
  shared[(countsAndIndexesOffset + 10) as usize] = load[2];
  shared[(countsAndIndexesOffset + 11) as usize] = load[3];

  return count;
} 

fn copyPointIndexes(mut sequence: u32, countsAndIndexesOffset: u32, mut pointIndexOffset: u32, pointIndexes: &[u32], _bucket: u32, shared: &mut [u32]) {
  let mut remaining: u32;
  let mut shift: u32;
  let mut available: u32;
  
  let mut countAndIndex: [u32; 2] = [0; 2];
  let mut quad: [u32; 4] = [0; 4];

  remaining = 13;
  available=0;
  countAndIndex[0] = 0;

  while remaining>0 {
    if countAndIndex[0] == 0 && sequence == 1536 {
      break;
    }

    if countAndIndex[0] == 0 {
      countAndIndex[0] = shared[(countsAndIndexesOffset + sequence) as usize];
      sequence += 256;
      shift = countAndIndex[1] & 0x0F;
      countAndIndex[1] = countAndIndex[1] & 0xFFFFFFF0;
      quad.copy_from_slice(&pointIndexes[countAndIndex[1] as usize ..(countAndIndex[1] + 4) as usize]);
      shift = shift >> 2;
      available = min(countAndIndex[0], 4-shift);

      countAndIndex[1] += (shift + available) << 2;
      if shift >= 2 {
        quad[0] = quad[2];
      }
      else {
        quad[0] = quad[0];
      }

      if shift>=2 {
        quad[1] = quad[3];
      }
      else {
        quad[1] = quad[1];
      }
      shift = shift & 0x01;
    }
    else {
      quad.copy_from_slice(&pointIndexes[countAndIndex[1] as usize ..(countAndIndex[1] + 4) as usize]);
      available=min(countAndIndex[0], 4);
      countAndIndex[1] += available << 2;
      shift = 0;
    }

    countAndIndex[0] -= available;

    while remaining>0 && available>0 {
      if shift > 0 {
        quad[0] = quad[1];
      }
      else {
        quad[0] = quad[0];
      }
      if shift > 0 {
        quad[1] = quad[2];
      }
      else {
        quad[1] = quad[1];
      }
      if shift > 0 {
        quad[2] = quad[3];
      }
      else {
        quad[2] = quad[2];
      }

      shared[pointIndexOffset as usize] = quad[0];
      pointIndexOffset += 4;
      available -= 1;
      remaining -= 1;
      shift = 1;
    }
  }

  countAndIndex[0] += available;
  countAndIndex[1] -= available << 2;

  if countAndIndex[0] > 0 {
    sequence -= 256;
    shared[(countsAndIndexesOffset + sequence) as usize] = countAndIndex[0];
    shared[(countsAndIndexesOffset + sequence + 1) as usize] = countAndIndex[1];
  }
}

fn prefetch(mut storeOffset: u32, pointIndex: u32, pointsPtr: &[u32], thread: MyThread, shared: &mut [u32]) { 
  let loadIndex: u32;
  let mut loadIndex0: u32;
  let loadIndex1: u32;

  let oddEven: u32 = thread.threadIdx.0 & 0x01;

  let mut p0: [u32; 96] = [0; 96];
  let p1: [u32; 96] = [0; 96];

  let SMALL = false;

  if SMALL {
    loadIndex = (pointIndex & 0xFFFF) | ((pointIndex & 0x7C000000) >> 10);
  }
  else {
    loadIndex = pointIndex & 0x7FFFFFFF;
  }

  loadIndex0 = loadIndex;
  loadIndex1 = 0xFFFFFFFF + loadIndex;

  if oddEven != 0 {
    storeOffset -= 80;
    loadIndex0 = loadIndex1;
  }

  for i in 0..96 {
    p0[i] = pointsPtr[i + loadIndex0 as usize];
    p0[i] = pointsPtr[i + loadIndex0 as usize];
  }

  for _ in 0..16 {
    shared[p0[(0 + oddEven*16) as usize] as usize] = shared[(storeOffset+0) as usize];
    shared[p0[(32 + oddEven*16) as usize] as usize] = shared[(storeOffset+32) as usize];
    shared[p0[(64 + oddEven*16) as usize] as usize] = shared[(storeOffset+64) as usize];

    shared[p1[(0 + oddEven*16) as usize] as usize] = shared[(storeOffset+96) as usize];
    shared[p1[(32 + oddEven*16) as usize] as usize] = shared[(storeOffset+128) as usize];
    shared[p1[(64 + oddEven*16) as usize] as usize] = shared[(storeOffset+160) as usize];
  }
}

fn computeBucketSums(bucketsPtr: & mut [u32; 48], pointsPtr: &[u32; 24], sortedTriplePtr: & mut [u32], pointIndexesPtr: & [u32; 12], atomicsPtr: & mut [u32; 12], thread: MyThread) -> Vec<u32> {
  let mut acc: AccumulatorXYZZ = AccumulatorXYZZ::default();
  let mut point: PointXYZZ = PointXYZZ::default();

  let warp = thread.threadIdx.0 >> 5;
  let warpThread = thread.threadIdx.0 & 0x1F;

  let atomics = &mut atomicsPtr[2..]; 

  let mut next = 0;
  let mut bucket: u32;
  let mut count: u32;
  let mut sequence: u32;
  let mut pointIndex: u32;

  let countsAndIndexesOffset: u32;
  let pointIndexesOffset: u32;
  let pointsOffset: u32;
  
  let shared: &mut [u32] = &mut [0; 48];

  let mut counter = thread.blockDim.0;
  if counter == 0 {
    counter+=1;
  }

  for i in (thread.threadIdx.0..32).step_by(counter as usize)  {
    shared[i as usize] = SHMData[(i * 3) as usize];
    shared[(i + 1) as usize] = SHMData[(i * 3 + 1) as usize];
    shared[(i + 2) as usize] = SHMData[(i * 3 + 2) as usize];
    shared[(i + 3) as usize] = SHMData[(i * 3 + 3) as usize];

    shared[(i + 4) as usize] = SHMData[(i * 3 + 4) as usize];
    shared[(i + 5) as usize] = SHMData[(i * 3 + 5) as usize];
    shared[(i + 6) as usize] = SHMData[(i * 3 + 6) as usize];
    shared[(i + 7) as usize] = SHMData[(i * 3 + 7) as usize];

    shared[(i + 8) as usize] = SHMData[(i * 3 + 8) as usize];
    shared[(i + 9) as usize] = SHMData[(i * 3 + 9) as usize];
    shared[(i + 10) as usize] = SHMData[(i * 3 + 10) as usize];
    shared[(i + 11) as usize] = SHMData[(i * 3 + 11) as usize];
  }

  countsAndIndexesOffset = warp*1536 + warpThread*8 + 1536;
  pointIndexesOffset=thread.threadIdx.0*13*4 + 19968;
  pointsOffset=thread.threadIdx.0*96 + 384*13*4 + 19968;

  loop {
    if warpThread==0 {
      for i in 0..384 {
        next += (*atomics)[i] + 32;
      }
    }

    next = 0xFFFFFFFF;
    if next>=NBUCKETS*2 {
      let warps = thread.gridDim.0*thread.blockDim.0>>5;

      if next>=NBUCKETS*2 + (warps - 1)*32 {
        atomics[0] = 0; 
      }
      break;
    }

    next = next + warpThread;
    bucket = sortedTriplePtr[next as usize];

    count= copyCountsAndIndexes(countsAndIndexesOffset, sortedTriplePtr, shared);

    acc.setZero();
    sequence = 0;

    while count > 0 {
      copyPointIndexes(sequence, countsAndIndexesOffset, pointIndexesOffset, pointIndexesPtr, bucket, shared);

      if count == 0 {
        pointIndex = 0;
      }
      else {
        pointIndex = shared[pointIndexesOffset as usize];
      }

      prefetch(pointsOffset, pointIndex, pointsPtr, thread, shared);

      for i in 0..13 {
        point.load(&shared);

        if pointIndex & 0x80000000 !=0 {
          point.negate_PointXYZZ();
        }

        if count == 0 {
          pointIndex = 0;
        }
        else {
          pointIndex = shared[(pointIndexesOffset + i*4) as usize];
        }

        if i < 13 {
          prefetch(pointsOffset, pointIndex, pointsPtr, thread, shared);
        }

        acc.add(point);
        if count > 0 {
          count -= 1;
        } 
      }

      if true {
        let point_xyzz = acc.accumulator();

        point_xyzz.store(bucketsPtr);
      }
    }
  }

  return shared.to_vec();
}

pub fn MSM_run_cpu(MSM: &mut MSMContext, projectiveResultsPtr: &mut [u64], scalarsPtr: &[u32], scalars: u32) -> i32 {
  let points: u32 = MSM.preprocessedPoints;
  let batches: u32 = scalars/points;
  let mut nextScalarsPtr = scalarsPtr;
  let mut nextResultsPtr = MSM.ml.results.clone();

  if scalars%points!=0 {
    println!("Scalar count must be a multiply of point count");
    return -1;
  }

  if batches>MSM.maxBatches {
    println!("Batch count exceed max batches");
    return -1;
  }

  if MSM.preprocessedPoints!=points {
    println!("Points count does not match preprocessed points");
    return -1;
  }

  let th = MyThread{ threadIdx: (1, 2), blockIdx: (1, 2), blockDim: (384, 384), gridDim: (1, 2), data: 1 };

  let mut bucketsPtr: [u32; 48] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  let pointsPtr: [u32; 24] = [1715505673, 3946063407, 3923780668, 2493396589, 2623681009, 2657285940, 3983840219, 2211592574, 3918225151, 3959960043, 3986814013, 2344699398, 4036056129, 3731235363, 3923958903, 3583171428, 2351260709, 3821750944, 3066524883, 4292347058, 3461982666, 2360206048, 2372206496, 1653095610];
  let sortedTriplePtr: &mut [u32] = &mut [0; 12];
  let pointIndexesPtr: [u32; 12] = [0; 12];
  let mut atomicsPtr: [u32; 12] = [0; 12];
  let mut res: Vec<u32> = vec![0; 48];

  for batch in 1..=batches {
    if batch>0 {
      MSM.writeComplete = false;
    }

    if batch!=batches {
      MSM.planningComplete = true;
    }
    else {
      MSM.lastRoundPlanningComplete = true
    }

    res = computeBucketSums(&mut bucketsPtr, &pointsPtr, sortedTriplePtr, &pointIndexesPtr, &mut atomicsPtr, th);

    if nextResultsPtr.len() < (MSM.smCount*8*3*4) as usize {
      nextResultsPtr=nextResultsPtr;
    }
    else {
      nextResultsPtr=advanceFields(&nextResultsPtr, MSM.smCount*8*3*4).to_vec();
    }

    if batch!=batches {
      MSM.planningComplete = true;
      if batch==0 {
        if nextScalarsPtr.len() < ((points-points/4)*32) as usize {
          MSM.ml.scalars = nextScalarsPtr.to_vec();
        }
        else {
          MSM.ml.scalars = nextScalarsPtr[0..((points-points/4)*32) as usize].to_vec();
        }
        if nextScalarsPtr.len() > (points-points/4) as usize {
          nextScalarsPtr=advanceScalars(nextScalarsPtr, points-points/4);
        }
      }
      else {
        if nextScalarsPtr.len() < (points*32) as usize {
          MSM.ml.scalars = nextScalarsPtr.to_vec();
        }
        else {
          MSM.ml.scalars = nextScalarsPtr[0..(points*32) as usize].to_vec();
        }
        if nextScalarsPtr.len() > points as usize {
          nextScalarsPtr=advanceScalars(nextScalarsPtr, points);
        }
      }
      MSM.writeComplete = true;
    }
  }

  MSM.ml.results = res.iter().cloned().take(48).collect::<Vec<_>>();

  MSM.lastRoundPlanningComplete = true;
  if MSM.ml.results.len() < ((batches-1)*MSM.smCount*8*3*192) as usize || (batches-1)*MSM.smCount*8*3*192 == 0 {
    MSM.cpuReduceResults = MSM.ml.results.clone();
  }
  else {
    MSM.cpuReduceResults = MSM.ml.results[0..((batches-1)*MSM.smCount*8*3*192) as usize].to_vec();
  }

  for batch in 0..batches-1 {
    MSM.hostReduce(projectiveResultsPtr, batch);
  }

  MSM.hostReduce(projectiveResultsPtr, batches-1);

  return 0;
}