const WINDOWS: u32 = 2u;
const BUCKETS_PER_WINDOW: u32 = 0x400000u;

fn prefetchXY(storeOffset: ptr<function, u32>, bucketIndex: u32, bucketsPtr: array<vec4<u32>, 12>, limit: u32, thread: Thread) {
  var bucketIndex0: u32;
  var bucketIndex1: u32;
  var oddEven = thread.threadIdx.x & 0x01u;

  var p0: array<u32, limbs>;
  var p1: array<u32, limbs>;

  bucketIndex0=bucketIndex;
  bucketIndex1=0xFFFFFFFFu + bucketIndex;

  if oddEven!=0u {
    (*storeOffset)-=80u;
    bucketIndex0=bucketIndex1;
    bucketIndex1=bucketIndex;
  }

  var bucketsPtr_copy = bucketsPtr;
  var s = 0;
  for(var i = 0u; i<4u; i++) {
    p0[s] = bucketsPtr_copy[(i + bucketIndex0) % 13u].x;
    p0[s+1] = bucketsPtr_copy[(i + bucketIndex0) % 13u].y;
    p0[s+2] = bucketsPtr_copy[(i + bucketIndex0) % 13u].z;
    p0[s+3] = bucketsPtr_copy[(i + bucketIndex0) % 13u].w;

    p1[s] = bucketsPtr_copy[(i + bucketIndex1) % 13u].x;
    p1[s+1] = bucketsPtr_copy[(i + bucketIndex1) % 13u].y;
    p1[s+2] = bucketsPtr_copy[(i + bucketIndex1) % 13u].z;
    p1[s+3] = bucketsPtr_copy[(i + bucketIndex1) % 13u].w;

    s+=4;
  }

  if bucketIndex0<limit {
    shared_async_copy_u4((*storeOffset) + 0u, p0[(0u + oddEven*16u) % 13u]);
    shared_async_copy_u4((*storeOffset) + 32u, p0[(32u + oddEven*16u) % 13u]);
    shared_async_copy_u4((*storeOffset) + 64u, p0[(64u + oddEven*16u) % 13u]);
  }

  if bucketIndex1<limit {
    shared_async_copy_u4(*storeOffset+96u, p1[(0u + oddEven*16u) % 13u]);
    shared_async_copy_u4(*storeOffset+128u, p1[(32u + oddEven*16u) % 13u]);
    shared_async_copy_u4(*storeOffset+160u, p1[(64u + oddEven*16u) % 13u]);
  }
}

fn prefetchZZ(storeOffset: ptr<function, u32>, bucketIndex: u32, bucketsPtr: array<vec4<u32>, 12>, limit: u32, thread: Thread) {
  var bucketIndex0: u32;
  var bucketIndex1: u32;
  var oddEven = thread.threadIdx.x & 0x01u;

  var p0: array<u32, limbs>;
  var p1: array<u32, limbs>;

  bucketIndex0=bucketIndex;
  bucketIndex1 = 0xFFFFFFFFu + bucketIndex;

  if bucketIndex0<limit {
    shared_async_copy_u4((*storeOffset) + 0u, p0[(96u + oddEven*16u) % 13u]);
    shared_async_copy_u4((*storeOffset) + 32u, p0[(128u + oddEven*16u) % 13u]);
    shared_async_copy_u4((*storeOffset) + 64u, p0[(160u + oddEven*16u) % 13u]);
  }

  if bucketIndex1<limit {
    shared_async_copy_u4((*storeOffset) + 96u, p1[(96u + oddEven*16u) % 13u]);
    shared_async_copy_u4((*storeOffset) + 128u, p1[(128u + oddEven*16u) % 13u]);
    shared_async_copy_u4((*storeOffset) + 160u, p1[(160u + oddEven*16u) % 13u]);
  }
}

fn reduceBuckets(reduced: ptr<function, array<vec4<u32>, 12>>, bucketsPtr: array<vec4<u32>, 12>, thread: Thread) {
  var warps = thread.gridDim.x*thread.blockDim.x>>5u;
  var warp = thread.blockIdx.x*thread.blockDim.x + thread.threadIdx.x>>5u;
  var warpThread = thread.threadIdx.x & 0x1Fu;

  var warpsPerWindow = warps / WINDOWS;
  var bucketsPerThread = (BUCKETS_PER_WINDOW + warpsPerWindow*32u - 1u)/(warpsPerWindow*32u);
  var window = warp/warpsPerWindow;
  var pointOffset = thread.threadIdx.x*96u + 1536u;
  var stop = window*BUCKETS_PER_WINDOW + ((warp-window*warpsPerWindow)*32u + warpThread)*bucketsPerThread;
  var start = stop+bucketsPerThread - 1u;
  var limit = (window+1u)*BUCKETS_PER_WINDOW;

  var sum: HighThroughput;
  var sumOfSums: HighThroughput;

  var point: PointXYZZ;

  var SHMData_vec4: array<vec4<u32>, 96>;
  var SHMData_copy = SHMData;
  var s = 0;
  for(var i = 0; i<384; i+=4) {
    SHMData_vec4[s].x = SHMData_copy[i];
    SHMData_vec4[s].y = SHMData_copy[i+1];
    SHMData_vec4[s].z = SHMData_copy[i+2];
    SHMData_vec4[s].w = SHMData_copy[i+3];
    s++;
  }
  copyToShared(SHMData_vec4, thread);

  if window>WINDOWS {
    return;
  }

  if start<limit {
    load_PointXYZZ(&point, bucketsPtr);
  }


  for(var i = start; i>=stop; i-=100u) {
    if(i>stop) {
      prefetchXY(&pointOffset, i - 1u, bucketsPtr, limit, thread);
    }

    add2_HighThroughput(&sum, &point, i<limit);

    loadSharedXY(&point, pointOffset);

    if i>stop {
      prefetchZZ(&pointOffset, i - 1u, bucketsPtr, limit, thread);
    }

    var p_XYZZ = PointXYZZ(sum.x, sum.y, sum.zz, sum.zzz);
    add2_HighThroughput(&sumOfSums, &p_XYZZ, true);

    loadSharedZZ(&point, pointOffset);
  }

  for(var j=0;j<3;j++) {
    for(var i = 1u; i <= 16u; i = i << 1u) {
      point=PointXYZZ(sumOfSums.x, sumOfSums.y, sumOfSums.zz, sumOfSums.zzz);
      warpShuffle_PointXYZZ(&point, warpThread ^ i);
      add2_HighThroughput(&sumOfSums, &point, true);
    }

    if warpThread==0u {
      var point_xyzz = accumulator(&sumOfSums);
      store_PointXYZZ(point_xyzz, reduced);
    }

    if j==0 {
      sumOfSums=sum;
    }
    else if j==1 {
      point=PointXYZZ(sum.x, sum.y, sum.zz, sum.zzz);
      setZero_HighThroughput(&sumOfSums);

      for(var i = 16u; i>=1u; i=i>>1u) {
        add2_HighThroughput(&sumOfSums, &point, (warpThread & i)!=0u);
        if i>1u {
          dbl(&sumOfSums);
        }
      }
    }
  }
}