const divisorApprox = array<u32, 24>(
  0x00000000u, 0xFFFFFFFFu, 0x7FFFFFFFu, 0x55555555u, 0x3FFFFFFFu, 0x33333333u,
  0x2AAAAAAAu, 0x24924924u, 0x1FFFFFFFu, 0x1C71C71Cu, 0x19999999u, 0x1745D174u, 
  0x15555555u, 0x13B13B13u, 0x12492492u, 0x11111111u, 0x0FFFFFFFu, 0x0F0F0F0Fu, 
  0x0E38E38Eu, 0x0D79435Eu, 0x0CCCCCCCu, 0x0C30C30Cu, 0x0BA2E8BAu, 0x0B21642Cu);

const GROUP: u32 = 13u;

fn copyCountsAndIndexes(countsAndIndexesOffset: u32, sortedCountsAndIndexes: array<vec4<u32>, 3>, bucket: u32) -> u32 { // +
  var count: u32;
  var load: vec4<u32>;

  load=sortedCountsAndIndexes[0];
  count=load.x + load.z;
  load.y=load.y<<2u;
  load.w=load.w<<2u;
  store_shared_u2(countsAndIndexesOffset, make_uint2(load.x, load.y));
  store_shared_u2(countsAndIndexesOffset + 2u, make_uint2(load.z, load.w));
  load=sortedCountsAndIndexes[1];
  count+=load.x + load.z;
  load.y=load.y<<2u;
  load.w=load.w<<2u;
  store_shared_u2(countsAndIndexesOffset + 4u, make_uint2(load.x, load.y));
  store_shared_u2(countsAndIndexesOffset + 6u, make_uint2(load.z, load.w));
  load=sortedCountsAndIndexes[2];
  count+=load.x + load.z;
  load.y=load.y<<2u;
  load.w=load.w<<2u;
  store_shared_u2(countsAndIndexesOffset + 8u, make_uint2(load.x, load.y));
  store_shared_u2(countsAndIndexesOffset + 10u, make_uint2(load.z, load.w));

  return count;
} 

fn copyPointIndexes(sequence: ptr<function, u32>, countsAndIndexesOffset: u32, pointIndexOffset: ptr<function, u32>, pointIndexes: array<vec4<u32>, limbs>, bucket: u32) { // +
  var remaining: u32;
  var shift: u32;
  var available: u32;

  var countAndIndex: vec2<u32>;
  var quad: vec4<u32>;

  remaining=GROUP;
  available=0u;
  countAndIndex.x=0u;

  var pointIndexes_copy = pointIndexes;

  while(remaining>0u) {
    if countAndIndex.x==0u && *sequence==1536u {
      break;
    }
    if countAndIndex.x==0u {
      countAndIndex=load_shared_u2(countsAndIndexesOffset + (*sequence)); 
      *sequence+=256u;
      shift=countAndIndex.y & 0x0Fu;
      countAndIndex.y=countAndIndex.y & 0xFFFFFFF0u;
      quad = pointIndexes_copy[countAndIndex.y];

      shift=shift>>2u;
      
      available=min(countAndIndex.x, 4u-shift);

      
      countAndIndex.y+=(shift + available)<<2u;
      if shift>=2u {
        quad.x = quad.z;
      }
      else {
        quad.x = quad.x;
      }
      if shift>=2u {
        quad.y = quad.w;
      }
      else {
        quad.y = quad.y;
      }
      shift=shift & 0x01u;
    }
    else {
      quad = pointIndexes_copy[countAndIndex.y];
      available=min(countAndIndex.x, 4u);
      countAndIndex.y+=available<<2u;
      shift=0u;
    }
    countAndIndex.x-=available;
    v_indices[12] = remaining;
    v_indices[13] = available;

    while(remaining>0u && available>0u) {
      if shift>0u {
        quad.x = quad.y;
      }
      else {
        quad.x = quad.x;
      }
      if shift>0u {
        quad.y = quad.z;
      }
      else {
        quad.y = quad.y;
      }
      if shift>0u {
        quad.z = quad.w;
      }
      else {
        quad.z = quad.z;
      }
      store_shared_u32(*pointIndexOffset, quad.x);
      *pointIndexOffset+=4u;
      available--;
      remaining--;
      shift=1u;
    }
  }

  // for(var i = 0; i<i32(limbs); i++ ) {
  //   memory.data[i] = 0xFFFFFFFFu;
  // }
  countAndIndex.x+=available;
  countAndIndex.y-=available<<2u;
  if countAndIndex.x>0u {
    *sequence-=256u;
    store_shared_u2(countsAndIndexesOffset + *sequence, countAndIndex);
  }
}

fn prefetch(storeOffset: ptr<function, u32>, pointIndex: u32, pointsPtr: array<u32, limbs>, thread: Thread) { // +
  var loadIndex: u32;
  var loadIndex0: u32;
  var loadIndex1: u32;
  var oddEven: u32 = thread.threadIdx.x & 0x01u;

  var p0: array<u32, 96>;
  var p1: array<u32, 96>;

  var SMALL = false;

  if SMALL {
    loadIndex=(pointIndex & 0xFFFFu) | ((pointIndex & 0x7C000000u) >> 10u);
  }
  else {
    loadIndex=pointIndex & 0x7FFFFFFFu;
  }

  loadIndex0=loadIndex;
  loadIndex1=0xFFFFFFFFu+loadIndex;

  if oddEven!=0u {
    *storeOffset-=80u;
    loadIndex0=loadIndex1;
    loadIndex1=loadIndex;
  }

  var pointsPtr_copy = pointsPtr;
  for(var i=0u; i<96u; i++){
    p0[i] = pointsPtr_copy[i + loadIndex0];
    p1[i] = pointsPtr_copy[i + loadIndex1];
  }

  
  shared_async_copy_u4(*storeOffset+0u, p0[0u + oddEven*16u]);
  shared_async_copy_u4(*storeOffset+32u, p0[32u + oddEven*16u]);
  shared_async_copy_u4(*storeOffset+64u, p0[64u + oddEven*16u]);
  shared_async_copy_u4(*storeOffset+96u, p1[0u + oddEven*16u]);
  shared_async_copy_u4(*storeOffset+128u, p1[32u + oddEven*16u]);
  shared_async_copy_u4(*storeOffset+160u, p1[64u + oddEven*16u]);
}

fn computeBucketSums(bucketsPtr: ptr<function, array<vec4<u32>, 12>>, pointsPtr: array<u32, limbs>, sortedTriplePtr: array<u32, limbs>, pointIndexesPtr: array<u32, limbs>, atomicsPtr: array<u32, limbs>, thread: Thread, global_id: vec3u) { // +
  var acc: HighThroughput;
  var point: PointXY;

  var warp = global_id.x>>5u;
  var warpThread = global_id.x & 0x1Fu;

  var atomics: array<u32, limbs>;
  var atomicsPtr_copy = atomicsPtr;
  for(var i = 0; i<i32(limbs); i++) {
    atomics[i] = atomicsPtr_copy[(i + 2) % 13];
  }
  var sortedBuckets = sortedTriplePtr;
  var sortedCountsAndIndexes: array<vec4<u32>, 3>;
  var s = 0;
  for(var i = 0; i<i32(limbs); i+=4) {
    sortedCountsAndIndexes[s].x = sortedBuckets[(u32(i) + NBUCKETS*2u + 32u) % 13u];
    sortedCountsAndIndexes[s].y = sortedBuckets[(u32(i) + NBUCKETS*2u + 32u + 1u) % 13u];
    sortedCountsAndIndexes[s].z = sortedBuckets[(u32(i) + NBUCKETS*2u + 32u + 2u) % 13u];
    sortedCountsAndIndexes[s].w = sortedBuckets[(u32(i) + NBUCKETS*2u + 32u + 3u) % 13u];
    s++;
  }
  var pointIndexes: array<vec4<u32>, limbs>;
  var pointIndexesPtr_copy = pointIndexesPtr;
  s = 0;
  for(var i = 0; i<i32(limbs); i+=4) {
    pointIndexes[s].x = pointIndexesPtr_copy[i];
    pointIndexes[s].y = pointIndexesPtr_copy[i+1];
    pointIndexes[s].z = pointIndexesPtr_copy[i+2];
    pointIndexes[s].w = pointIndexesPtr_copy[i+3];
    s++;
  }

  var next: u32;
  var bucket: u32;
  var count: u32;
  var sequence: u32;
  var pointIndex: u32;

  var countsAndIndexesOffset: u32;
  var pointIndexesOffset: u32;
  var pointsOffset: u32;

  var SHMData_vec4: array<vec4<u32>, 96>;
  var SHMData_copy = SHMData;
  s = 0;
  for(var i = 0; i<384; i+=4) {
    SHMData_vec4[s].x = SHMData_copy[i];
    SHMData_vec4[s].y = SHMData_copy[i+1];
    SHMData_vec4[s].z = SHMData_copy[i+2];
    SHMData_vec4[s].w = SHMData_copy[i+3];
    s++;
  }
  copyToShared(SHMData_vec4, thread);

  countsAndIndexesOffset=warp*1536u + warpThread*8u + 1536u;  
  pointIndexesOffset=thread.threadIdx.x*GROUP*4u + 19968u;
  pointsOffset=thread.threadIdx.x*96u + 384u*GROUP*4u + 19968u;

  while(true) {
    if warpThread==0u {
      for(var i = 0; i<384; i++) {
        next+=atomicsPtr_copy[i] + 32u;
      }
      
    }
    next = 0xFFFFFFFFu;
    if next>=NBUCKETS*2u {
      var warps = thread.gridDim.x*thread.blockDim.x>>5u;

      if next>=NBUCKETS*2u + (warps - 1u)*32u {
        atomics[0]=0u;  
      }
      break;
    }
    next=next + warpThread;

    bucket=sortedBuckets[next];
    var sortedCountsAndIndexes_to_func: array<vec4<u32>, 3>;
    s = 0;
    for(var i = 0; i<12; i+=4) {
      sortedCountsAndIndexes_to_func[s].x = sortedCountsAndIndexes[(u32(i) + next*3u) %13u].x;
      sortedCountsAndIndexes_to_func[s].y = sortedCountsAndIndexes[(u32(i) + next*3u) % 13u].y;
      sortedCountsAndIndexes_to_func[s].z = sortedCountsAndIndexes[(u32(i) + next*3u) % 13u].z;
      sortedCountsAndIndexes_to_func[s].w = sortedCountsAndIndexes[(u32(i) + next*3u) % 13u].w;
      s++;
    }

    count=copyCountsAndIndexes(countsAndIndexesOffset, sortedCountsAndIndexes_to_func, bucket);

    setZero_HighThroughput(&acc);

    sequence=0u;

    while(count>0u) {
      copyPointIndexes(&sequence, countsAndIndexesOffset, &pointIndexesOffset, pointIndexes, bucket);
      if count == 0u {
        pointIndex = 0u;
      }
      else {
        pointIndex = load_shared_u32(pointIndexesOffset);
      }
      prefetch(&pointsOffset, pointIndex, pointsPtr, thread);
      for(var i = 0; i<=i32(GROUP);i++) {
        loadShared_PointXY(&point, pointsOffset);

        if (pointIndex & 0x80000000u)!=0u {
          negate_PointXY(&point);
        }

        if count==0u {
          pointIndex = 0u;
        }
        else {
          pointIndex = load_shared_u32(pointIndexesOffset + u32(i)*4u);
        }

        if i<i32(GROUP) {
          prefetch(&pointsOffset, pointIndex, pointsPtr, thread);
        }

        add1_HighThroughput(&acc, &point, count>0u);
        if count>0u {
          count--;
        } 
      }

      if true {
        var point_xyzz = accumulator(&acc);

        store_PointXYZZ(point_xyzz, bucketsPtr);
      }
      else {
        var result: PointXY;

        result = normalize(&acc);
        fromInternal_PointXY(&result);
        var arr: array<vec4<u32>, 6>;

        store_PointXY(result, &arr);

        for (var i : i32 = 0; i < 6; i++) {
          (*bucketsPtr)[i] = arr[i];
        }
      }
    }
  }
}