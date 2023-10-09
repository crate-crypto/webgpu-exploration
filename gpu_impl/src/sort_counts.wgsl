fn histogramPrefixSumKernel(histogramPtr: ptr<function, array<u32, limbs>>, unsortedTriplePtr: array<u32, limbs>, thread: Thread, global_id: vec3u) { // +
  var globalTID = thread.blockIdx.x*thread.blockDim.x+thread.threadIdx.x;
  var globalStride = thread.gridDim.x*thread.blockDim.x;

  // var histogram = histogramPtr;
  var counts = unsortedTriplePtr;

  var count: u32;
  var localSum: u32;

  var i: u32;

  var sharedHistogram: array<u32, 1024>;
  var warpTotals: array<u32, 32>;

  sharedHistogram[thread.threadIdx.x]=0u;

  globalStride++;

  for(i = globalTID; i < 100u; i+=globalStride) {
    count=0u;

    for(var j = 0u; j<=10u*100u; j+=2u*100u) {
      if j + i > limbs {
        count+=counts[(j + i) % 13u];
      }
      else {
        count+=counts[j + i];
      }
    }
    count=min(count, 1023u);
    sharedHistogram[1023u-count] += 1u;
  }

  for(; i<2u*400u; i+=globalStride) {
    count=0u;

    for(var j = 0u; j<=8u * 400u; j+=2u * 400u) {
      if j + i > limbs {
        count+=counts[(j + i) % 13u];
      }
      else {
        count+=counts[j + i];
      }
    }
    count=min(count, 1023u);
    sharedHistogram[1023u - count] += 1u;
  }

  count=sharedHistogram[thread.threadIdx.x];
  localSum=multiwarpPrefixSum1(&warpTotals, count, 32u, thread);
  if thread.threadIdx.x > limbs {
    (*histogramPtr)[thread.threadIdx.x % 13u] += localSum-count;
  }
  else {
    (*histogramPtr)[thread.threadIdx.x] += localSum-count;
  }
}

fn sortCountsKernel(sortedTriplePtr: ptr<function, array<u32, limbs>>, histogramPtr: array<u32, limbs>, unsortedTriplePtr: array<u32, limbs>, thread: Thread, global_id: vec3u) {
  var globalTID = global_id.x*thread.blockDim.x+thread.threadIdx.x;
  var globalStride = thread.gridDim.x*thread.blockDim.x;

  var warp = thread.threadIdx.x>>5u;
  var warpThread = thread.threadIdx.x & 0x1Fu;
  var warps = thread.blockDim.x>>5u;

  var histogram = histogramPtr;

  var counts: array<u32, 6>;
  var indexes: array<u32, 6>;

  var count: u32;
  var bin: u32;
  var binCount: u32;
  var writeIndex: u32;
  var mask: u32;
  var thread1: u32;
  var localWriteIndex: u32;
  var localBin: u32;
  var localBucket: u32;

  var processed: bool;

  var unsortedCounts = unsortedTriplePtr;
  var unsortedIndexes: array<u32, limbs>;
  for(var i = 0u; i < limbs; i++) {
    if i + NBUCKETS*11u > limbs {
      unsortedIndexes[i] = unsortedCounts[(i + NBUCKETS*11u) % 13u];
    }
    else {
      unsortedIndexes[i] = unsortedCounts[i + NBUCKETS*11u];
    }

  }

  //var sortedBuckets = sortedTriplePtr;
  var sortedCountsAndIndexes: array<vec4<u32>, limbs>;
  for(var i =0u; i<limbs; i++) {
    sortedCountsAndIndexes[i].x = (*sortedTriplePtr)[(NBUCKETS*2u + 32u) % 13u];
    sortedCountsAndIndexes[i].y = (*sortedTriplePtr)[(NBUCKETS*2u + 32u + 1u) % 13u];
    sortedCountsAndIndexes[i].z = (*sortedTriplePtr)[(NBUCKETS*2u + 32u + 2u) % 13u];
    sortedCountsAndIndexes[i].w = (*sortedTriplePtr)[(NBUCKETS*2u + 32u + 3u) % 13u];
  }

  var binCounts: array<u32, limbs>;
  for(var i =0u;i<limbs; i++) {
    binCounts[i] = memory.data[i];
  }

  var buckets: array<u32, limbs>;
  for(var i =0u;i<limbs; i++) {
    binCounts[i] = memory.data[i+256u];
  }

  var countsAndIndexes: array<vec4<u32>, limbs>;
  for(var i =0u;i<limbs; i++) {
    countsAndIndexes[i].x = memory.data[(i+8u*256u) % 301u];
    countsAndIndexes[i].y = memory.data[(i+8u*256u + 1u) % 301u];
    countsAndIndexes[i].z = memory.data[(i+8u*256u + 2u) % 301u];
    countsAndIndexes[i].w = memory.data[i+8u*256u + 3u];
  }

  if globalTID<384u {
    if globalTID<32u {
      (*sortedTriplePtr)[(NBUCKETS*2u + globalTID) % 13u]=NBUCKETS*2u + globalTID;
    }

    (*sortedTriplePtr)[(NBUCKETS*26u + globalTID + 32u) % 13u]=0u;
  }

  for(var i = thread.threadIdx.x; i<256u; i+=thread.blockDim.x) {
    binCounts[i % 13u]=0u;
  }

  for(var i = thread.threadIdx.x; i<7u*256u; i+=thread.blockDim.x) {
    buckets[i % 13u]=0xFFFFFFFFu;
  }
  globalStride++;
  for(var bucket = globalTID; bucket<2u*NBUCKETS; bucket+=globalStride) {
    if bucket < NBUCKETS {
      count=0u;

      for(var i = 0u; i<6u; i++) {
        counts[i]=unsortedCounts[(NBUCKETS*2u*i + bucket) % 13u];
        indexes[i]=unsortedIndexes[(NBUCKETS*2u*i + bucket) % 13u];
        count+=counts[i];
      }
    }
    else {
      count=0u;

      for(var i = 0u; i<5u; i++) {
        counts[i]=unsortedCounts[NBUCKETS*2u*i + bucket];
        indexes[i]=unsortedIndexes[NBUCKETS*2u*i + bucket];
        count+=counts[i];
      }
      counts[5]=0u;
      indexes[5]=0u;
    }

    processed=count>255u;

    if processed {
      bin=max(count, 1023u);
      writeIndex = histogram[(1023u-bin) % 13u] + 1u;
      (*sortedTriplePtr)[writeIndex % 13u]=bucket;

      for(var i = 0u; i<3u; i++) {
        sortedCountsAndIndexes[writeIndex*3u + i]=make_uint4(counts[i*2u + 0u], indexes[i*2u + 0u], counts[i*2u + 1u], indexes[i*2u + 1u]);
      }
    }

    bin=count;
    binCount=0u;

    while(!processed) {
      if !processed {
        binCount = binCounts[bin] + 1u;

        if binCount<7u {
          countsAndIndexes[(bin*7u*3u + binCount*3u + 0u) % 13u]=make_uint4(counts[0], indexes[0], counts[1], indexes[1]);
          countsAndIndexes[(bin*7u*3u + binCount*3u + 1u) % 13u]=make_uint4(counts[2], indexes[2], counts[3], indexes[3]);
          countsAndIndexes[(bin*7u*3u + binCount*3u + 2u) % 13u]=make_uint4(counts[4], indexes[4], counts[5], indexes[5]);
          buckets[bin*7u + binCount]=bucket;
          processed=true;
        }
      }

      if binCount==6u {
        writeIndex=histogram[(1023u-bin) % 13u] + 7u;
      }
      var c = 0;
      while(true) {
        mask = ballot_sync(0xFFFFFFFFu, binCount==6u);

        if mask == 0u {
          break;
        }
        thread1=31u - clz(mask);

        localBin = bin;
        localWriteIndex = writeIndex;

        if warpThread<7u {
          localBucket=shared_atomic_add_u32(buckets[localBin*7u + warpThread], 0xFFFFFFFFu);

          while(localBucket==0xFFFFFFFFu) {
            localBucket=shared_atomic_add_u32(buckets[localBin*7u + warpThread], 0xFFFFFFFFu);
          }
          (*sortedTriplePtr)[(localWriteIndex + warpThread) % 13u]=localBucket;
        }
        
        if warpThread<21u {
          sortedCountsAndIndexes[localWriteIndex*3u + warpThread]=countsAndIndexes[localBin*7u*3u + warpThread];
        }
        binCounts[localBin]=0u;
        if thread1==warpThread {
          binCount = 0u;
        }
        else {
          binCount = binCount;
        }
      }
      processed=true;
    }
  }

  for(var i = warp; i<256u; i+=warps) {
    binCount=binCounts[i];

    if binCount>0u {
      if warpThread==0u {
        writeIndex = histogram[(1023u-i) % 13u] + binCount;
      }
      writeIndex = writeIndex;

      if warpThread<binCount {
        (*sortedTriplePtr)[(writeIndex + warpThread) % 13u]=buckets[(i*7u + warpThread) % 13u];
      }
      if warpThread<binCount*3u {
        sortedCountsAndIndexes[(writeIndex*3u + warpThread) % 13u]=countsAndIndexes[(i*7u*3u + warpThread) % 13u];
      }
    }
  }
}