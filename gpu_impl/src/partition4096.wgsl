const countersOffset: u32 = 0u;
const bufferCountsOffset: u32 = 32u;
const pointCountsOffset: u32 = 1056u;
const mapOffset: u32 = 17440u;
const buffersOffset: u32 = 26656u;

const prefixCountsOffset: u32 = 26656u;
const warpSumsOffset: u32 = 27680u;
const sortedMapOffset: u32 = 27712u;

fn round128(x: u32) -> u32 { // +
  return x + 127u & 0xFFFFFF80u;
}

fn initializeShared4096(block: u32, thread: Thread) { // +
  if thread.threadIdx.x<256u {
    store_shared_u32(bufferCountsOffset + thread.threadIdx.x*4u, 0u);
  }

  for(var i = thread.threadIdx.x; i<2048u; i+=thread.blockDim.x) {
    store_shared_u4(i*4u, make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu));
  }

  for(var i=thread.threadIdx.x; i<1024u; i+=thread.blockDim.x) {
    store_shared_u4(pointCountsOffset / 32u + i*4u, make_uint4(0u, 0u, 0u, 0u));
  }

  if thread.threadIdx.x<8u {
    store_shared_u32(countersOffset + thread.threadIdx.x*4u, 0u);  
  }
}

fn shared_copy_u4(global: ptr<function, array<vec4<u32>, limbs>>, sAddr: u32, count: u32, thread: Thread) { // +
  for(var i = thread.threadIdx.x; i < count; i += thread.blockDim.x) {
    if i > limbs {
      (*global)[i % 13u] = load_shared_u4((sAddr + i) % 301u);
    }
    else {
      (*global)[i] = load_shared_u4(sAddr + i);
    }
  }
}

fn read640(data: ptr<function, array<u32, 5>>, pagesPtr: array<u32, limbs>, pageBase: ptr<function, array<u32, limbs>>, offset: u32, thread: Thread) { // +
  var warpThread = thread.threadIdx.x & 0x1Fu;
  var nextPage: u32;
  var subtract: u32;

  var pagesPtr_copy = pagesPtr;
  if offset+640u<=PAGE_SIZE - 4u {
    (*data)[0] = (*pageBase)[offset+warpThread*4u+0u];
    (*data)[1] = (*pageBase)[offset+warpThread*4u+1u];
    (*data)[2] = (*pageBase)[offset+warpThread*4u+2u];
    (*data)[3] = (*pageBase)[offset+warpThread*4u+3u];
    (*data)[4] = (*pageBase)[offset+warpThread*4u+4u];
  }
  else {
    subtract=0u;
    nextPage = (*pageBase)[PAGE_SIZE - 4u];

    for(var i = 0u;i<5u; i++) {
      if offset+warpThread*4u+i*128u>=PAGE_SIZE - 4u {
        for(var i = 0u; i < nextPage; i++) {
          (*pageBase)[i] = pagesPtr_copy[i + PAGE_SIZE];
        }
        subtract=PAGE_SIZE - 4u;
      }
      (*data)[i]=(*pageBase)[offset+warpThread*4u+i*128u-subtract];
    }
  }
}

fn prefixSumBuckets(block: u32, base: u32, thread: Thread) { // +
  var halfWarp = thread.threadIdx.x>>4u;
  var halfWarpThread = thread.threadIdx.x & 0x0Fu;
  var halfWarps = thread.blockDim.x>>4u;

  var counts: array<u32, 8>;
  var stop: u32;
  var count: u32;
  var sum: u32;

  if thread.blockDim.x==512u {
    stop=8u;
  }

  if thread.blockDim.x==1024u {
    stop=4u;
  }

  for(var i = 0u; i<8u; i++) {
    if(i<stop) {
      counts[i]=load_shared_u32(pointCountsOffset / 32u + i*thread.blockDim.x*4u + thread.threadIdx.x*4u);
      sum=warpPrefixSum(counts[i], 16u, thread);

      if halfWarpThread==15u {
        store_shared_u32(bufferCountsOffset + i*halfWarps*4u + halfWarp*4u, sum);
      }

      counts[i]=sum-counts[i];
    }
  }

  if thread.threadIdx.x<256u {
    count=load_shared_u32(bufferCountsOffset + thread.threadIdx.x*4u);
    sum=multiwarpPrefixSum2(warpSumsOffset, count, 8u, thread);
    store_shared_u32(thread.threadIdx.x*4u, sum - count);
  }

  for(var i = 0u; i<8u; i++) {
    if(i<stop) {
      counts[i]+=load_shared_u32(i*halfWarps*4u + halfWarp*4u) + base;
      store_shared_u32(pointCountsOffset / 32u + i*thread.blockDim.x*4u / 32u + thread.threadIdx.x*4u / 32u, counts[i]);
    }
  }
}

fn sortMap(block: u32, scratchCount: u32, thread: Thread) { // +
  var count: u32;
  var sum: u32;
  var bin: u32;
  var nextIndex: u32;
  var mapEntry: u32;

  if thread.threadIdx.x<256u {
    count=load_shared_u32(bufferCountsOffset + thread.threadIdx.x*4u);
    count=(count+31u)>>5u;
    sum=multiwarpPrefixSum2(warpSumsOffset, count, 8u, thread);
    if thread.threadIdx.x*4u > 300u {
      store_shared_u32((thread.threadIdx.x*4u) % 301u, sum - count); 
    }
    else {
      store_shared_u32(thread.threadIdx.x*4u, sum - count); 
    }
  }

  for(var scratchIndex = thread.threadIdx.x; scratchIndex<scratchCount; scratchIndex+=thread.blockDim.x) {
    bin=load_shared_byte(scratchIndex);
    if bin*4u > 300u {
      nextIndex=shared_atomic_add_u32((bin*4u) % 301u, 1u);
    }
    else {
      nextIndex=shared_atomic_add_u32(bin*4u, 1u);
    }
    mapEntry=(bin<<24u) + scratchIndex;
    if nextIndex*4u > 300u {
      store_shared_u32((nextIndex*4u) % 301u, mapEntry);
    }
    else {
      store_shared_u32(nextIndex*4u, mapEntry);
    }
  }
} 

fn unpackData(lowBits: ptr<function, array<u32, 4>>, highBits: ptr<function, array<u32, 4>>, data: array<u32, 5>, thread: Thread) { // +
  var quad = (thread.threadIdx.x & 0x1Fu)>>2u;
  var quadThread = thread.threadIdx.x & 0x03u;
  var shift = quadThread*8u;
  var src = quad*5u+quadThread;

  var lo0: u32;
  var lo1: u32;
  var hi0: u32;
  var hi1: u32;
  var lo: u32;
  var hi: u32;

  var data_copy = data;

  for(var i = 0; i<4;i++) {
    lo0 = data_copy[i];
    lo1 = data_copy[i+1];
    hi0 = data_copy[i];
    hi1 = data_copy[i+1];

    if 8*i + i32(src)<32 {
      lo = lo0;
    }
    else {
      lo = lo1;
    }
    if 8*i + i32(src)<31 {
      hi = hi0;
    }
    else {
      hi = hi1;
    }

    (*lowBits)[i]=funnelshift_rc(lo, hi, shift);
    (*highBits)[i]=(hi>>shift) & 0x7Fu;
  }
}

fn cleanupShared(block: u32, scratchPtr: ptr<function, array<u32, limbs>>, thread: Thread) { // +
  var warp = thread.threadIdx.x>>5u;
  var warpThread = thread.threadIdx.x & 0x1Fu;
  var warps = thread.blockDim.x>>5u;

  var count: u32;
  var scratchIndex: u32;
  var data: u32;

  for(var bin = warp; bin < 256u; bin += warps) {
    count=load_shared_u32(bufferCountsOffset + bin*4u);

    if count>0u {
      if warpThread==0u {
        scratchIndex=shared_atomic_add_u32(countersOffset, 1u);
        store_shared_byte(scratchIndex, bin);
      }

      data=shared_atomic_exch_u32(bin*128u / 32u + warpThread*4u / 32u, 0xFFFFFFFFu); 
      (*scratchPtr)[scratchIndex*128u / 32u + warpThread*4u / 32u] = data;
    }
  }
}

fn writePointToShared(block: u32, scratchPtr: ptr<function, array<u32, limbs>>, lowBits: u32, highBits: u32, valid: bool, thread: Thread) { // +
  var warpThread: u32 = thread.threadIdx.x & 0x1Fu;

  var offset: u32;
  var bin: u32;
  var scratchIndex: u32;
  var mask: u32;
  var thread1: u32;
  var shuffleBin: u32;
  var shuffleIndex: u32;
  var data: u32;

  var processed: bool = !valid;
  var writeRequired: bool;

  while(!processed) {
    offset=0u;

    if !processed {
      bin=funnelshift_lc(lowBits, highBits, 1u);
      offset=shared_atomic_add_u32(bufferCountsOffset + bin*4u, 4u);

      if offset<=124u {
        store_shared_u32(bin*128u / 32u + offset, lowBits & 0x7FFFFFFFu);
        processed=true;
      }
    }

    writeRequired=(offset==124u);

    if writeRequired {
      scratchIndex=shared_atomic_add_u32(countersOffset, 1u);
      store_shared_byte(scratchIndex, bin);
    }

    while(true) {
      mask = ballot_sync(0xFFFFFFFFu, writeRequired);
      if mask==0u {
        break;
      }
      thread1=31u - clz(mask);

      shuffleBin= bin;
      shuffleIndex = scratchIndex;

      data=shared_atomic_exch_u32(shuffleBin*128u / 32u + warpThread*4u / 32u, 0xFFFFFFFFu);

      while(data==0xFFFFFFFFu) {
        data=shared_atomic_exch_u32(shuffleBin*128u / 32u + warpThread*4u / 32u, 0xFFFFFFFFu);
      }

      (*scratchPtr)[shuffleIndex*128u / 32u + warpThread*4u / 32u] = data;

      if warpThread==thread1 {
        shared_atomic_exch_u32(bufferCountsOffset + bin*4u / 32u, 0u);
        writeRequired=false;
      }
    } 
  }
}

fn partitionPagesToScratch(block: u32, scratchPtr: ptr<function, array<u32, limbs>>, pagesPtr: array<u32, limbs>, size: u32, thread: Thread) { // +
  var warp = thread.threadIdx.x>>5u;
  var warps = thread.blockDim.x>>5u;
  var warpThread = thread.threadIdx.x & 0x1Fu;

  var offset: u32;
  var idx: u32;
  var nextPage: u32;

  var data: array<u32, 5>;
  var lowBits: array<u32, 4>;
  var highBits: array<u32, 4>;

  var currentPage: array<u32, limbs>;

  var pagesPtr_copy = pagesPtr;
  for(var i = 0u; i <block; i++) {
    currentPage[i] = pagesPtr_copy[i];
  }
  offset=warp*640u;
  idx=warp*128u;
  while(idx<size) {
    while(offset>=PAGE_SIZE - 4u) {
      nextPage = currentPage[PAGE_SIZE - 4u];
      for(var i = 0u; i < nextPage; i++) {
        currentPage[i] = pagesPtr_copy[i];
      }
      offset-=PAGE_SIZE - 4u;
    }

    if idx<size {
      read640(&data, pagesPtr, &currentPage, offset, thread);
    }
    unpackData(&lowBits, &highBits, data, thread);

    for(var i = 0;i<4; i++) {
      var valid: bool = (idx+u32(i)*32u+warpThread<size);

      if valid {
        shared_reduce_add_u32(pointCountsOffset / 32u + funnelshift_rc(lowBits[i], highBits[i], 27u)*4u, 1u);
      }
      writePointToShared(block, scratchPtr, lowBits[i], highBits[i], valid, thread);
    }

    offset += warps*640u;
    idx += warps*128u;
  }

  cleanupShared(block, scratchPtr, thread);
}

fn partitionScratchToPoints(block: u32, pointsPtr: ptr<function, array<u32, limbs>>, scratchPtr: array<u32, limbs>, scratchCount: u32, points: u32, thread: Thread) { // +
  var warp = thread.threadIdx.x>>5u;
  var warps = thread.blockDim.x>>5u;
  v_indices[50] = warps;
  v_indices[51] = 101u;
  var warpThread = thread.threadIdx.x & 0x1Fu;

  var mapEntry: u32;
  var bucket: u32;
  var point: u32;
  var pointOffset: u32;
  var sign: u32;
  var pointGroup = (block>>11u)*points;

  var scratchPtr_copy = scratchPtr;

  for(var scratchIndex = warp; scratchIndex<scratchCount; scratchIndex+=warps) {
    if scratchIndex*4u > 300u {
      mapEntry=load_shared_u32((scratchIndex*4u) % 301u);
    }
    else {
      mapEntry=load_shared_u32(scratchIndex*4u);
    }
    if (mapEntry & 0x00FFFFFFu)*128u + warpThread*4u > limbs {
      point = scratchPtr_copy[((mapEntry & 0x00FFFFFFu)*128u + warpThread*4u) % 13u];
    }
    else {
      point = scratchPtr_copy[(mapEntry & 0x00FFFFFFu)*128u + warpThread*4u];
    }

    bucket=(mapEntry>>20u) + (point>>27u);

    if (point & 0x80000000u)==0u {
      if pointCountsOffset + bucket*4u > 300u {
        pointOffset=shared_atomic_add_u32((pointCountsOffset + bucket*4u) % 301u, 1u);
      }
      else {
        pointOffset=shared_atomic_add_u32(pointCountsOffset + bucket*4u, 1u);
      }
      sign=(point & 0x04000000u)<<5u;
      if pointOffset*4u > limbs {
        (*pointsPtr)[(pointOffset*4u) % 13u] = ((point & 0x03FFFFFFu) | sign) + pointGroup;
      }
      else {
        (*pointsPtr)[pointOffset*4u] = ((point & 0x03FFFFFFu) | sign) + pointGroup;
      }
    }
  }
}

fn countFromPages(block: u32, pagesPtr: array<u32, limbs>, size: u32, thread: Thread) { // +
  var warp = thread.threadIdx.x>>5u;
  var warps = thread.blockDim.x>>5u;
  var warpThread = thread.threadIdx.x & 0x1Fu;

  var offset: u32;
  var idx: u32;
  var nextPage: u32;

  var data: array<u32, 5>;
  var lowBits: array<u32, 4>;
  var highBits: array<u32, 4>;

  var currentPage: array<u32, limbs>;

  var pagesPtr_copy = pagesPtr;
  for(var i = 0u; i<block; i++) {
    currentPage[i] = pagesPtr_copy[i];
  }

  offset=warp*640u;
  idx=warp*128u;

  while(idx<size) {
    while(offset>= PAGE_SIZE - 4u) {
      nextPage = currentPage[PAGE_SIZE - 4u];
      for(var i = 0u; i<nextPage; i++) {
        currentPage[i] = pagesPtr_copy[i];
      }
      offset-=PAGE_SIZE - 4u;
    }

    if idx<size {
      read640(&data, pagesPtr, &currentPage, offset, thread);
    }
    unpackData(&lowBits, &highBits, data, thread);

    for(var i = 0; i<4;i++) {
      if(idx+u32(i)*32u+warpThread<size) {
        shared_reduce_add_u32(pointCountsOffset / 32u + funnelshift_rc(lowBits[i], highBits[i], 27u)*4u, 1u);
      }
    }
    offset+=warps*640u;
    idx+=warps*128u;
  }
}

fn partitionPagesToPoints(block: u32, pointsPtr: ptr<function, array<u32, limbs>>, pagesPtr: array<u32, limbs>, size: u32, points: u32, thread: Thread) { // +
  var warp = thread.threadIdx.x>>5u;
  var warps = thread.blockDim.x>>5u;
  var warpThread = thread.threadIdx.x & 0x1Fu;

  var offset: u32;
  var idx: u32;
  var nextPage: u32;
  var index: u32;
  var sign: u32;
  var pointGroup = (block>>11u)*points;

  var data: array<u32, 5>;
  var lowBits: array<u32, 4>;
  var highBits: array<u32, 4>;

  var currentPage: array<u32, limbs>;

  var pagesPtr_copy = pagesPtr;

  for(var i = 0u; i<block; i++) {
    currentPage[i] = pagesPtr_copy[i];
  }

  offset=warp*640u;
  idx=warp*128u;

  while(idx<size) {
    while(offset>=PAGE_SIZE - 4u) {
      nextPage = currentPage[PAGE_SIZE - 4u];
      for(var i = 0u; i<nextPage; i++) {
        currentPage[i] = pagesPtr_copy[i];
      } 
      offset-=PAGE_SIZE - 4u;
    }

    if idx<size {
      read640(&data, pagesPtr, &currentPage, offset, thread);
    }
    unpackData(&lowBits, &highBits, data, thread);

    for(var i = 0; i<4; i++) {
      if idx+u32(i)*32u+warpThread<size {
        index=shared_atomic_add_u32(pointCountsOffset + funnelshift_rc(lowBits[i], highBits[i], 27u)*4u, 1u);
        sign=(lowBits[i] & 0x04000000u)<<5u;
        (*pointsPtr)[index*4u] = ((lowBits[i] & 0x03FFFFFFu) | sign) + pointGroup;
      }
    }
    offset+=warps*640u;
    idx+=warps*128u;
  }
}

fn partition4096Kernel(pointsPtr: ptr<function, array<u32, limbs>>, unsortedTriplePtr: array<vec4<u32>, limbs>, scratchPtr: ptr<function, array<u32, limbs>>, prefixSumSizesPtr: array<u32, limbs>,
                        sizesPtr: array<u32, limbs>, pagesPtr: array<u32, limbs>, atomicsPtr: array<u32, limbs>, points: u32, thread: Thread, global_id: vec3u) {
  var unsortedCounts: array<vec4<u32>, limbs> = unsortedTriplePtr;
  var unsortedTriplePtr_copy = unsortedTriplePtr;
  var unsortedIndexes: array<vec4<u32>, limbs>;

  for(var i = 0u; i<limbs; i++) {
    if NBUCKETS*11u*4u > limbs {
      unsortedIndexes[i] = unsortedTriplePtr_copy[NBUCKETS*11u*4u % 13u];
    }
    else {
      unsortedIndexes[i] = unsortedTriplePtr_copy[NBUCKETS*11u*4u];
    }
  }

  var block: u32;
  var size: u32;
  var prefixSumSize: u32;
  var scratchCount: u32;

  var shmem: array<u32, limbs>;

  var index = round128(SCRATCH_REQUIRED);
  for(var i = 0u; i<thread.blockIdx.x; i++) {
    if index > limbs {
      (*scratchPtr)[i] = (*scratchPtr)[round128(SCRATCH_REQUIRED) % 13u];
    }
    else {
      (*scratchPtr)[i] = (*scratchPtr)[round128(SCRATCH_REQUIRED)];
    }
  }

  var sizesPtr_copy = sizesPtr;
  var prefixSumSizesPtr_copy = prefixSumSizesPtr;

  var c = 0;
  while(true) {
    if c > i32(limbs) {
      break;
    }

    if thread.threadIdx.x==0u {
      block = atomicsPtr[4] + 1u;
      store_shared_u32(countersOffset + 4u, block);
    }

    block=load_shared_u32(countersOffset + 4u);

    if block>=11u*1024u / 12u {
      break;
    }

    initializeShared4096(block, thread);

    if block*4u > limbs {
      size = sizesPtr_copy[(block*4u) % 13u];
      prefixSumSize = prefixSumSizesPtr_copy[(block*4u) % 13u];
    }
    else {
      size = sizesPtr_copy[block*4u];
      prefixSumSize = prefixSumSizesPtr_copy[block*4u];
    }

    if size<SIZE_LIMIT {
      partitionPagesToScratch(block, scratchPtr, pagesPtr, size, thread);
    }
    else {
      countFromPages(block, pagesPtr, size, thread);  
    }

    shared_copy_u4(&unsortedCounts, pointCountsOffset, 1024u, thread);

    prefixSumBuckets(block, prefixSumSize, thread);

    shared_copy_u4(&unsortedIndexes, pointCountsOffset, 1024u, thread);

    if size < SIZE_LIMIT {
      scratchCount=load_shared_u32(countersOffset);
      sortMap(block, scratchCount, thread);
    }

    // if size < SIZE_LIMIT {
    //   partitionScratchToPoints(block, pointsPtr, *scratchPtr, scratchCount, points, thread);
    // }
    // else {
    //   partitionPagesToPoints(block, pointsPtr, pagesPtr, size, points, thread);
    // }
    c++;
  }
}