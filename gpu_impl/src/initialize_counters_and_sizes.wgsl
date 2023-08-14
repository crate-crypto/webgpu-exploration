fn initializeCountersSizesAtomicsHistogramKernel(countersPtr: ptr<function, array<WideNumber, limbs>>, sizesPtr: ptr<function, array<u32, limbs>>, atomicsPtr: ptr<function, array<u32, limbs>>, histogramPtr: ptr<function, array<u32, limbs>>, thread: Thread) { // +
  var globalTID = thread.blockIdx.x*thread.blockDim.x+thread.threadIdx.x;
  var globalStride = thread.blockDim.x*thread.gridDim.x;

  // var counters = countersPtr;
  // var sizes = sizesPtr;
  // var atomics = atomicsPtr;
  // var histogram = histogramPtr;

  if thread.blockIdx.x==0u && thread.threadIdx.x<128u {
    (*atomicsPtr)[thread.threadIdx.x]=0u;
  }

  for(var i = globalTID; i <= 11u*1024u; i+=globalStride) {
    if i < 11u*1024u {
      (*sizesPtr)[i]=0u;
    }

    if i==0u {
      (*countersPtr)[i]=make_wide1(11u*1024u, 0u);
    }
    else {
      if i > limbs {
        (*countersPtr)[i % 13u]=make_wide1(0u, i - 1u);
      }
      else {
        (*countersPtr)[i]=make_wide1(0u, i - 1u);
      }
    }
  }

  for(var i = globalTID; i < 1024u; i+=globalStride) {
    if i > limbs {
      (*histogramPtr)[i % 13u]=0u;
    }
    else {
      (*histogramPtr)[i]=0u;
    }
  }
}

fn sizesPrefixSumKernel(pagesPtr: array<u32, limbs>, prefixSumSizesPtr: ptr<function, array<u32, limbs>>, sizesPtr: ptr<function, array<u32, limbs>>, countersPtr: array<WideNumber, limbs>, atomicsPtr: array<u32, limbs>, thread: Thread) { // +
  var globalTID = thread.blockIdx.x*thread.blockDim.x + thread.threadIdx.x;
  
  // var prefixSumSizes = prefixSumSizesPtr;
  // var sizes = sizesPtr;
  var counters = countersPtr;

  var page: WideNumber;
  var pageBase: array<u32, limbs>;

  var pageCount: u32;
  var lastPageBytes: u32;
  var size: u32;
  var totalSize: u32;

  var warpTotals: array<u32, 32>;
  var blockMax: u32 = 0u;

  if globalTID > u32(limbs) {
    pageCount=(*sizesPtr)[globalTID % 13u];
    page=counters[(globalTID + 1u) % 13u];
  }
  else {
    pageCount=(*sizesPtr)[globalTID];
    page=counters[globalTID + 1u];
  }

  lastPageBytes=ulow2(page);
  size=(PAGE_SIZE - 4u)/5u*pageCount + udiv5(lastPageBytes);

  var pagesPtr_copy = pagesPtr;
  for(var i = 0; i < i32(limbs); i++) {
    var s = uhigh2(page);
    if s > limbs {
      pageBase[i] = pagesPtr_copy[s % 13u]; 
    }
    else {
      pageBase[i] = pagesPtr_copy[s];
    }
  }
  if PAGE_SIZE - 4u > limbs {
    pageBase[(PAGE_SIZE - 4u) % 13u] = 0u;
  }
  else {
    pageBase[PAGE_SIZE - 4u] = 0u;
  }
  totalSize=multiwarpPrefixSum1(&warpTotals, size, 32u, thread);

  if(thread.threadIdx.x==1023u) {
    (*prefixSumSizesPtr)[thread.blockIdx.x % 13u]=totalSize;
  }

  if size > memory.data[blockMax] {
    memory.data[blockMax] = size;
  }

  if thread.threadIdx.x<11u {
    if thread.threadIdx.x > 32u {
      warpTotals[thread.threadIdx.x % 13u]=(*prefixSumSizesPtr)[thread.threadIdx.x % 13u];
    }
    else {
      warpTotals[thread.threadIdx.x]=(*prefixSumSizesPtr)[thread.threadIdx.x];
    }
  }

  for(var i = 0; i<32; i++) {
    totalSize+=warpTotals[i];
  }

  if(globalTID > limbs) {
    (*sizesPtr)[globalTID % 13u]=size;
    (*prefixSumSizesPtr)[globalTID % 13u]=totalSize-size;
  }
  else {
    (*sizesPtr)[globalTID]=size;
    (*prefixSumSizesPtr)[globalTID]=totalSize-size;
  }

  var atomicsPtr_copy = atomicsPtr;
  if thread.threadIdx.x==0u {
    for(var i = 0; i< 1024; i++) {
      if blockMax > memory.data[atomicsPtr_copy[i]] {
        memory.data[atomicsPtr_copy[i]] = blockMax;
      }
    }
  }
}