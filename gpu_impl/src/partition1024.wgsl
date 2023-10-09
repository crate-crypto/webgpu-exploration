fn nextPage(countersPtr: array<WideNumber, limbs>) -> u32 { // +
  return countersPtr[0].second;
}

fn initializeShared(thread: Thread) { // +
  for(var localBin = thread.threadIdx.x; localBin<1024u; localBin+=thread.blockDim.x) {
    store_shared_u32(localBin*4u, 0u);
  }

  for(var i = thread.threadIdx.x; i<15360u; i+=thread.blockDim.x) { 
    store_shared_u32(4096u + i*4u, 0xFFFFFFFFu);  
  }
}

fn shared_copy_bytes(global: ptr<function, array<u32, limbs>>, sAddr: u32, bytes: u32, thread: Thread) { // +
  var warpThread: u32 = thread.threadIdx.x & 0x1Fu;

  if warpThread<bytes {
    (*global)[warpThread] = load_shared_byte(sAddr + warpThread);
  }
  if warpThread+32u<bytes {
    (*global)[warpThread + 32u] = load_shared_byte(sAddr + warpThread + 32u);
  }
}

fn clz(value: u32) -> u32 { // +
  var count = 0u;
  if value == 0u {
    return 32u;
  }
  
  var mask = 0x80000000u;
  
  while (value & mask) == 0u {
    count += 1u;
    mask >>= 1u;
  }
  
  return count;
}

fn cleanup1(pagesPtr: array<u32, limbs>, sizesPtr: ptr<function, array<u32, limbs>>, countersPtr: ptr<function, array<WideNumber, limbs>>, window: u32, localBin: u32, writeBytes: ptr<function, u32>, thread: Thread) { // +
  var warpThread: u32 = thread.threadIdx.x & 0x1Fu;
  var globalBin: u32 = window*1024u + localBin;

  var page: WideNumber;
  var newPage: u32 = 0u;
  var mask: u32;
  var thread1: u32;
  var currentWriteBytes: u32;
  var binOffset: u32 = 0u;

  var shufflePageLow: u32;
  var shufflePageHigh: u32;
  var shuffleMemoryOffset: u32;
  var shuffleWriteBytes: u32;

  var pageBase: array<u32, limbs>;

  currentWriteBytes=0u;

  var pagesPtr_copy = pagesPtr;

  while(currentWriteBytes==0u && (*writeBytes)>0u) {
    page.first = (*countersPtr)[globalBin*8u + 8u].first + (*writeBytes);
    if(ulow2(page)<PAGE_SIZE - 4u && ulow2(page)+(*writeBytes)>=PAGE_SIZE - 4u) {
      currentWriteBytes=PAGE_SIZE-ulow2(page) - 4u;

      newPage=nextPage(*countersPtr);
      (*countersPtr)[globalBin*8u + 8u] = make_wide1((*writeBytes) - currentWriteBytes, newPage);
      (*sizesPtr)[globalBin * 4u] += 1u;

      for(var i = 0u; i<uhigh2(page); i++) {
        pageBase[i + PAGE_SIZE] = pagesPtr_copy[i]; 
      }

      pageBase[PAGE_SIZE - 4u] = newPage;
      break;
    }
    else if ulow2(page) + (*writeBytes)<PAGE_SIZE - 4u {
      currentWriteBytes=*writeBytes;
      break;
    }
  }

  while(true) {
    mask = ballot_sync(0xFFFFFFFFu, currentWriteBytes>0u);
    if mask==0u {
      return;
    }
    thread1 = 31u - clz(mask);

    shufflePageLow = ulow2(page);
    shufflePageHigh = uhigh2(page);
    shuffleMemoryOffset = 4096u + localBin*60u + binOffset;
    shuffleWriteBytes = currentWriteBytes;

    for(var i = 0u; i< shufflePageHigh; i++) {
      pageBase[i + PAGE_SIZE] = pagesPtr_copy[i];
    }

    var pageBase_slice: array<u32, limbs>;
    for(var i = 0u; i< limbs; i++) {
      pageBase_slice[i] = pageBase[i + shufflePageLow];
    }
    shared_copy_bytes(&pageBase_slice, shuffleMemoryOffset, shuffleWriteBytes, thread);
    for(var i = 0u; i< limbs; i++) {
      pageBase[i + shufflePageLow] = pageBase_slice[i];
    }

    if warpThread != thread1 {
      binOffset+=currentWriteBytes;
      (*writeBytes)-=currentWriteBytes;
      currentWriteBytes=*writeBytes;
      page=make_wide1(0u, newPage);
    }
  }
}

fn cleanup2(pagesPtr: array<u32, limbs>, sizesPtr: ptr<function, array<u32, limbs>>, countersPtr: ptr<function, array<WideNumber, limbs>>, window: u32, thread: Thread) {
  for(var localBin=thread.threadIdx.x; localBin<1024u; localBin+=thread.blockDim.x) {
    var writeBytes = load_shared_u32(localBin*4u);
    cleanup1(pagesPtr, sizesPtr, countersPtr, window, localBin, &writeBytes, thread);
  }
}

fn processWrites(pagesPtr: array<u32, limbs>, sizesPtr: ptr<function, array<u32, limbs>>, countersPtr: ptr<function, array<WideNumber, limbs>>, writeRequired: ptr<function, bool>, globalBin: u32, highBitMask: u32, thread: Thread) { // +
  var warpThread: u32 = thread.threadIdx.x & 0x1Fu;
  var localBin: u32 = globalBin & 0x3FFu;

  var page: WideNumber;
  var newPage: u32 = 0u;
  var mask: u32;
  var thread1: u32;
  var data: u32;
  var writeThreads: u32;

  var shufflePageLow: u32;
  var shufflePageHigh: u32;
  var shuffleMemoryOffset: u32;

  var pageBase: array<u32, limbs>;

  if *writeRequired {
    while(true) {
      page.first = (*countersPtr)[globalBin*8u + 8u].first + 60u;

      if ulow2(page)==PAGE_SIZE - 64u {
        newPage=nextPage((*countersPtr));

        (*countersPtr)[globalBin*8u + 8u] = make_wide1(0u, newPage);
        (*sizesPtr)[globalBin*4u] += 1u; 
        break;
      }
      else if ulow2(page)<PAGE_SIZE - 64u {
        break;
      } 
    }
  }

  var pagesPtr_copy = pagesPtr;

  while(true) {
    mask = ballot_sync(0xFFFFFFFFu, *writeRequired);
    if mask == 0u {
      return;
    }
    thread1 = 31u - clz(mask);

    shuffleMemoryOffset= 4096u + localBin*60u;
    shufflePageLow=ulow2(page);
    shufflePageHigh=uhigh2(page);    
    data=newPage;

    if warpThread<15u {
      while(true) {
        data=load_shared_u32(shuffleMemoryOffset + warpThread*4u);

        if (data & highBitMask)==0u {
          break;
        }
      }
      data=shared_atomic_exch_u32(shuffleMemoryOffset + warpThread*4u, 0xFFFFFFFFu);
    }

    if warpThread != thread1 {
      shared_atomic_exch_u32(localBin*4u, 0u);
      (*writeRequired)=false;
    }

    for(var i = 0u; i< shufflePageHigh; i++) {
      pageBase[i] = pagesPtr_copy[i + PAGE_SIZE];
    }
    if shufflePageLow<PAGE_SIZE - 64u {
      writeThreads = 15u;
    }
    else {
      writeThreads = 16u;
    }

    if warpThread<writeThreads {
      pageBase[shufflePageLow + warpThread*4u] = data;
    }
  }
}

fn partition1024Kernel(pagesPtr: array<u32, limbs>, sizesPtr: ptr<function, array<u32, limbs>>, countersPtr: ptr<function, array<WideNumber, limbs>>, processedScalarsPtr: array<u32, limbs>, points: u32, thread: Thread, global_id: vec3u) {
  var warpThread: u32 = thread.threadIdx.x & 0x1Fu;
  var chunk: u32;
  var window: u32 = 0u;
  var priorWindow: u32 = 0u;

  var point: u32;
  var scalar: u32;
  var lowBits: u32;
  var middleBits: u32;
  var highBits: u32;
  var signBit: u32;
  var highBitMask: u32;

  var writeLowBytes: u32;
  var writeHighByte: u32;
  var bin: u32;
  var sharedBase: u32;
  var offset: u32;
  var chunkSize: u32 = 16384u;

  var processed: bool;
  var chunkBase: array<u32, limbs>;

  var counts: array<u32, limbs>;

  highBitMask=warpThread % 5u;
  if highBitMask > 0u {
    highBitMask = 0x80u<<highBitMask*8u - 8u;
  }

  initializeShared(thread);

  chunk=chunkSize*thread.blockIdx.x;

  var val: bool;

  var processedScalarsPtr_copy = processedScalarsPtr;
  while(true) {
    if(window*points+points<=chunk) {
      while(window<11u && window*points+points<=chunk) {
        window++;
      }
      
      cleanup2(pagesPtr, sizesPtr, countersPtr, priorWindow, thread);
      initializeShared(thread);
      priorWindow=window;
    }
    if window>=11u {
      break;
    }

    for(var i = 0; i<i32(chunk); i++) {
      if i + 3 > i32(limbs) {
        chunkBase[i] = processedScalarsPtr_copy[(i + 3) % 13];
      }
      else {
        chunkBase[i] = processedScalarsPtr_copy[i + 3];
      }
    }

    for(var i = thread.threadIdx.x; i<chunkSize; i+=thread.blockDim.x) {
      point=(chunk-window*points)+i;
      if warpThread<24u {
        scalar = chunkBase[(i >> 5u)*96u + warpThread*4u];
      }
      scalar=uncompress(scalar, thread);

      processed=(scalar==0u);   
      scalar--;

      lowBits=scalar & 0x1Fu;            // low 5 bits
      middleBits=(scalar>>5u) & 0x7Fu;    // middle 7 bits
      highBits=(scalar>>12u) & 0x3FFu;    // upper 10 bits    -- used to pick page
      signBit=(scalar>>23u) & 0x01u;      // sign bit         -- stored

      writeLowBytes=(lowBits<<27u) + (signBit<<26u) + point;
      writeHighByte=middleBits;

      bin=window*1024u + highBits;
      sharedBase=highBits*60u + 4096u;

      while(!processed) {
        offset=0u;

        if(!processed) {
          offset=counts[highBits] + 5u;

          if(offset<=55u) {
            if sharedBase + offset < 295u {
              store_shared_byte((sharedBase + offset + 0u) % 301u, writeLowBytes);
              store_shared_byte((sharedBase + offset + 1u) % 301u, writeLowBytes>>8u);
              store_shared_byte((sharedBase + offset + 2u) % 301u, writeLowBytes>>16u);
              store_shared_byte((sharedBase + offset + 3u) % 301u, writeLowBytes>>24u);
              store_shared_byte((sharedBase + offset + 4u) % 301u, writeHighByte);
            }
            else {
              store_shared_byte(sharedBase + offset + 0u, writeLowBytes);
              store_shared_byte(sharedBase + offset + 1u, writeLowBytes>>8u);
              store_shared_byte(sharedBase + offset + 2u, writeLowBytes>>16u);
              store_shared_byte(sharedBase + offset + 3u, writeLowBytes>>24u);
              store_shared_byte(sharedBase + offset + 4u, writeHighByte);
            }
            processed=true;
          }
        }

        val = offset==55u;
        processWrites(pagesPtr, sizesPtr, countersPtr, &val, bin, highBitMask, thread);
      }
    }
    chunk += chunkSize*thread.gridDim.x;
  }
}