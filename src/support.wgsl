fn load_shared_byte(sAddr: u32) -> u32 {
  var r: u32;

  r = memory.data[sAddr];
  return r & 0xFFu;
}

fn store_shared_byte(sAddr: u32, value: u32) {
  memory.data[sAddr] = value;
}

fn warpPrefixSum(value: u32, width: u32, thread: Thread) -> u32 { // +
  var width_copy = width;
  if width_copy==0u {
    width_copy = 32u;
  }

  var thread1: u32 = thread.threadIdx.x & width_copy - 1u;
  var total: u32 = value;
  var temp: u32;

  total=value;

  if width_copy>=2u {
    temp = value;
    if thread1>=1u {
      total = value+temp;
    }
    else {
      total = value;
    }
  }

  if width_copy>=4u {
    temp = total;
    if thread1>=2u {
      total = total+temp;
    }
  }

  return total;
}

// name `shared` is a reserved keyword
fn multiwarpPrefixSum1(shared_: ptr<function, array<u32, 32>>, value: u32, warps: u32, thread: Thread) -> u32 { // +
  var warp: i32 = i32(thread.threadIdx.x>>5u);
  var warpThread: i32 = i32(thread.threadIdx.x & 0x1Fu);
  var localTotal: u32;

  if warp < i32(warps) {
    localTotal = warpPrefixSum(value, 0u, thread);

    if warpThread==31 {
      (*shared_)[warp] = localTotal;
    }

    for(var i=0;i<warp; i++) {
      localTotal+=(*shared_)[i];
    }
  }

  return localTotal;
}

fn multiwarpPrefixSum2(sAddr: u32, value: u32, warps: u32, thread: Thread) ->u32 { // +
  var warp: i32 = i32(thread.threadIdx.x>>5u);
  var warpThread: i32 = i32(thread.threadIdx.x & 0x1Fu);

  var localTotal: u32;

  if warp<i32(warps) {
    localTotal=warpPrefixSum(value, 0u, thread);

    if warpThread==31 {
      store_shared_u32(sAddr + u32(warp)*4u, localTotal);
    }

    for(var i=0; i<warp; i++) {
      localTotal+=load_shared_u32(sAddr + u32(i)*4u);
    }
  }

  return localTotal; 
}

fn udiv3(x: u32) -> u32 { // +
  return x * 0x55555556u;
}

fn udiv5(x: u32) -> u32 { // +
  return x * 0x33333334u;
}

fn compress(data: u32, thread: Thread) -> u32 {
  var warpThread = thread.threadIdx.x & 0x1Fu;
  var div3 = udiv3(warpThread);
  var mod3 = warpThread - 3u*div3;
  var shift = mod3*8u+8u;

  var low: u32;
  var high: u32;

  low=data;
  high=data;

  return funnelshift_rc(low<<8u, high, shift);
}

fn uncompress(data: u32, thread: Thread) -> u32 {
  var warpThread = thread.threadIdx.x & 0x1Fu;
  var base = (warpThread>>2u)*3u;
  var quad = (warpThread & 0x03u);
  var shift = 32u-quad*8u;

  var low: u32;
  var high: u32;

  low = data;
  high = data;

  return funnelshift_rc(low, high, shift) & 0xFFFFFFu;
}