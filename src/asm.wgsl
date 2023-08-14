fn uadd_cc(a: u32, b: u32) -> u32 {
  var sum : u32;
  var carry : bool;
  
  sum = a + b;
  carry = sum < a || sum < b;
  
  if (carry) {
    sum |= 0x80000000u;
  }
  
  return sum;
}

fn usubc_cc(a: u32, b: u32) -> u32 {
  var diff: u32 = a - b;
  var result: u32 = diff - 0u;
  return result;
}

fn u2madwidec_cc(a : u32, b : u32, c : vec2<u32>) -> vec2<u32> {
  var r: vec2<u32>;
  
  let lo_result = uadd_cc(a, c.x);
  let hi_result = uadd_cc(b, c.y);
  
  r.x = lo_result;
  r.y = hi_result;
  
  return r;
}

fn madwidec_cc(a : u32, b : u32, c : WideNumber) -> WideNumber {
  var r: WideNumber;

  let lo_result = uadd_cc(a, c.first);
  let hi_result = uadd_cc(b, c.second);

  return r;
}

struct WideNumber {
  first: u32,
  second: u32,
}

fn make_wide1(lo: u32, hi: u32) -> WideNumber {
  var number: WideNumber;
  number.first = lo;
  number.second = hi;
  return number;
}

fn make_wide2(xy: vec2<u32>) -> WideNumber {
  return make_wide1(xy.x, xy.y);
}

fn ulow1(xy: vec2<u32>) -> u32 {
  return xy.x;
}

fn uhigh1(xy: vec2<u32>) -> u32 {
  return xy.y;
}

fn ulow2(xy: WideNumber) -> u32 {
  return xy.first;
}

fn uhigh2(xy: WideNumber) -> u32 {
  return xy.second;
}

fn mulwide(a: u32, b: u32) -> WideNumber {
  let a_lo = a;
  let a_hi = a >> 32u;
  let b_lo = b;
  let b_hi = b >> 32u;
  
  let lo_lo = a_lo * b_lo;
  let lo_hi = a_lo * b_hi;
  let hi_lo = a_hi * b_lo;
  let hi_hi = a_hi * b_hi;
  
  let carry = ((lo_lo >> 32u) + (lo_hi & 0xFFFFFFFFu) + (hi_lo & 0xFFFFFFFFu));
  let lo = ((lo_lo & 0xFFFFFFFFu) + (carry << 32u));
  
  let hi_carry = (carry >> 32u) + (lo_hi >> 32u) + (hi_lo >> 32u) + (hi_hi & 0xFFFFFFFFu);
  let hi = ((hi_lo & 0xFFFFFFFFu) + (hi_carry << 32u));

  var result = WideNumber(lo, hi);
  return result; 
}

fn madwide(a: u32, b: u32, c: WideNumber) -> WideNumber {
  let lo: u32 = a * b;
  let hi: u32 = select(b, 0u, lo > a) + select(a, 0u, lo > b) + c.second;
  let result: WideNumber = WideNumber(lo + c.first, hi);
  return result;
}

fn prmt(lo: u32, hi: u32, control: u32) -> u32 {
  var r : u32 = 0u;

  for (var i : i32 = 0; i < 32; i = i + 1) {
    var mask : u32 = 1u << u32(i);
    var lo_bit : u32 = (lo & mask) >> u32(i);
    var hi_bit : u32 = (hi & mask) >> u32(i);
    var control_bit : u32 = (control & mask) >> u32(i);

    var result_bit : u32 = 0u;

    if (control_bit == 0u) {
      result_bit = lo_bit;
    } else if (control_bit == 1u) {
      result_bit = hi_bit;
    } else if (control_bit == 2u) {
      result_bit = 0u;
    } else if (control_bit == 3u) {
      result_bit = 1u;
    }

    r = r | (result_bit << u32(i));
  }

  return r;
}

fn make_uint2(x: u32, y: u32) -> vec2<u32> {
  return vec2<u32>(x, y);
}

fn make_uint4(x: u32, y: u32, z: u32, w: u32) -> vec4<u32> {
  return vec4<u32>(x, y, z, w);
}

struct SharedMemory {
  data : array<u32>,
};

@group(0) @binding(1) var<storage, read_write> memory: SharedMemory;

fn load_shared_u4(s_addr: u32) -> vec4<u32> {
  var result: vec4<u32>;

  result.x = memory.data[s_addr];
  result.y = memory.data[s_addr + 1u];
  result.z = memory.data[s_addr + 2u];
  result.w = memory.data[s_addr + 3u];

  return result;
}

fn load_shared_u4_2(x: ptr<function, u32>, y: ptr<function, u32>, z: ptr<function, u32>, w: ptr<function, u32>, s_addr: u32) {
  (*x) = memory.data[s_addr];
  (*y) = memory.data[s_addr + 1u];
  (*z) = memory.data[s_addr + 2u];
  (*w) = memory.data[s_addr + 3u];
}

fn load_shared_u2(s_addr: u32) -> vec2<u32> {
  var result: vec2<u32>;

  result.x = memory.data[s_addr];
  result.y = memory.data[s_addr + 1u];

  return result;
}

fn store_shared_u2(s_addr: u32, value: vec2<u32>) {
  memory.data[s_addr] = value.x;
  memory.data[s_addr + 1u] = value.y;
}

fn store_shared_u4(s_addr: u32, value: vec4<u32>) {
  memory.data[s_addr] = value.x;
  memory.data[s_addr + 1u] = value.y;
  memory.data[s_addr + 2u] = value.z;
  memory.data[s_addr + 3u] = value.w;
}

fn store_shared_u32(s_addr: u32, value: u32) {
  memory.data[s_addr] = value;
}

fn load_shared_u32(s_addr: u32) -> u32 {
  return memory.data[s_addr];
}

fn shared_atomic_exch_u32(sAddr: u32, value: u32) -> u32 {
  var r: u32;

  r = memory.data[sAddr];
  memory.data[sAddr] = memory.data[value];

  return r; 
}

fn shared_reduce_add_u32(sAddr: u32, value: u32) {
  memory.data[sAddr] += value;
}

fn shared_atomic_add_u32(sAddr: u32, value: u32) -> u32 {
  var r: u32;

  memory.data[sAddr] += value;
  r = memory.data[sAddr];

  return r;
}

struct GlobalMemory {
  data : array<u32>,
}

@group(0) @binding(2) var<storage, read_write> global_memory: GlobalMemory;

fn shared_async_copy_u4(sAddr: u32, pointer: u32) {
  for(var i =0; i<16;i++){
    memory.data[(pointer+u32(i)) % 301u] = memory.data[(sAddr+u32(i)) % 301u];
  }
}

struct Thread {
  threadIdx: vec2<u32>,
  blockIdx: vec2<u32>,
  blockDim: vec2<u32>,
  gridDim: vec2<u32>,
  data: u32,
}

