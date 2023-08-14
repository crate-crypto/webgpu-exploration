const BLS12377_MainField_NP0: u32 = 0xFFFFFFFFu;
const zeroConstant: u32 = 0u;

// namespace BLS12377
const limbs: u32 = 12u;
//const bytes: u32 = limbs * 4;
const bytes: u32 = 48u;

//typedef BLS12377::MainField MainField;
// type array<u32, limbs> = array<u32, limbs>;
// type EvenOdd = array<u64, limbs>;

fn qTerm(lowWord: u32) -> u32 { // +
  // Since np0 is 0xFFFFFFFF, this is just -lowWord
  return zeroConstant - lowWord;
}

fn loadShared(field: ptr<function, array<u32, limbs>>, offset: u32) { // +
  // uint4 -> vec4<u32>
  var u4a: vec4<u32>;
  var u4b: vec4<u32>;
  var u4c: vec4<u32>;

  u4a=load_shared_u4(offset);
  u4b=load_shared_u4(offset+16u);
  u4c=load_shared_u4(offset+32u);

  (*field)[0]=u4a.x;
  (*field)[1]=u4a.y;
  (*field)[2]=u4a.z;
  (*field)[3]=u4a.w;
  (*field)[4]=u4b.x; 
  (*field)[5]=u4b.y;
  (*field)[6]=u4b.z;
  (*field)[7]=u4b.w;
  (*field)[8]=u4c.x;
  (*field)[9]=u4c.y;
  (*field)[10]=u4c.z;
  (*field)[11]=u4c.w;
}

// loads data from pointer to field
// void* ptr -> ptr<function, u32>
fn load(field: ptr<function, array<u32, limbs>>, pointer: array<vec4<u32>, 3>) { // +
  var u4a: vec4<u32>;
  var u4b: vec4<u32>;
  var u4c: vec4<u32>;

  u4a = pointer[0];
  u4b = pointer[1];
  u4c = pointer[2];

  (*field)[0]=u4a.x;
  (*field)[1]=u4a.y;
  (*field)[2]=u4a.z;
  (*field)[3]=u4a.w;
  (*field)[4]=u4b.x; 
  (*field)[5]=u4b.y;
  (*field)[6]=u4b.z;
  (*field)[7]=u4b.w;
  (*field)[8]=u4c.x;
  (*field)[9]=u4c.y;
  (*field)[10]=u4c.z;
  (*field)[11]=u4c.w;
}

fn loadUnaligned(field: ptr<function, array<u32, limbs>>, pointer: ptr<function, array<vec4<u32>, 3>>) { // +
  var j: i32 = 0;
  for(var i: i32 = 0; i < i32(limbs); i = i + 4) {
    (*field)[i] = (*pointer)[j].x;
    (*field)[i+1] = (*pointer)[j].y;
    (*field)[i+2] = (*pointer)[j].z;
    (*field)[i+3] = (*pointer)[j].w;
    j++;
  }
}

// stores field array<u32, limbs>s into a pointer
// void* ptr -> ptr<function, u32>
fn store(pointer: ptr<function, array<vec4<u32>, 3>>, field: array<u32, limbs>) { // +
  var u4a: vec4<u32>;
  var u4b: vec4<u32>;
  var u4c: vec4<u32>;

  u4a.x=field[0];
  u4a.y=field[1];
  u4a.z=field[2];
  u4a.w=field[3];
  u4b.x=field[4];
  u4b.y=field[5];
  u4b.z=field[6];
  u4b.w=field[7];
  u4c.x=field[8];
  u4c.y=field[9];
  u4c.z=field[10];
  u4c.w=field[11];

  (*pointer)[0] = u4a;
  (*pointer)[1] = u4b;
  (*pointer)[2] = u4c;
}

fn isEqual(a: array<u32, limbs>, b: array<u32, limbs>) -> bool { // +
  var compare : u32 = a[0] ^ b[0];

  var a_copy = a;
  var b_copy = b;

  for (var i : i32 = 1; i < i32(limbs); i = i + 1) {
    compare = compare | (a_copy[i] ^ b_copy[i]);
  }

  return compare == 0u;
}

fn setConstant(field: ptr<function, array<u32, limbs>>, index: u32) { // +
  var u4a: vec4<u32>;
  var u4b: vec4<u32>;
  var u4c: vec4<u32>;

  u4a=load_shared_u4(512u*0u + 16u*index);
  u4b=load_shared_u4(512u*0u + 16u*index);
  u4c=load_shared_u4(512u*0u + 16u*index);

  (*field)[0]=u4a.x;
  (*field)[1]=u4a.y;
  (*field)[2]=u4a.z;
  (*field)[3]=u4a.w;
  (*field)[4]=u4b.x; 
  (*field)[5]=u4b.y;
  (*field)[6]=u4b.z;
  (*field)[7]=u4b.w;
  (*field)[8]=u4c.x;
  (*field)[9]=u4c.y;
  (*field)[10]=u4c.z;
  (*field)[11]=u4c.w;
}

fn setZero(field: ptr<function, array<u32, limbs>>) { // +
  setConstant(field, 0u);
}

fn setN(field: ptr<function, array<u32, limbs>>) { // +
  setConstant(field, 1u);
}

fn setOne(field: ptr<function, array<u32, limbs>>) { // +
  setConstant(field, 16u);
}

fn setRSquared(field: ptr<function, array<u32, limbs>>) { // +
  setConstant(field, 17u);
}

// name `set` is a reserved keyword
fn set_(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var a_copy = a;
  for (var i : i32 = 0; i < 12; i = i + 1) {
    (*field)[i]=a_copy[i];
  }
}

fn setPermuteLow(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var a_copy = a;

  for (var i : i32 = 0; i < 12; i = i + 1) {
    (*field)[i]=prmt(a_copy[i], 0u, 0x3210u); 
  }
}

fn setPermuteHigh(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var a_copy = a;

  for (var i : i32 = 0; i < 12; i = i + 1) {
    (*field)[i]=prmt(0u, a_copy[i], 0x7654u); 
  }
}

fn swap(a: ptr<function, array<u32, limbs>>, b: ptr<function, array<u32, limbs>>) { // +
  for (var i : i32 = 0; i < 12; i = i + 1) {
    var swap: u32 = (*a)[i];
    (*a)[i] = (*b)[i];
    (*b)[i] = swap;
  }
}

fn add(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>, b: array<u32, limbs>) { // +
  mp_add(limbs, field, a, b);
}

fn sub(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>, b: array<u32, limbs>) { // +
  mp_sub_carry(limbs, field, a, b);
}

fn mul1(r: ptr<function, array<u32, limbs>>, a: array<u32, limbs>, b: array<u32, limbs>, n: array<u32, limbs>) { // +
  var carry: bool;
  var evenOdd: array<WideNumber, limbs>;

  // if defined(PHONY)
  let PHONY = true;

  if PHONY
  {
    // useful for inspecting the SASS
    carry = false;

    var a_copy = a;
    var b_copy = b;
    for (var i : i32 = 0; i < 12; i = i + 1) {
      evenOdd[i] = mulwide(a_copy[i], 0xCAFEBABEu);
      evenOdd[i] = madwide(b_copy[i], 0xBAADF00Du, evenOdd[i]);
    }
  }
  else
  {
    carry = mp_mul_red_cl(&evenOdd, a, b, n);
  }
  mp_merge_cl(r, evenOdd, carry);
}

fn sqr1(r: ptr<function, array<u32, limbs>>, temp: ptr<function, array<u32, limbs>>, a: array<u32, limbs>, n: array<u32, limbs>) { // +
  var carry: bool;
  var evenOdd: array<WideNumber, limbs>;

  // if defined(PHONY)
  let PHONY = true;

  if PHONY
  {
    // useful for inspecting the SASS
    carry = false;
    var a_copy = a;
    for (var i : i32 = 0; i < 12; i = i + 1) {
      evenOdd[i] = mulwide(a_copy[i], 0xCAFEBABEu);
    }
  }
  else
  {
    carry = mp_sqr_red_cl(&evenOdd, temp, a, n);
  }
  mp_merge_cl(r, evenOdd, carry);
}

fn mul2(evenOdd: ptr<function, array<WideNumber, limbs>>, carry: ptr<function, bool>, a: array<u32, limbs>, b: array<u32, limbs>, n: array<u32, limbs>) { // +
  // if defined(PHONY)
  let PHONY = true;

  if PHONY
  {
    // useful for inspecting the SASS
    *carry = false;
    var a_copy = a;
    var b_copy = b;
    for (var i : i32 = 0; i < 12; i = i + 1) {
      (*evenOdd)[i] = mulwide(a_copy[i], 0xCAFEBABEu);
      (*evenOdd)[i] = madwide(b_copy[i], 0xBAADF00Du, (*evenOdd)[i]);
    }
  }
  else
  {
    *carry = mp_mul_red_cl(evenOdd, a, b, n);
  }
}

fn sqr2(evenOdd: ptr<function, array<WideNumber, limbs>>, temp: ptr<function, array<u32, limbs>>, carry: ptr<function, bool>, a: array<u32, limbs>, n: array<u32, limbs>) -> bool { // +
  // if defined(PHONY)
  let PHONY = true;

  if PHONY
  {
    // useful for inspecting the SASS
    *carry = false;
    var a_copy = a;
    for (var i : i32 = 0; i < 12; i = i + 1) {
      (*evenOdd)[i] = mulwide(a_copy[i], 0xCAFEBABEu);
    }
  }
  else
  {
    *carry = mp_sqr_red_cl(evenOdd, temp, a, n);
  }
  return *carry;
}

fn merge(r: ptr<function, array<u32, limbs>>, evenOdd: array<WideNumber, limbs>, carry: bool) { // +
  mp_merge_cl(r, evenOdd, carry);
}

fn addN(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var localN: array<u32, limbs>;

  localN[0]  = 0x00000001u;
  localN[1]  = 0x8508C000u;
  localN[2]  = 0x30000000u;
  localN[3]  = 0x170B5D44u;
  localN[4]  = 0xBA094800u;
  localN[5]  = 0x1EF3622Fu;
  localN[6]  = 0x00F5138Fu;
  localN[7]  = 0x1A22D9F3u;
  localN[8]  = 0x6CA1493Bu;
  localN[9]  = 0xC63B05C0u;
  localN[10] = 0x17C510EAu;
  localN[11] = 0x01AE3A46u;
  add(field, a, localN);
}

fn add2N(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var local2N: array<u32, limbs>;

  local2N[0]  = 0x00000002u;
  local2N[1]  = 0x0A118000u;
  local2N[2]  = 0x60000001u;
  local2N[3]  = 0x2E16BA88u;
  local2N[4]  = 0x74129000u;
  local2N[5]  = 0x3DE6C45Fu;
  local2N[6]  = 0x01EA271Eu;
  local2N[7]  = 0x3445B3E6u;
  local2N[8]  = 0xD9429276u;
  local2N[9]  = 0x8C760B80u;
  local2N[10] = 0x2F8A21D5u;
  local2N[11] = 0x035C748Cu;
  add(field, a, local2N);
}

fn add3N(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var local3N: array<u32, limbs>;

  local3N[0]  = 0x00000003u;
  local3N[1]  = 0x8F1A4000u;
  local3N[2]  = 0x90000001u;
  local3N[3]  = 0x452217CCu;
  local3N[4]  = 0x2E1BD800u;
  local3N[5]  = 0x5CDA268Fu;
  local3N[6]  = 0x02DF3AADu;
  local3N[7]  = 0x4E688DD9u;
  local3N[8]  = 0x45E3DBB1u;
  local3N[9]  = 0x52B11141u;
  local3N[10] = 0x474F32C0u;
  local3N[11] = 0x050AAED2u;
  add(field, a, local3N);
}

fn add4N(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var local4N: array<u32, limbs>;

  local4N[0]  = 0x00000004u;
  local4N[1]  = 0x14230000u;
  local4N[2]  = 0xC0000002u;
  local4N[3]  = 0x5C2D7510u;
  local4N[4]  = 0xE8252000u;
  local4N[5]  = 0x7BCD88BEu;
  local4N[6]  = 0x03D44E3Cu;
  local4N[7]  = 0x688B67CCu;
  local4N[8]  = 0xB28524ECu;
  local4N[9]  = 0x18EC1701u;
  local4N[10] = 0x5F1443ABu;
  local4N[11] = 0x06B8E918u;
  add(field, a, local4N);
}

fn add5N(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var local5N: array<u32, limbs>;

  local5N[0]  = 0x00000005u;
  local5N[1]  = 0x992BC000u;
  local5N[2]  = 0xF0000002u;
  local5N[3]  = 0x7338D254u;
  local5N[4]  = 0xA22E6800u;
  local5N[5]  = 0x9AC0EAEEu;
  local5N[6]  = 0x04C961CBu;
  local5N[7]  = 0x82AE41BFu;
  local5N[8]  = 0x1F266E27u;
  local5N[9]  = 0xDF271CC2u;
  local5N[10] = 0x76D95495u;
  local5N[11] = 0x0867235Eu;
  add(field, a, local5N);
}

fn add6N(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var local6N: array<u32, limbs>;

  local6N[0]  = 0x00000006u;
  local6N[1]  = 0x1E348000u;
  local6N[2]  = 0x20000003u;
  local6N[3]  = 0x8A442F99u;
  local6N[4]  = 0x5C37B000u;
  local6N[5]  = 0xB9B44D1Eu;
  local6N[6]  = 0x05BE755Au;
  local6N[7]  = 0x9CD11BB2u;
  local6N[8]  = 0x8BC7B762u;
  local6N[9]  = 0xA5622282u;
  local6N[10] = 0x8E9E6580u;
  local6N[11] = 0x0A155DA4u;
  add(field, a, local6N);
}

fn negateAddN(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var localN: array<u32, limbs>;

  localN[0]  = 0x00000001u;
  localN[1]  = 0x8508C000u;
  localN[2]  = 0x30000000u;
  localN[3]  = 0x170B5D44u;
  localN[4]  = 0xBA094800u;
  localN[5]  = 0x1EF3622Fu;
  localN[6]  = 0x00F5138Fu;
  localN[7]  = 0x1A22D9F3u;
  localN[8]  = 0x6CA1493Bu;
  localN[9]  = 0xC63B05C0u;
  localN[10] = 0x17C510EAu;
  localN[11] = 0x01AE3A46u;
  sub(field, a, localN);
}

fn negateAdd4N(field: ptr<function, array<u32, limbs>>, a: array<u32, limbs>) { // +
  var local4N: array<u32, limbs>;

  local4N[0]  = 0x00000004u;
  local4N[1]  = 0x14230000u;
  local4N[2]  = 0xC0000002u;
  local4N[3]  = 0x5C2D7510u;
  local4N[4]  = 0xE8252000u;
  local4N[5]  = 0x7BCD88BEu;
  local4N[6]  = 0x03D44E3Cu;
  local4N[7]  = 0x688B67CCu;
  local4N[8]  = 0xB28524ECu;
  local4N[9]  = 0x18EC1701u;
  local4N[10] = 0x5F1443ABu;
  local4N[11] = 0x06B8E918u;
  sub(field, a, local4N);
}

fn isZero(field: array<u32, limbs>) -> bool { // +
  var x: u32;
  var mult: u32;
  var loaded: array<u32, limbs>;

  // note, this routine only works for field <= 8N
  x = (field[10] * u32(2 ^ 4)) | (field[11] / u32(2 ^ 28));
  mult = x * 10u;

  if (mult*0x1AE3A461u+3u-x)<=3u {
    setConstant(&loaded, mult);
    return mp_comp_eq(loaded, field);
  }
  return false;
}

fn reduce(r: ptr<function, array<u32, limbs>>, field: array<u32, limbs>) {
  var x: u32;
  var mult: u32;
  var local: array<u32, limbs>;

  // note, this routine only works for field <= 8N
  x = (field[10] * u32(2 ^ 4)) | (field[11] / u32(2 ^ 28));
  mult = x * 10u;

  setConstant(&local, mult);
  if !mp_sub_carry(limbs, r, field, local) {
    addN(r, (*r));
  }
}

fn warpShuffle(r: ptr<function, array<u32, limbs>>, field: array<u32, limbs>, sourceLane: u32) {
  var field_copy =field;

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    (*r)[i] = field_copy[i];
  }
}

// name `external` is a reserved keyword
fn toInternal(r: ptr<function, array<u32, limbs>>, external_: array<u32, limbs>) {
  var localRSquared: array<u32, limbs>;
  var localN: array<u32, limbs>;

  setRSquared(&localRSquared);
  setN(&localN);
  mul1(r, external_, localRSquared, localN);

  if mp_comp_ge(*r, localN) {
    mp_sub(limbs, r, *r, localN);
  }
}

fn fromInternal(r: ptr<function, array<u32, limbs>>, internal: array<u32, limbs>) {
  var one: array<u32, limbs>;
  var localN: array<u32, limbs>;

  mp_zero(&one);
  one[0] = 1u;
  setN(&localN);
  mul1(r, internal, one, localN);
}

// namespace BLS12377 

struct PointXY {
  x: array<u32, limbs>,
  y: array<u32, limbs>,
}; 

// start of PointXY impl

fn initialize_PointXY(self_: ptr<function, PointXY>, xValue: array<u32, limbs>, yValue: array<u32, limbs>) { // +
  var self_x = (*self_).x;
  var self_y = (*self_).y;

  set_(&self_x, xValue);
  set_(&self_y, yValue);

  (*self_).x = self_x;
  (*self_).y = self_y;
}

fn loadShared_PointXY(self_: ptr<function, PointXY>, offset: u32) {
  var self_x = (*self_).x;
  var self_y = (*self_).y;

  loadShared(&self_x, (offset + bytes*0u));
  loadShared(&self_y, (offset + bytes*1u));

  (*self_).x = self_x;
  (*self_).y = self_y;

  var x: i32 = 0;
  let q: ptr<function, i32> = &x;
}

fn load_PointXY(self_: ptr<function, PointXY>, pointer: array<vec4<u32>, 6>) { // +
  var slice1: array<vec4<u32>, 3>;

  slice1[0] = pointer[0];
  slice1[1] = pointer[1];
  slice1[2] = pointer[2];

  var slice2: array<vec4<u32>, 3>;

  slice2[0] = pointer[3];
  slice2[1] = pointer[4];
  slice2[2] = pointer[5];

  var self_x = (*self_).x;
  var self_y = (*self_).y;

  load(&self_x, slice1);
  load(&self_y, slice2);

  (*self_).x = self_x;
  (*self_).y = self_y;
}

fn loadUnaligned_PointXY(self_: ptr<function, PointXY>, pointer: ptr<function, array<vec4<u32>, 6>>) { // +
  var self_x = (*self_).x;
  var self_y = (*self_).y;

  var slice1: array<vec4<u32>, 3>;

  slice1[0] = (*pointer)[0];
  slice1[1] = (*pointer)[1];
  slice1[2] = (*pointer)[2];

  var slice2: array<vec4<u32>, 3>;

  slice2[0] = (*pointer)[3];
  slice2[1] = (*pointer)[4];
  slice2[2] = (*pointer)[5];

  loadUnaligned(&self_x, &slice1);
  loadUnaligned(&self_y, &slice2);

  (*self_).x = self_x;
  (*self_).y = self_y;
}

fn store_PointXY(self_: PointXY, pointer: ptr<function, array<vec4<u32>, 6>>) { // +
  var slice1: array<vec4<u32>, 3>;

  slice1[0] = (*pointer)[0];
  slice1[1] = (*pointer)[1];
  slice1[2] = (*pointer)[2];

  var slice2: array<vec4<u32>, 3>;

  slice2[0] = (*pointer)[3];
  slice2[1] = (*pointer)[4];
  slice2[2] = (*pointer)[5];

  store(&slice1, self_.x);
  store(&slice2, self_.y);

  (*pointer)[0] = slice1[0];
  (*pointer)[1] = slice1[1];
  (*pointer)[2] = slice1[2];
  (*pointer)[3] = slice2[0];
  (*pointer)[4] = slice2[1];
  (*pointer)[5] = slice2[2];
}

fn negate_PointXY(self_: ptr<function, PointXY>) {
  var self_y = (*self_).y;

  negateAddN(&self_y, self_y);

  (*self_).y = self_y;
}

fn reduce_PointXY(self_: ptr<function, PointXY>) {
  var self_x = (*self_).x;
  var self_y = (*self_).y;

  reduce(&self_x, self_x);
  reduce(&self_y, self_y);

  (*self_).x = self_x;
  (*self_).y = self_y;
}

fn warpShuffle_PointXY(self_: ptr<function, PointXY>, sourceLane: u32) {
  var self_x = (*self_).x;
  var self_y = (*self_).y;

  warpShuffle(&self_x, self_x, sourceLane);
  warpShuffle(&self_y, self_y, sourceLane);

  (*self_).x = self_x;
  (*self_).y = self_y;
}

fn fromInternal_PointXY(self_: ptr<function, PointXY>) {
  var self_x = (*self_).x;
  var self_y = (*self_).y;

  fromInternal(&self_x, self_x);
  fromInternal(&self_y, self_y);

  (*self_).x = self_x;
  (*self_).y = self_y;
}

// end of PointXY impl

struct PointXYZZ {
  x: array<u32, limbs>,
  y: array<u32, limbs>,
  zz: array<u32, limbs>,
  zzz: array<u32, limbs>,
}; 

// start of PointXYZZ impl

fn initialize_PointXYZZ(self_: ptr<function, PointXYZZ>, xValue: array<u32, limbs>, yValue: array<u32, limbs>, zzValue: array<u32, limbs>, zzzValue: array<u32, limbs>) { // +
  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;
  
  set_(&self_x, xValue);
  set_(&self_y, yValue);
  set_(&self_zz, zzValue);
  set_(&self_zzz, zzzValue);

  (*self_).x = self_x;
  (*self_).y = self_y;
  (*self_).zz = self_zz;
  (*self_).zzz = self_zzz;
}

fn loadSharedXY(self_: ptr<function, PointXYZZ>, offset: u32) {
  var self_x = (*self_).x;
  var self_y = (*self_).y;

  loadShared(&self_x, (offset + bytes*0u));
  loadShared(&self_y, (offset + bytes*1u));

  (*self_).x = self_x;
  (*self_).y = self_y;
}

fn loadSharedZZ(self_: ptr<function, PointXYZZ>, offset: u32) {
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  loadShared(&self_zz, (offset + bytes*0u));
  loadShared(&self_zzz, (offset + bytes*1u));

  (*self_).zz = self_zz;
  (*self_).zzz = self_zzz;
}

fn loadShared_PointXYZZ(self_: ptr<function, PointXYZZ>, offset: u32) {
  loadSharedXY(self_, offset);
  loadSharedZZ(self_, (offset + bytes*2u));
}

fn load_PointXYZZ(self_: ptr<function, PointXYZZ>, pointer: array<vec4<u32>, 12>) { // +
  var slice1: array<vec4<u32>, 3>;

  slice1[0] = pointer[0];
  slice1[1] = pointer[1];
  slice1[2] = pointer[2];

  var slice2: array<vec4<u32>, 3>;

  slice2[0] = pointer[3];
  slice2[1] = pointer[4];
  slice2[2] = pointer[5];

  var slice3: array<vec4<u32>, 3>;

  slice3[0] = pointer[6];
  slice3[1] = pointer[7];
  slice3[2] = pointer[8];

  var slice4: array<vec4<u32>, 3>;

  slice4[0] = pointer[9];
  slice4[1] = pointer[10];
  slice4[2] = pointer[11];

  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  load(&self_x, slice1);
  load(&self_y, slice2);
  load(&self_zz, slice3);
  load(&self_zzz, slice4);

  (*self_).x = self_x;
  (*self_).y = self_y;
  (*self_).zz = self_zz;
  (*self_).zzz = self_zzz;
}

fn loadUnaligned_PointXYZZ(self_: ptr<function, PointXYZZ>, pointer: ptr<function, array<vec4<u32>, 12>>) { // +
  var slice1: array<vec4<u32>, 3>;

  slice1[0] = (*pointer)[0];
  slice1[1] = (*pointer)[1];
  slice1[2] = (*pointer)[2];

  var slice2: array<vec4<u32>, 3>;

  slice2[0] = (*pointer)[3];
  slice2[1] = (*pointer)[4];
  slice2[2] = (*pointer)[5];

  var slice3: array<vec4<u32>, 3>;

  slice3[0] = (*pointer)[6];
  slice3[1] = (*pointer)[7];
  slice3[2] = (*pointer)[8];

  var slice4: array<vec4<u32>, 3>;

  slice4[0] = (*pointer)[9];
  slice4[1] = (*pointer)[10];
  slice4[2] = (*pointer)[11];

  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  loadUnaligned(&self_x, &slice1);
  loadUnaligned(&self_y, &slice2);
  loadUnaligned(&self_zz, &slice3);
  loadUnaligned(&self_zzz, &slice4);

  (*self_).x = self_x;
  (*self_).y = self_y;
  (*self_).zz = self_zz;
  (*self_).zzz = self_zzz;
}

fn store_PointXYZZ(self_: PointXYZZ, pointer: ptr<function, array<vec4<u32>, 12>>) { // +
  var slice1: array<vec4<u32>, 3>;

  slice1[0] = (*pointer)[0];
  slice1[1] = (*pointer)[1];
  slice1[2] = (*pointer)[2];

  var slice2: array<vec4<u32>, 3>;

  slice2[0] = (*pointer)[3];
  slice2[1] = (*pointer)[4];
  slice2[2] = (*pointer)[5];

  var slice3: array<vec4<u32>, 3>;

  slice2[0] = (*pointer)[6];
  slice2[1] = (*pointer)[7];
  slice2[2] = (*pointer)[8];

  var slice4: array<vec4<u32>, 3>;

  slice2[0] = (*pointer)[9];
  slice2[1] = (*pointer)[10];
  slice2[2] = (*pointer)[11];

  store(&slice1, self_.x);
  store(&slice2, self_.y);
  store(&slice3, self_.zz);
  store(&slice4, self_.zzz);

  (*pointer)[0] = slice1[0];
  (*pointer)[1] = slice1[1];
  (*pointer)[2] = slice1[2];
  (*pointer)[3] = slice2[0];
  (*pointer)[4] = slice2[1];
  (*pointer)[5] = slice2[2];
  (*pointer)[6] = slice3[0];
  (*pointer)[7] = slice3[1];
  (*pointer)[8] = slice3[2];
  (*pointer)[9] = slice4[0];
  (*pointer)[10] = slice4[1];
  (*pointer)[11] = slice4[2];
}

fn negate_PointXYZZ(self_: ptr<function, PointXYZZ>) {
  var self_y = (*self_).y;

  negateAdd4N(&self_y, self_y);

  (*self_).y = self_y;
}

fn reduce_PointXYZZ(self_: ptr<function, PointXYZZ>) {
  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  reduce(&self_x, self_x);
  reduce(&self_y, self_y);
  reduce(&self_zz, self_zz);
  reduce(&self_zzz, self_zzz);

  (*self_).x = self_x;
  (*self_).y = self_y;
  (*self_).zz = self_zz;
  (*self_).zzz = self_zzz;
}

fn warpShuffle_PointXYZZ(self_: ptr<function, PointXYZZ>, sourceLane: u32) {
  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  warpShuffle(&self_x, self_x, sourceLane);
  warpShuffle(&self_y, self_y, sourceLane);
  warpShuffle(&self_zz, self_zz, sourceLane);
  warpShuffle(&self_zzz, self_zzz, sourceLane);

  (*self_).x = self_x;
  (*self_).y = self_y;
  (*self_).zz = self_zz;
  (*self_).zzz = self_zzz;
}

fn fromInternal_PointXYZZ(self_: ptr<function, PointXYZZ>) {
  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  fromInternal(&self_x, self_x);
  fromInternal(&self_y, self_y);
  fromInternal(&self_zz, self_zz);
  fromInternal(&self_zzz, self_zzz);

  (*self_).x = self_x;
  (*self_).y = self_y;
  (*self_).zz = self_zz;
  (*self_).zzz = self_zzz;
}

// end of PointXYZZ impl

// namespace CurveXYZZ

struct HighThroughput {
  x: array<u32, limbs>,
  y: array<u32, limbs>,
  zz: array<u32, limbs>,
  zzz: array<u32, limbs>,
  infinity: bool,
  affine: bool,
};

// start of HighThroughput impl

fn setZero_HighThroughput(self_: ptr<function, HighThroughput>) { // +
  (*self_).infinity=true;
  (*self_).affine=false;
}

fn load_HighThroughput(self_: ptr<function, HighThroughput>, pointer: array<vec4<u32>, 12>) { // +
  (*self_).infinity=false;
  (*self_).affine=false;

  var slice1: array<vec4<u32>, 3>;

  slice1[0] = pointer[0];
  slice1[1] = pointer[1];
  slice1[2] = pointer[2];

  var slice2: array<vec4<u32>, 3>;

  slice2[0] = pointer[3];
  slice2[1] = pointer[4];
  slice2[2] = pointer[5];

  var slice3: array<vec4<u32>, 3>;

  slice3[0] = pointer[6];
  slice3[1] = pointer[7];
  slice3[2] = pointer[8];

  var slice4: array<vec4<u32>, 3>;

  slice4[0] = pointer[9];
  slice4[1] = pointer[10];
  slice4[2] = pointer[11];

  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  load(&self_x, slice1);
  load(&self_y, slice2);
  load(&self_zz, slice3);
  load(&self_zzz, slice4);

  (*self_).x = self_x;
  (*self_).y = self_y;
  (*self_).zz = self_zz;
  (*self_).zzz = self_zzz;
}

fn add1_HighThroughput(self_: ptr<function, HighThroughput>, point: ptr<function, PointXY>, valid: bool) { // +
  var evenOdd: array<WideNumber, limbs>;
  var N: array<u32, limbs>;
  var A: array<u32, limbs>;
  var B: array<u32, limbs>;
  var T0: array<u32, limbs>;
  var T1: array<u32, limbs>;
  var state: u32;
  var carry: bool;
  var done: bool;
  var uniformAffine: bool;
  var dbl: bool = false;
  var square: bool = false;

  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  var point_x = (*point).x;
  var point_y = (*point).y;

  if (*self_).infinity {
    (*self_).infinity = false;
    setPermuteLow(&self_x, point_x);
    setPermuteLow(&self_y, point_y);

    (*self_).x = self_x;
    (*self_).y = self_y;

    (*self_).affine=valid;

    if !valid {
      setZero(&self_zz);

      (*self_).zz = self_zz;
    }
    return;
  }

  // uniformAffine=__all_sync(0xFFFFFFFF, (*self_).affine);

  done = !valid;

  if valid && !(*self_).affine && isZero((*self_).zz) {
    setPermuteLow(&self_x, point_x);
    setPermuteLow(&self_y, point_y);

    (*self_).x = self_x;
    (*self_).y = self_y;
    (*point).x = point_x;
    (*point).y = point_y;

    
    (*self_).affine=true;
    done=true;
  }

  if uniformAffine {
    sub(&T0, point_x, self_x);
    add6N(&T0, T0);  
    sub(&T1, point_y, self_y);
    add4N(&T1, T1);
    square = true;
    setPermuteLow(&A, T0);
    state = 2u;

    (*self_).x = self_x;
    (*self_).y = self_y;
    (*point).x = point_x;
    (*point).y = point_y;
  }
  else {
    if (*self_).affine {
      setOne(&self_zz);
      setOne(&self_zzz);

      (*self_).zz = self_zz;
      (*self_).zzz = self_zzz;
    }
    setPermuteLow(&A, self_zz); 
    setPermuteLow(&B, point_x);

    (*self_).zz = self_zz;
    (*point).x = point_x;
    state = 0u;
  }

  N[0]=0x00000001u; 
  N[1]=0x8508C000u; 
  N[2]=0x30000000u; 
  N[3]=0x170B5D44u; 
  N[4]=0xBA094800u; 
  N[5]=0x1EF3622Fu;
  N[6]=0x00F5138Fu; 
  N[7]=0x1A22D9F3u; 
  N[8]=0x6CA1493Bu; 
  N[9]=0xC63B05C0u; 
  N[10]=0x17C510EAu; 
  N[11]=0x01AE3A46u;

  while (!done) {
    if(!square) 
    {
      mul2(&evenOdd, &carry, A, B, N);
    }
    else 
    {
      sqr2(&evenOdd, &B, &carry, A, N);
      square=false;
    }
    switch (state) {
      case 0u, default:{
        merge(&T0, evenOdd, carry);
        sub(&T0, T0, self_x);
        add6N(&T0, T0);
        setPermuteLow(&A, self_zzz); 
        setPermuteLow(&B, point_y);
        state = 1u;

        (*self_).x = self_x;
        (*self_).zzz = self_zzz;
        (*point).y = point_y;
        break;
      }
      case 1u:{
        merge(&T1, evenOdd, carry);
        sub(&T1, T1, self_y);
        add4N(&T1, T1);
        setPermuteLow(&A, T0);
        state = 2u;

        (*self_).y = self_y;
        break;
      }
      case 2u:{
        dbl= isZero(T0) && isZero(T1);
        merge(&B, evenOdd, carry);
        if uniformAffine {
          setPermuteLow(&self_zz, B);
          setPermuteLow(&A, self_x);
          state=4u;

          (*self_).zz = self_zz;
          (*self_).x = self_x;
        }
        else {
          setPermuteLow(&A, self_zz);
          state=3u;

          (*self_).zz = self_zz;
        }
        break;
      }
      case 3u:{
        merge(&self_zz, evenOdd, carry);
        setPermuteLow(&A, self_x);
        state=4u;

        (*self_).zz = self_zz;
        (*self_).x = self_x;
        break;
      }
      case 4u:{
        setPermuteLow(&A, T0);
        merge(&T0, evenOdd, carry);
        state=5u;
        break;
      }
      case 5u:{
        merge(&B, evenOdd, carry);

        if uniformAffine {
          setPermuteLow(&self_zzz, B);
          setPermuteLow(&A, self_y);
          state=7u;

          (*self_).zzz = self_zzz;
          (*self_).y = self_y;
        }
        else {
          setPermuteLow(&A, self_zzz);
          state=6u;

          (*self_).zzz = self_zzz;
        }
        break;
      }
      case 6u:{
        merge(&self_zzz, evenOdd, carry);
        setPermuteLow(&A, self_y);
        state=7u;

        (*self_).zzz = self_zzz;
        (*self_).y = self_y;
        break;
      }
      case 7u:{
        merge(&self_y, evenOdd, carry);
        add(&self_x, B, T0);
        (*self_).x = self_x;
        add(&self_x, self_x, T0);
        square=true;
        setPermuteLow(&A, T1);
        state=8u;

        (*self_).x = self_x;
        (*self_).y = self_y;
        break;
      }
      case 8u:{
        merge(&T1, evenOdd, carry);
        sub(&self_x, T1, self_x);
        (*self_).x = self_x;
        add4N(&self_x, self_x);
        (*self_).x = self_x;
        sub(&T0, T0, self_x);
        add6N(&T0, T0);
        setPermuteLow(&B, T0);
        state=9u;

        (*self_).x = self_x;
        break;
      }
      case 9u:{
        (*self_).affine=false;
        merge(&T0, evenOdd, carry);
        sub(&self_y, T0, self_y);
        (*self_).y = self_y;
        add2N(&self_y, self_y);

        (*self_).y = self_y;
        done=true;
      }
    }
  }

  // if __all_sync(0xFFFFFFFF, !dbl){
  //   return;
  // }

  done=!dbl;
  add(&T0, point_y, point_y);

  (*point).y = point_y;

  square=true;
  setPermuteLow(&A, T0);
  state=0u;

  while (!done) {
    if !square {
      mul2(&evenOdd, &carry, A, B, N);
    }
    else {
      sqr2(&evenOdd, &B, &carry, A, N);
      square=false;
    }
    switch (state) {
      case 0u, default {
        merge(&self_zz, evenOdd, carry);
        (*self_).zz = self_zz;
        setPermuteLow(&B, self_zz);
        state=1u;

        (*self_).zz = self_zz;
        break;
      }
      case 1u {
        merge(&self_zzz, evenOdd, carry);
        setPermuteLow(&A, point_x);
        state=2u;

        (*point).x = point_x;
        (*self_).zzz = self_zzz;
        break;
      }
      case 2u {
        merge(&T0, evenOdd, carry);
        setPermuteLow(&A, point_y); 
        setPermuteLow(&B, self_zzz);
        state=3u;

        (*point).y = point_y;
        (*self_).zzz = self_zzz;
        break;
      }
      case 3u {
        merge(&self_y, evenOdd, carry);
        square=true;
        setPermuteLow(&A, point_x);
        state=4u;

        (*self_).y = self_y;
        (*point).x = point_x;
        break;
      }
      case 4u {
        merge(&T1, evenOdd, carry);
        square=true;
        add(&A, T1, T1);
        add(&A, A, T1);
        state=5u;
        break;
      }
      case 5u {
        merge(&self_x, evenOdd, carry);
        (*self_).x = self_x;
        add(&B, T0, T0);
        sub(&self_x, self_x, B);
        (*self_).x = self_x;
        add3N(&self_x, self_x);
        (*self_).x = self_x;
        sub(&B, T0, self_x);
        add5N(&B, B);
        state=6u;

        (*self_).x = self_x;
        break;
      }
      case 6u {
        merge(&T0, evenOdd, carry);
        sub(&self_y, T0, self_y);
        (*self_).y = self_y;
        add2N(&self_y, self_y);
        done=true;

        (*self_).y = self_y;
        break; 
      }
    }
  }
  // __syncwarp(0xFFFFFFFF);
}

fn add2_HighThroughput(self_: ptr<function, HighThroughput>, point: ptr<function, PointXYZZ>, valid: bool) { // +
  var evenOdd: array<WideNumber, limbs>;
  var N: array<u32, limbs>;
  var A: array<u32, limbs>;
  var B: array<u32, limbs>;
  var T0: array<u32, limbs>;
  var T1: array<u32, limbs>;

  var state: u32 = 0u;
  var carry: bool;
  var done: bool;
  var dbl: bool;
  var square: bool = false;

  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;
  var point_x = (*point).x;
  var point_y = (*point).y;
  var point_zz = (*point).zz;
  var point_zzz = (*point).zzz;

  if (*self_).infinity {
    (*self_).infinity=false;      
    (*self_).affine=false;
    setPermuteLow(&self_x, point_x);
    setPermuteLow(&self_y, point_y);

    (*self_).x = self_x;
    (*self_).y = self_y;
    (*point).x = point_x;
    (*point).y = point_y;

    for (var i : i32 = 0; i < 12; i = i + 1) {
      if valid {
        (*self_).zz[i] = (*point).zz[i];
      }
      else {
        (*self_).zz[i] = 0u;
      }
      setPermuteLow(&self_zzz, point_zzz);

      (*self_).zzz = self_zzz;
      (*point).zzz = point_zzz;
      return;
    }
  }

  if (*self_).affine {
    (*self_).affine = false;
    setOne(&self_zz);
    setOne(&self_zzz);

    (*self_).zz = self_zz;
    (*self_).zzz = self_zzz;
  }

  if(valid && isZero((*self_).zz)) {
    setPermuteLow(&self_x, point_x);
    setPermuteLow(&self_y, point_y);
    setPermuteLow(&self_zz, point_zz);
    setPermuteLow(&self_zzz, point_zzz);
    done = true;

    (*self_).x = self_x;
    (*self_).y = self_y;
    (*self_).zz = self_zz;
    (*self_).zzz = self_zzz;
    (*point).x = point_x;
    (*point).y = point_y;
    (*point).zz = point_zz;
    (*point).zzz = point_zzz;
  }
  else {
    done=!valid || isZero((*point).zz);
  }

  N[0]=0x00000001u; 
  N[1]=0x8508C000u; 
  N[2]=0x30000000u; 
  N[3]=0x170B5D44u; 
  N[4]=0xBA094800u; 
  N[5]=0x1EF3622Fu;
  N[6]=0x00F5138Fu; 
  N[7]=0x1A22D9F3u; 
  N[8]=0x6CA1493Bu; 
  N[9]=0xC63B05C0u; 
  N[10]=0x17C510EAu; 
  N[11]=0x01AE3A46u;

  setPermuteLow(&A, self_zz); 
  setPermuteLow(&B, point_x); 

  (*self_).zz = self_zz;
  (*point).x = point_x;

  while (!done) {
    if !square {
      mul2(&evenOdd, &carry, A, B, N);
    } 
    else {
      sqr2(&evenOdd, &B, &carry, A, N);
      square = false;
    }
    switch (state) {
      case 0u, default: {
        merge(&T0, evenOdd, carry);
        setPermuteLow(&A, self_zzz); 
        setPermuteLow(&B, point_y);

        (*self_).zzz = self_zzz;
        (*point).y = point_y;
        break;
      }
      case 1u:{
        merge(&T1, evenOdd, carry);
        setPermuteLow(&A, self_x); 
        setPermuteLow(&B, point_zz); 

        (*self_).x = self_x;
        (*point).zz = point_zz;
        break;
      }
      case 2u:{
        merge(&self_x, evenOdd, carry);
        setPermuteLow(&A, self_y); 
        setPermuteLow(&B, point_zzz); 

        (*self_).x = self_x;
        (*self_).y = self_y;
        (*point).zzz = point_zzz;
        break;
      }
      case 3u:{
        merge(&self_y, evenOdd, carry);
        (*self_).y = self_y;
        sub(&T0, T0, self_x);
        (*self_).x = self_x;
        add6N(&T0, T0);
        sub(&T1, T1, self_y);
        add4N(&T1, T1);       
        square=true;
        setPermuteLow(&A, T0);

        (*self_).x = self_x;
        (*self_).y = self_y;
        break;
      }
      case 4u:{
        dbl=isZero(T0) && isZero(T1);
        merge(&B, evenOdd, carry);
        setPermuteLow(&A, self_zz);

        (*self_).zz = self_zz;
        break;
      }
      case 5u:{
        merge(&self_zz, evenOdd, carry);
        setPermuteLow(&A, self_x);

        (*self_).x = self_x;
        (*self_).zz = self_zz;
        break;
      }
      case 6u:{
        setPermuteLow(&A, T0);
        merge(&T0, evenOdd, carry);
        break;
      }
      case 7u:{
        merge(&B, evenOdd, carry);   // PPP is in B
        add(&self_x, B, T0);
        (*self_).x = self_x;
        add(&self_x, self_x, T0);
        (*self_).x = self_x;
        setPermuteLow(&A, self_zzz);

        (*self_).x = self_x;
        (*self_).zzz = self_zzz;
        break;
      }
      case 8u:{
        merge(&self_zzz, evenOdd, carry);
        setPermuteLow(&A, self_y);

        (*self_).zzz = self_zzz;
        (*self_).y = self_y;
        break;
      }
      case 9u:{
        merge(&self_y, evenOdd, carry);
        setPermuteLow(&A, self_zz); 
        setPermuteLow(&B, point_zz); 

        (*self_).y = self_y;
        (*self_).zz = self_zz;
        (*point).zz = point_zz;
        break;
      }
      case 10u:{
        merge(&self_zz, evenOdd, carry);
        setPermuteLow(&A, self_zzz); 
        setPermuteLow(&B, point_zzz); 

        (*self_).zz = self_zz;
        (*self_).zzz = self_zzz;
        (*point).zzz = point_zzz;
        break;
      }
      case 11u:{
        merge(&self_zzz, evenOdd, carry);
        square=true;
        setPermuteLow(&A, T1);

        (*self_).zzz = self_zzz;
        break;
      }
      case 12u:{
        merge(&T1, evenOdd, carry);
        sub(&self_x, T1, self_x);
        (*self_).x = self_x;
        add4N(&self_x, self_x);
        (*self_).x = self_x;
        sub(&T0, T0, self_x);
        add6N(&T0, T0);
        setPermuteLow(&B, T0);

        (*self_).x = self_x;
        break;
      }
      case 13u:{
        (*self_).affine=false;
        merge(&T0, evenOdd, carry);
        sub(&self_y, T0, self_y);
        (*self_).y = self_y;
        add2N(&self_y, self_y);

        (*self_).y = self_y;

        if dbl {
          add(&T0, point_y, point_y);
          square=true;
          setPermuteLow(&A, T0);

          (*point).y = point_y;
        }
        else
        {
          done=true;
        }
        break;
      }
      case 14u:{
        merge(&T0, evenOdd, carry);
        setPermuteLow(&B, T0);
        break;
      }
      case 15u:{
        merge(&T1, evenOdd, carry);
        setPermuteLow(&A, point_zz);

        (*point).zz = point_zz;
        break;
      }
      case 16u:{
        merge(&self_zz, evenOdd, carry);
        setPermuteLow(&A, T1); 
        setPermuteLow(&B, point_zzz); 

        (*self_).zz = self_zz;
        (*point).zzz = point_zzz;
        break;
      }
      case 17u:{
        merge(&self_zzz, evenOdd, carry);
        setPermuteLow(&B, point_y);

        (*self_).zzz = self_zzz;
        (*point).y = point_y;
        break;
      }
      case 18u:{
        merge(&self_y, evenOdd, carry);
        setPermuteLow(&A, point_x); 
        setPermuteLow(&B, T0); 

        (*self_).y = self_y;
        (*point).x = point_x;
        break;
      }
      case 19u:{
        merge(&T0, evenOdd, carry);
        square=true;
        break;
      }
      case 20u:{
        merge(&T1, evenOdd, carry);
        square=true;
        add(&A, T1, T1);
        add(&A, A, T1);
        break;
      }
      case 21u:{
        merge(&self_x, evenOdd, carry);
        (*self_).x = self_x;
        add(&T1, T0, T0);
        sub(&self_x, self_x, T1);
        (*self_).x = self_x;
        add3N(&self_x, self_x);
        (*self_).x = self_x;
        sub(&B, T0, self_x);
        add5N(&B, B);

        (*self_).x = self_x;
        break;
      }
      case 22u:{
        merge(&T0, evenOdd, carry);
        sub(&self_y, T0, self_y);
        (*self_).y = self_y;
        add2N(&self_y, self_y);
        done=true;

        (*self_).y = self_y;
        break;
      }
    }
    state++;
  }
}

fn dbl(self_: ptr<function, HighThroughput>) { // +
  var evenOdd: array<WideNumber, limbs>;
  var N: array<u32, limbs>;
  var A: array<u32, limbs>;
  var B: array<u32, limbs>;
  var T0: array<u32, limbs>;
  var T1: array<u32, limbs>;

  var state: u32 = 0u;
  var carry: bool;
  var done: bool;
  var square: bool;

  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  if (*self_).affine {
    (*self_).affine=false;
    done=false;
    setOne(&self_zz);
    setOne(&self_zzz);

    (*self_).zz = self_zz;
    (*self_).zzz = self_zzz;
  }
  else {
    done = (*self_).infinity || isZero((*self_).zz);
  }

  N[0]=0x00000001u; 
  N[1]=0x8508C000u; 
  N[2]=0x30000000u; 
  N[3]=0x170B5D44u; 
  N[4]=0xBA094800u; 
  N[5]=0x1EF3622Fu;
  N[6]=0x00F5138Fu; 
  N[7]=0x1A22D9F3u; 
  N[8]=0x6CA1493Bu; 
  N[9]=0xC63B05C0u; 
  N[10]=0x17C510EAu; 
  N[11]=0x01AE3A46u;

  square=true;
  add(&T1, self_y, self_y);

  (*self_).y = self_y;
  setPermuteLow(&A, T1);

  while (!done) {
    if !square {
      mul2(&evenOdd, &carry, A, B, N);
    } 
      
    else {
      sqr2(&evenOdd, &B, &carry, A, N);
      square=false;
    }

    switch (state) {
      case 0u, default: {
        merge(&T0, evenOdd, carry);
        setPermuteLow(&B, T0);
        break;
      }
      case 1u: {
        merge(&T1, evenOdd, carry);
        setPermuteLow(&A, self_zz);

        (*self_).zz = self_zz;
        break;
      }
      case 2u: {
        merge(&self_zz, evenOdd, carry);
        setPermuteLow(&A, T1); 
        setPermuteLow(&B, self_zzz); 

        (*self_).zz = self_zz;
        (*self_).zzz = self_zzz;
        break;
      }
      case 3u: {
        merge(&self_zzz, evenOdd, carry);
        setPermuteLow(&B, self_y);

        (*self_).y = self_y;
        (*self_).zzz = self_zzz;
        break;
      }
      case 4u: {
        merge(&self_y, evenOdd, carry);
        setPermuteLow(&A, self_x); 
        setPermuteLow(&B, T0); 

        (*self_).x = self_x;
        (*self_).y = self_y;
        break;
      }
      case 5u: {
        merge(&T0, evenOdd, carry);
        square=true;
        break;
      }
      case 6u: {
        merge(&T1, evenOdd, carry);
        square=true;
        add(&A, T1, T1);
        add(&A, A, T1);
        break;
      }
      case 7u: {
        merge(&self_x, evenOdd, carry);
        (*self_).x = self_x;
        add(&T1, T0, T0);
        sub(&self_x, self_x, T1);
        (*self_).x = self_x;
        add3N(&self_x, self_x);
        (*self_).x = self_x;
        sub(&B, T0, self_x);
        add5N(&B, B);

        (*self_).x = self_x;
        break;
      }
      case 8u: {
        merge(&T0, evenOdd, carry);
        sub(&self_y, T0, self_y);
        (*self_).y = self_y;
        add2N(&self_y, self_y);
        done=true;

        (*self_).y = self_y;
        break;
      }
    }
    state++;
  }
}

fn accumulator(self_: ptr<function, HighThroughput>) -> PointXYZZ { // +
  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  if (*self_).infinity || (!(*self_).affine && isZero((*self_).zz)) {
    setZero(&self_x);
    setZero(&self_y);
    setZero(&self_zz);
    setZero(&self_zzz);

    (*self_).x = self_x;
    (*self_).y = self_y;
    (*self_).zz = self_zz;
    (*self_).zzz = self_zzz;
  }
  else if (*self_).affine  {
    setOne(&self_zz);
    setOne(&self_zzz);

    (*self_).zz = self_zz;
    (*self_).zzz = self_zzz;
  }

  var res: PointXYZZ;
  initialize_PointXYZZ(&res, self_x, self_y, self_zz, self_zzz);

  (*self_).x = self_x;
  (*self_).y = self_y;
  (*self_).zz = self_zz;
  (*self_).zzz = self_zzz;

  return res;
}

fn normalize(self_: ptr<function, HighThroughput>) -> PointXY {
  var N: array<u32, limbs>;
  var A: array<u32, limbs>;
  var B: array<u32, limbs>;
  var T0: array<u32, limbs>;
  var T1: array<u32, limbs>;

  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  if((*self_).infinity || (!(*self_).affine && isZero((*self_).zz))) {
    setZero(&self_x);
    setZero(&self_y);

    (*self_).x = self_x;
    (*self_).y = self_y;
  }
  else if !(*self_).affine {
    fromInternal(&A, self_zzz);

    N[0]=0x00000001u; 
    N[1]=0x8508C000u; 
    N[2]=0x30000000u; 
    N[3]=0x170B5D44u; 
    N[4]=0xBA094800u; 
    N[5]=0x1EF3622Fu;
    N[6]=0x00F5138Fu; 
    N[7]=0x1A22D9F3u; 
    N[8]=0x6CA1493Bu; 
    N[9]=0xC63B05C0u; 
    N[10]=0x17C510EAu;
    N[11]=0x01AE3A46u;

    // Finds the inverse of A -- not super fast, but it is simple, and very compact

    mp_copy(&B, N);   //  A=array<u32, limbs>, B=N
    mp_zero(&T0);     //  T0=1, T1=0

    T0[0]=1u;
    mp_zero(&T1);

    var c = 0;

    while(mp_logical_or(A)!=0u) {
      if (A[0] & 0x01u)!=0u {
        if(mp_comp_gt(B, A)) {
          swap(&A, &B);
          swap(&T0, &T1);
        }
        mp_sub(limbs, &A, A, B);
        if !mp_sub_carry(limbs, &T0, T0, T1) {
          mp_add(limbs, &T0, T0, N);
        } 
      }
      mp_shift_right(&A, A, 1u);
      if (T0[0] & 0x01u)!=0u
      {
        mp_add(limbs, &T0, T0, N);
      }
      mp_shift_right(&T0, T0, 1u);
    }

    toInternal(&T0, T1);
    mul1(&self_y, self_y, T0, N);
    (*self_).y = self_y;
    reduce(&self_y, self_y);
    mul1(&T0, self_zz, T0, N);
    (*self_).zz = self_zz;
    mul1(&T1, T0, T0, N);
    mul1(&self_x, self_x, T1, N);
    (*self_).x = self_x;
    reduce(&self_x, self_x);
    (*self_).affine=true;

    (*self_).x = self_x;
    (*self_).y = self_y;
    (*self_).zz = self_zz;
    (*self_).zzz = self_zzz;
  }

  var res: PointXY;
  initialize_PointXY(&res, self_x, self_y);

  (*self_).x = self_x;
  (*self_).y = self_y;
  return res;
}

fn fromInternal_HighThroughput(self_: ptr<function, HighThroughput>) { // +
  var self_x = (*self_).x;
  var self_y = (*self_).y;
  var self_zz = (*self_).zz;
  var self_zzz = (*self_).zzz;

  fromInternal(&self_x, self_x);
  fromInternal(&self_y, self_y);
  fromInternal(&self_zz, self_zz);
  fromInternal(&self_zzz, self_zzz);

  (*self_).x = self_x;
  (*self_).y = self_y;
  (*self_).zz = self_zz;
  (*self_).zzz = self_zzz;
}

