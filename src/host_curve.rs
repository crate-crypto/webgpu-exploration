#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

use serde::Deserialize;

const NP0: u32 = 0xFFFFFFFF;

const N: [u32; 12] = [
  0x00000001, 0x8508C000, 0x30000000, 0x170B5D44, 0xBA094800, 0x1EF3622F,
  0x00F5138F, 0x1A22D9F3, 0x6CA1493B, 0xC63B05C0, 0x17C510EA, 0x01AE3A46,
];

const R: [u32; 12] = [
  0xFFFFFF68, 0x02CDFFFF, 0x7FFFFFB1, 0x51409F83, 0x8A7D3FF2, 0x9F7DB3A9,     // 2^384 mod N
  0x6E7C6305, 0x7B4E97B7, 0x803C84E8, 0x4CF495BF, 0xE2FDF49A, 0x008D6661,
];

const RCubed: [u32; 12] = [
  0x8815DE20, 0x581F532F, 0xBE329585, 0xE50F4148, 0x0449F513, 0x2BE8B118,     // 2^(384*3) mod N
  0xC804A20E, 0x6A2A9516, 0x13590CB9, 0x3F725407, 0xC0E7DDA5, 0x01065AB4,
];

pub fn setZero(r: &mut [u32; 12]) {
  for i in 0..12 {
    r[i]=0;
  }
}

pub fn setOne(r: &mut [u32; 12]) {
  for i in 0..12 {
    if i == 0 {
      r[i]=1;
    }
    else {
      r[i]=0;
    }
  }
}

pub fn setR(r: &mut [u32; 12]) {
  for i in 0..12 {
    r[i]=R[i];
  }
}

pub fn set(r: &mut [u32; 12], field: [u32; 12]) {
  for i in 0..12 {
    r[i]=field[i];
  }
}

pub fn load(field: &mut [u32; 12], ptr: &[u32]) {
  for i in 0..12 {
    field[i]=ptr[i];
  }
}

pub fn store(ptr: &mut [u32], field: [u32; 12]) {
  for i in 0..12 {
    ptr[i]=field[i];
  }
}

pub fn exportField(ptr: &mut [u64], field: [u32; 12]) {
  for i in 0..6 {
    ptr[i]=(field[i*2+1].wrapping_shl(32)) as u64 | (field[i*2]) as u64;
  }
}

pub fn isZero(field: [u32; 12]) -> bool {
  for i in 0..12 {
    if field[i]!=0 {
      return false; 
    }
  }
  return true;
}

pub fn isGE(a: [u32; 12], b: [u32; 12]) -> bool {
  let mut acc: i64 = 0;

  for i in 0..12 {
    acc = acc.wrapping_add((a[i] as u64).wrapping_sub(b[i] as u64) as i64);
    acc=acc.wrapping_shr(32);
  }

  return acc>=0;
}

pub fn addN(r: &mut[u32; 12], field: [u32; 12]) {
  let mut acc: i64 = 0;

  for i in 0..12 {
    acc = acc.wrapping_add((field[i] as u64).wrapping_add(N[i] as u64) as i64);
    r[i] = acc as u32;
    acc=acc.wrapping_shr(32);
  }
}

pub fn subN(r: &mut[u32; 12], field: [u32; 12]) -> bool {
  let mut acc: i64 = 0;

  for i in 0..12 {
    acc = acc.wrapping_add((field[i] as u64).wrapping_sub(N[i] as u64) as i64);
    r[i] = acc as u32;
    acc=acc.wrapping_shr(32);
  }

  return acc>=0;
}

pub fn add(r: &mut[u32; 12], a: [u32; 12], b: [u32; 12]) {
  let mut acc: i64 = 0;

  for i in 0..12 {
    acc = acc.wrapping_add((a[i] as u64).wrapping_add(b[i] as u64).wrapping_sub(N[i] as u64) as i64);
    r[i] = acc as u32;
    acc=acc.wrapping_shr(32);
  }

  if acc>=0 {
    return;
  }
  addN(r, *r);
}

pub fn sub(r: &mut[u32; 12], a: [u32; 12], b: [u32; 12]) {
  let mut acc: i64 = 0;

  for i in 0..12 {
    acc = acc.wrapping_add((a[i] as u64).wrapping_sub(b[i] as u64) as i64);
    r[i] = acc as u32;
    acc=acc.wrapping_shr(32);
  }

  if acc>=0 {
    return;
  }
  addN(r, *r);
}

pub fn mul(r: &mut[u32; 12], a: [u32; 12], b: [u32; 12]) {
  let mut acc: u64;
  let mut high: u64 = 0;
  let mut q: u64;

  let mut res: [u32; 12] = [0; 12];

  for i in 0..12 {
    res[i]=0;
  }

  for j in 0..12 {
    acc=0;
    for i in 0..12 { 
      acc = acc.wrapping_add((a[j] as u64).wrapping_mul(b[i] as u64).wrapping_add(res[i] as u64));
      res[i] = acc as u32;
      acc=acc.wrapping_shr(32);
    }

    high=high.wrapping_add(acc);
    q = res[0].wrapping_mul(NP0) as u64;
    acc = q.wrapping_mul(N[0] as u64).wrapping_add(res[0] as u64);
    acc=acc.wrapping_shr(32);

    for i in 1..12 { 
      acc = acc.wrapping_add(q.wrapping_mul(N[i] as u64).wrapping_add(res[i] as u64));
      res[i - 1] = acc as u32;
      acc=acc.wrapping_shr(32);
    }
    high=high.wrapping_add(acc);
    res[11]=high as u32;
    high = high.wrapping_shr(32);
  }

  if high!=0 || isGE(res, N) {
    subN(r, res);
  }
  else{
    set(r, res);
  }
}

pub fn shiftRight(r: &mut[u32; 12], field: [u32; 12], mut bits: u32) {
  let words: u32 = bits.wrapping_shr(5);
  let left: u32;

  if words>0 {
    for i in 0..12 {
      if i+words<12 {
        r[i as usize]=field[(i+words) as usize];
      }
      else {
        r[i as usize]=0;
      }
    }

    bits=bits.wrapping_sub(words.wrapping_mul(32));
  }
  else {
    for i in 0..12 { 
      r[i]=field[i];
    }
  }

  if bits==0 {
    return;
  }
  left=32-bits;
  for i in 0..11 { 
    r[i]=(r[i].wrapping_shr(bits)) | (r[i+1].wrapping_shl(left));
  }
  r[11]=r[11].wrapping_shr(bits);
}

pub fn swap(a: &mut[u32; 12], b: &mut[u32; 12]) {
  let mut swap: u32;

  for i in 0..12 { 
    swap=a[i];
    a[i]=b[i];
    b[i]=swap;
  }
}

pub fn reduce(r: &mut[u32; 12], field: [u32; 12]) {
  set(r, field);
  while isGE(*r, N) {
    subN(r, *r);
  }
}

fn print(field: [u32; 12]) {
  for i in (0..=11).rev() {
    println!("{:08X}", field[i]);
  }
}

pub fn inverse(r: &mut[u32; 12], field: [u32; 12]) {
  let mut A: [u32; 12] = [0; 12];
  let mut B: [u32; 12]= [0; 12];
  let mut X: [u32; 12]= [0; 12];
  let mut Y: [u32; 12]= [0; 12];
  
  set(&mut A, field);
  set(&mut B, N);
  setOne(&mut X);
  setZero(&mut Y); 

  let mut A_copy;
  let mut X_copy;

  while !isZero(A) {
    if (A[0] & 0x01)!=0 {
      if !isGE(A, B) {
        swap(&mut A, &mut B);
        swap(&mut X, &mut Y);
      }

      A_copy = A;
      sub(&mut A, A_copy, B);

      X_copy = X;
      sub(&mut X, X_copy, Y);
    }

    A_copy = A;
    shiftRight(&mut A, A_copy, 1);

    if (X[0] & 0x01)!=0 {
      X_copy = X;
      addN(&mut X, X_copy);
    }
    X_copy = X;
    shiftRight(&mut X, X_copy, 1);
  }

  mul(r, Y, RCubed); 
}

fn dump(field: [u32; 12]) {
  let mut local: [u32; 12] = [0; 12];

  setOne(&mut local);
  let local_copy = local;
  mul(&mut local, local_copy, field);
  for i in (0..=11).rev() {
    println!("{:08X}", local[i]);
  }
}

fn negateAdd4N(field: &mut [u32; 12], a: &[u32; 12]) {
  let mut local4N: [u32; 12] = [0; 12];

  local4N[0]  = 0x00000004;
  local4N[1]  = 0x14230000;
  local4N[2]  = 0xC0000002;
  local4N[3]  = 0x5C2D7510;
  local4N[4]  = 0xE8252000;
  local4N[5]  = 0x7BCD88BE;
  local4N[6]  = 0x03D44E3C;
  local4N[7]  = 0x688B67CC;
  local4N[8]  = 0xB28524EC;
  local4N[9]  = 0x18EC1701;
  local4N[10] = 0x5F1443AB;
  local4N[11] = 0x06B8E918;

  sub(field, *a, local4N);
}

#[derive(Default, Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct PointXYZZ{
  pub x: [u32; 12],
  pub y: [u32; 12],
  pub zz: [u32; 12],
  pub zzz: [u32; 12],
}

impl PointXYZZ {
  pub fn reduce(&mut self) {
    let x_copy = self.x;
    let y_copy = self.y;
    let zz_copy = self.zz;
    let zzz_copy = self.zzz;
    reduce(&mut self.x, x_copy);
    reduce(&mut self.y, y_copy);
    reduce(&mut self.zz, zz_copy);
    reduce(&mut self.zzz, zzz_copy);
  }

  pub fn initialize(&mut self, xValue: [u32; 12], yValue: [u32; 12], zzValue: [u32; 12], zzzValue: [u32; 12]) {
    set(&mut self.x, xValue);
    set(&mut self.y, yValue);
    set(&mut self.zz, zzValue);
    set(&mut self.zzz, zzzValue);
  }

  pub fn load(&mut self, ptr: &[u32]) {
    load(&mut self.x, ptr);
    load(&mut self.y, &ptr[12..]);
    load(&mut self.zz, &ptr[24..]);
    load(&mut self.zzz, &ptr[36..]);
    self.reduce();
  }

  pub fn store(self, ptr: &mut [u32]) {
    store(ptr, self.x);
    store(&mut ptr[12..], self.y);
    store(&mut ptr[24..], self.zz);
    store(&mut ptr[36..], self.zzz);
  }

  pub fn normalize(&mut self) {
    let mut I: [u32; 12] = [0; 12];

    if isZero(self.zz) {
      setZero(&mut self.x);
      setZero(&mut self.y);
      setZero(&mut self.zzz);
      return;
    }

    inverse(&mut I, self.zzz);
    let y_copy = self.y;
    mul(&mut self.y, y_copy, I);
    let I_copy = I;
    mul(&mut I, I_copy, self.zz);
    let I_copy = I;
    mul(&mut I, I_copy, I_copy);
    let x_copy = self.x;
    mul(&mut self.x, x_copy, I);
    setR(&mut self.zz);
    setR(&mut self.zzz);
  }

  pub fn dump(self) {
    print!(" x = ");
    dump(self.x);
    print!(" y = ");
    dump(self.y);
    print!(" zz = ");
    dump(self.zz);
    print!(" zzz = ");
    dump(self.zzz);
  }

  pub fn negate_PointXYZZ(&mut self) {
    let a = self.y.clone();

    negateAdd4N(&mut self.y, &a);
  }
}

#[derive(Default, Debug, Deserialize, PartialEq)]
pub struct AccumulatorXYZZ {
  pub xyzz: PointXYZZ,
}

impl AccumulatorXYZZ {
  pub fn accumulator(&mut self) -> PointXYZZ { 
    if isZero(self.xyzz.zz) {
      setZero(&mut self.xyzz.x);
      setZero(&mut self.xyzz.y);
      setZero(&mut self.xyzz.zz);
      setZero(&mut self.xyzz.zzz);
    }
    else {
      setOne(&mut self.xyzz.zz);
      setOne(&mut self.xyzz.zzz);
    }

    let mut res = PointXYZZ::default();
    res.initialize(self.xyzz.x, self.xyzz.y, self.xyzz.zz, self.xyzz.zzz);

    return res;
  }

  pub fn initialize(&mut self) {
    setZero(&mut self.xyzz.zz);
  }

  pub fn set(&mut self, x: [u32; 12], y: [u32; 12], zz: [u32; 12], zzz: [u32; 12]) {
    set(&mut self.xyzz.x, x);
    set(&mut self.xyzz.y, y);
    set(&mut self.xyzz.zz, zz);
    set(&mut self.xyzz.zzz, zzz);
  }

  pub fn setZero(&mut self) {
    setZero(&mut self.xyzz.zz);
  }

  pub fn dbl(&mut self, point: PointXYZZ) {
    let mut U: [u32; 12] = [0; 12];
    let mut V: [u32; 12] = [0; 12];
    let mut W: [u32; 12] = [0; 12];
    let mut S: [u32; 12] = [0; 12];
    let mut M: [u32; 12] = [0; 12];
    let mut T: [u32; 12] = [0; 12];
    let mut X: [u32; 12] = [0; 12];
    let mut Y: [u32; 12] = [0; 12];
    let mut ZZ: [u32; 12] = [0; 12];
    let mut ZZZ: [u32; 12] = [0; 12];

    if isZero(point.zz) {
      setZero(&mut self.xyzz.zz);
      return;
    }

    add(&mut U, point.y, point.y);
    mul(&mut V, U, U);
    mul(&mut W, U, V);
    mul(&mut ZZ, V, point.zz);
    mul(&mut ZZZ, W, point.zzz);
    mul(&mut S, point.x, V);
    mul(&mut M, point.x, point.x);
    add(&mut T, M, M);
    let M_copy = M;
    add(&mut M, T, M_copy);
    mul(&mut X, M, M);
    let X_copy = X;
    sub(&mut X, X_copy, S);
    let X_copy = X;
    sub(&mut X, X_copy, S);
    sub(&mut T, S, X);
    mul(&mut Y, M, T);
    mul(&mut T, W, point.y);
    let Y_copy = Y;
    sub(&mut Y, Y_copy, T);
    self.set(X, Y, ZZ, ZZZ);
  }

  pub fn add(&mut self, point: PointXYZZ) {
    let mut U1: [u32; 12] = [0; 12];
    let mut U2: [u32; 12] = [0; 12];
    let mut S1: [u32; 12] = [0; 12];
    let mut S2: [u32; 12] = [0; 12];
    let mut P: [u32; 12] = [0; 12];
    let mut RR: [u32; 12] = [0; 12];
    let mut PP: [u32; 12] = [0; 12];
    let mut PPP: [u32; 12] = [0; 12];
    let mut Q: [u32; 12] = [0; 12];
    let mut T: [u32; 12] = [0; 12];
    let mut X: [u32; 12] = [0; 12];
    let mut Y: [u32; 12] = [0; 12];
    let mut ZZ: [u32; 12] = [0; 12];
    let mut ZZZ: [u32; 12] = [0; 12];

    if isZero(point.zz) {
      return;
    }

    if isZero(self.xyzz.zz) {
      self.set(point.x, point.y, point.zz, point.zzz);
      return;
    }

    mul(&mut U1, self.xyzz.x, point.zz);
    mul(&mut U2, point.x, self.xyzz.zz);
    mul(&mut S1, self.xyzz.y, point.zzz);
    mul(&mut S2, point.y, self.xyzz.zzz);
    sub(&mut P, U2, U1);
    sub(&mut RR, S2, S1);

    if isZero(P) && isZero(RR) {
      self.dbl(point);
      return;
    }

    mul(&mut PP, P, P);
    mul(&mut PPP, PP, P);
    mul(&mut Q, U1, PP);
    mul(&mut X, RR, RR);
    let X_copy = X;
    sub(&mut X, X_copy, PPP);
    let X_copy = X;
    sub(&mut X, X_copy, Q);
    let X_copy = X;
    sub(&mut X, X_copy, Q);
    sub(&mut T, Q, X);
    mul(&mut Y, RR, T);
    mul(&mut T, S1, PPP);
    let Y_copy = Y;
    sub(&mut Y, Y_copy, T);
    mul(&mut ZZ, self.xyzz.zz, point.zz);
    let ZZ_copy = ZZ;
    mul(&mut ZZ, ZZ_copy, PP);
    mul(&mut ZZZ, self.xyzz.zzz, point.zzz);
    let ZZZ_copy = ZZZ;
    mul(&mut ZZZ, ZZZ_copy, PPP);
    self.set(X, Y, ZZ, ZZZ);
  }
}