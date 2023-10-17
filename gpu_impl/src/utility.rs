#![allow(non_snake_case)]

use std::mem;

pub fn u64_as_slice_u32(input: &[u64]) -> &[u32] {
  const size_u64: usize = mem::size_of::<u64>();
  const size_u32: usize = mem::size_of::<u32>();

  assert!(input.len() % (size_u64/size_u32) == 0);

  let ratio = size_u64 as f32 / size_u32 as f32;

  let length = (input.len() as f32 * ratio) as usize;

  unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u32, length) }
}

pub fn u64_as_mut_slice_u32(input: &mut [u64]) -> &mut [u32] {
  const size_u64: usize = mem::size_of::<u64>();
  const size_u32: usize = mem::size_of::<u32>();

  assert!(input.len() % (size_u64/size_u32) == 0);

  let ratio = size_u64 as f32 / size_u32 as f32;

  let length = (input.len() as f32 * ratio) as usize;

  unsafe { std::slice::from_raw_parts_mut(input.as_mut_ptr() as *mut u32, length) }
}

pub fn u32_as_slice_u64(input: &[u32]) -> &[u64] {
  const size_u64: usize = mem::size_of::<u64>();
  const size_u32: usize = mem::size_of::<u32>();

  let ratio = size_u32 as f32 / size_u64 as f32;

  let length = (input.len() as f32 * ratio) as usize;

  unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u64, length) }
}

/// function for converting u32 slice into a u64 slice
/// input value:
/// input - u32 slice
/// output value:
/// u64 slice
pub fn u32_as_mut_slice_u64(input: &mut [u32]) -> &mut [u64] {
  const size_u64: usize = mem::size_of::<u64>();
  const size_u32: usize = mem::size_of::<u32>();

  let ratio = size_u32 as f32 / size_u64 as f32;

  let length = (input.len() as f32 * ratio) as usize;

  unsafe { std::slice::from_raw_parts_mut(input.as_mut_ptr() as *mut u64, length) }
}

/// function for converting u8 slice into a u32 slice
/// input value:
/// input - u8 slice
/// output value:
/// u32 slice
pub fn u8_as_mut_slice_u32(input: &mut [u8]) -> &mut [u32] {
  const size_u32: usize = mem::size_of::<u32>();
  const size_u8: usize = mem::size_of::<u8>();

  assert!(input.len() % (size_u32/size_u8) == 0);

  let ratio = size_u8 as f32 / size_u32 as f32;

  let length = (input.len() as f32 * ratio) as usize;

  unsafe { std::slice::from_raw_parts_mut(input.as_mut_ptr() as *mut u32, length) }
}

/// function for converting u8 slice into a u64 slice
/// input value:
/// input - u8 slice
/// output value:
/// u64 slice
pub fn u8_as_slice_u32(input: &[u8]) -> &[u32] {
  const size_u32: usize = mem::size_of::<u32>();
  const size_u8: usize = mem::size_of::<u8>();

  assert!(input.len() % (size_u32/size_u8) == 0);

  let ratio = size_u8 as f32 / size_u32 as f32;

  let length = (input.len() as f32 * ratio) as usize;

  unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u32, length) }
}

/// function for converting u32 slice into a u8 slice
/// input value:
/// input - u32 slice
/// output value:
/// u8 slice
pub fn u32_as_mut_slice_u8(input: &mut [u32]) -> &mut [u8] {
  const size_u32: usize = mem::size_of::<u32>();
  const size_u8: usize = mem::size_of::<u8>();

  let ratio = size_u32 as f32 / size_u8 as f32;

  let length = (input.len() as f32 * ratio) as usize;

  unsafe { std::slice::from_raw_parts_mut(input.as_mut_ptr() as *mut u8, length) }
}

pub fn u32_as_slice_u8(input: &[u32]) -> &[u8] {
  const size_u32: usize = mem::size_of::<u32>();
  const size_u8: usize = mem::size_of::<u8>();

  let ratio = size_u32 as f32 / size_u8 as f32;

  let length = (input.len() as f32 * ratio) as usize;

  unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, length) }
}

#[cfg(test)]
mod tests {

  use super::*;

  #[test]
  fn u32_as_mut_slice_u8_test() {
    let mut new_vec_u32: Vec<u32> = vec![1,2,3,4];
    {
      let new_vec_u8 = u32_as_mut_slice_u8(&mut new_vec_u32);
      assert_eq!(new_vec_u8, &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]);
    }
    assert_eq!(new_vec_u32, &[1,2,3,4]);

    {
      let new_vec_u8 = u32_as_mut_slice_u8(&mut new_vec_u32);
      new_vec_u8[0] = 7;
    }
    assert_eq!(new_vec_u32, &[7,2,3,4]);    
  }

  #[test]
  fn u32_as_slice_u8_test() {
    let new_vec_u32: Vec<u32> = vec![1,2,3,4];
    let new_vec1_u8 = u32_as_slice_u8(&new_vec_u32);
    let new_vec2_u8 = u32_as_slice_u8(&new_vec_u32);
    let new_vec3_u8 = u32_as_slice_u8(&new_vec_u32);

    assert_eq!(new_vec_u32, &[1,2,3,4]);  
    assert_eq!(new_vec1_u8, &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]);  
    assert_eq!(new_vec3_u8, new_vec2_u8);  
  }

  #[test]
  fn u8_as_mut_slice_u32_test() {
    let mut new_vec_u8: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0];
    {
      let new_vec_u32 = u8_as_mut_slice_u32(&mut new_vec_u8);
      assert_eq!(new_vec_u32, &[1, 2, 3, 4]);
    }
    assert_eq!(new_vec_u8, &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]);

    {
      let new_vec_u32 = u8_as_mut_slice_u32(&mut new_vec_u8);
      new_vec_u32[0] = 7;
      assert_eq!(new_vec_u32, &[7, 2, 3, 4]);
    }
    assert_eq!(new_vec_u8, &[7, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]);    
  }

  #[test]
  fn u8_as_slice_u32_test() {
    let new_vec_u8: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0];
    let new_vec1_u32 = u8_as_slice_u32(&new_vec_u8);
    let new_vec2_u32 = u8_as_slice_u32(&new_vec_u8);
    let new_vec3_u32 = u8_as_slice_u32(&new_vec_u8);

    assert_eq!(new_vec_u8, &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]);  
    assert_eq!(new_vec1_u32, &[1, 2, 3, 4]);  
    assert_eq!(new_vec2_u32, new_vec3_u32);  
  }

  #[test]
  fn u64_as_slice_u32_test() {
    let new_vec_u64: Vec<u64> = vec![1,2,3,4];
    let new_vec1_u32 = u64_as_slice_u32(&new_vec_u64);
    let new_vec2_u32 = u64_as_slice_u32(&new_vec_u64);
    let new_vec3_u32 = u64_as_slice_u32(&new_vec_u64);

    assert_eq!(new_vec_u64, &[1,2,3,4]);  
    assert_eq!(new_vec1_u32, &[1, 0, 2, 0, 3, 0, 4, 0]);  
    assert_eq!(new_vec2_u32, new_vec3_u32);
  }

  #[test]
  fn u32_as_slice_u64_test() {
    let new_vec_u32: Vec<u32> = vec![1, 0, 2, 0, 3, 0, 4, 0];
    let new_vec1_u64 = u32_as_slice_u64(&new_vec_u32);
    let new_vec2_u64 = u32_as_slice_u64(&new_vec_u32);
    let new_vec3_u64 = u32_as_slice_u64(&new_vec_u32);

    assert_eq!(new_vec_u32, &[1, 0, 2, 0, 3, 0, 4, 0]);  
    assert_eq!(new_vec1_u64, &[1, 2, 3, 4]);  
    assert_eq!(new_vec2_u64, new_vec3_u64);
  }

  #[test]
  fn u64_as_mut_slice_u32_test() {
    let mut new_vec_u64: Vec<u64> = vec![1, 2, 3, 4];
    {
      let new_vec_u32 = u64_as_mut_slice_u32(&mut new_vec_u64);
      assert_eq!(new_vec_u32, &[1, 0, 2, 0, 3, 0, 4, 0]);
    }
    assert_eq!(new_vec_u64, &[1, 2, 3, 4]);

    {
      let new_vec_u32 = u64_as_mut_slice_u32(&mut new_vec_u64);
      new_vec_u32[0] = 7;
      assert_eq!(new_vec_u32, &[7, 0, 2, 0, 3, 0, 4, 0]);
    }
    assert_eq!(new_vec_u64, &[7, 2, 3, 4]);    
  }
}