@compute
@workgroup_size(1)
fn computeNP0_test() {
  v_indices[0] = computeNP0(v_indices[0]);
}

@compute
@workgroup_size(1)
fn mp_zero_test() {
  var x = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  
  mp_zero(&x);

  for(var i = 0; i < i32(limbs); i++) {
    v_indices[i] = x[i];
  }
}

@compute
@workgroup_size(1)
fn mp_copy_test() {
  var dst = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var x = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  
  mp_copy(&dst, x);

  for(var i = 0; i < i32(limbs); i++) {
    v_indices[i] = dst[i];
  }
}

@compute
@workgroup_size(1)
fn mp_logical_or_test() {
  var x = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  
  v_indices[0] = mp_logical_or(x);
}

@compute
@workgroup_size(1)
fn mp_shift_right_test() {
  var r = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var x = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  
  mp_shift_right(&r, x, 1u);

  for(var i = 0; i < i32(limbs); i++) {
    v_indices[i] = r[i];
  }
}

@compute
@workgroup_size(1)
fn mp_shift_left_test() {
  var r = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var x = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  
  mp_shift_left(&r, x, 1u);

  for(var i = 0; i < i32(limbs); i++) {
    v_indices[i] = r[i];
  }
}

@compute
@workgroup_size(1)
fn mp_add_test() {
  var r = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var a = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var b = array<u32, limbs>(0u, 100u,200u,300u,400u,500u,600u,700u,800u,900u,1000u,1100u);
  
  mp_add(limbs, &r, a, b);

  for(var i = 0; i < i32(limbs); i++) {
    v_indices[i] = r[i];
  }
}

@compute
@workgroup_size(1)
fn mp_add_carry_test() {
  var r = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var a = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var b = array<u32, limbs>(0u, 100u,200u,300u,400u,500u,600u,700u,800u,900u,1000u,1100u);
  
  var carry = mp_add_carry(&r, a, b);

  for(var i = 0; i < i32(limbs); i++) {
    if i != 0 {
      v_indices[i] = r[i];
    }
    else {
      v_indices[i] = u32(carry);
    }
  }
}

@compute
@workgroup_size(1)
fn mp_sub_test() {
  var r = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var a = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var b = array<u32, limbs>(0u, 100u,200u,300u,400u,500u,600u,700u,800u,900u,1000u,1100u);
  
  mp_sub(limbs, &r, b, a);

  for(var i = 0; i < i32(limbs); i++) {
    v_indices[i] = r[i];
  }
}

@compute
@workgroup_size(1)
fn mp_sub_carry_test() {
  var r = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var a = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var b = array<u32, limbs>(0u, 100u,200u,300u,400u,500u,600u,700u,800u,900u,1000u,1100u);
  
  var carry = mp_sub_carry(limbs, &r, b, a);

  for(var i = 0; i < i32(limbs); i++) {
    if i != 0 {
      v_indices[i] = r[i];
    }
    else {
      v_indices[i] = u32(carry);
    }
  }
}

@compute
@workgroup_size(1)
fn mp_comp_eq_test() {
  var a = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var b = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  
  v_indices[0] = u32(mp_comp_eq(b, a));
}

@compute
@workgroup_size(1)
fn mp_comp_ge_test() {
  var a = array<u32, limbs>(1u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var b = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  
  v_indices[0] = u32(mp_comp_ge(b, a));
}

@compute
@workgroup_size(1)
fn mp_comp_gt_test() {
  var a = array<u32, limbs>(1u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var b = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  
  v_indices[0] = u32(mp_comp_gt(b, a));
}

@compute
@workgroup_size(1)
fn mp_select_test() {
  var r = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var a = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var b = array<u32, limbs>(0u, 100u,200u,300u,400u,500u,600u,700u,800u,900u,1000u,1100u);
  
  mp_select(&r, true, a, b);


  for(var i = 0; i < i32(limbs); i++) {
    v_indices[i] = r[i];
  }
}

@compute
@workgroup_size(1)
fn mp_mul_red_cl_test() {
  var evenOdd1: WideNumber;
  var evenOdd2: WideNumber;
  var evenOdd3: WideNumber;
  var evenOdd4: WideNumber;
  var evenOdd5: WideNumber;
  var evenOdd6: WideNumber;
  var evenOdd7: WideNumber;
  var evenOdd8: WideNumber;
  var evenOdd9: WideNumber;
  var evenOdd10: WideNumber;
  var evenOdd11: WideNumber;
  var evenOdd12: WideNumber;

  var evenOdd = array<WideNumber, limbs>(evenOdd1, evenOdd2, evenOdd3,evenOdd4,evenOdd5,evenOdd6,evenOdd7,evenOdd8,evenOdd9,evenOdd10,evenOdd11,evenOdd12);

  var n = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var a = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var b = array<u32, limbs>(0u, 100u,200u,300u,400u,500u,600u,700u,800u,900u,1000u,1100u);
  
  v_indices[0] = u32(mp_mul_red_cl(&evenOdd, a, b, n));
}

@compute
@workgroup_size(1)
fn mp_sqr_red_cl_test() {
  var evenOdd1: WideNumber;
  var evenOdd2: WideNumber;
  var evenOdd3: WideNumber;
  var evenOdd4: WideNumber;
  var evenOdd5: WideNumber;
  var evenOdd6: WideNumber;
  var evenOdd7: WideNumber;
  var evenOdd8: WideNumber;
  var evenOdd9: WideNumber;
  var evenOdd10: WideNumber;
  var evenOdd11: WideNumber;
  var evenOdd12: WideNumber;

  var evenOdd = array<WideNumber, limbs>(evenOdd1, evenOdd2, evenOdd3,evenOdd4,evenOdd5,evenOdd6,evenOdd7,evenOdd8,evenOdd9,evenOdd10,evenOdd11,evenOdd12);

  var n = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var a = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var temp = array<u32, limbs>(0u, 100u,200u,300u,400u,500u,600u,700u,800u,900u,1000u,1100u);
  
  mp_sqr_red_cl(&evenOdd, &temp, a, n);

  for(var i = 0; i < i32(limbs); i++) {
    v_indices[i] = evenOdd[i].first;
  }
  for(var i = 0; i < i32(limbs); i++) {
    v_indices[i+12] = evenOdd[i].second;
  }
}

@compute
@workgroup_size(1)
fn mp_merge_cl_test() {
  var evenOdd1: WideNumber = WideNumber(100u, 100u);
  var evenOdd2: WideNumber= WideNumber(100u, 100u);
  var evenOdd3: WideNumber= WideNumber(100u, 100u);
  var evenOdd4: WideNumber= WideNumber(100u, 100u);
  var evenOdd5: WideNumber= WideNumber(100u, 100u);
  var evenOdd6: WideNumber= WideNumber(100u, 100u);
  var evenOdd7: WideNumber= WideNumber(100u, 100u);
  var evenOdd8: WideNumber= WideNumber(100u, 100u);
  var evenOdd9: WideNumber= WideNumber(100u, 100u);
  var evenOdd10: WideNumber= WideNumber(100u, 100u);
  var evenOdd11: WideNumber= WideNumber(100u, 100u);
  var evenOdd12: WideNumber= WideNumber(100u, 100u);

  var evenOdd = array<WideNumber, limbs>(evenOdd1, evenOdd2, evenOdd3,evenOdd4,evenOdd5,evenOdd6,evenOdd7,evenOdd8,evenOdd9,evenOdd10,evenOdd11,evenOdd12);

  var r = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  
  mp_merge_cl(&r, evenOdd, true);

  for(var i = 0; i < i32(limbs); i++) {
    v_indices[i] = r[i];
  }
}