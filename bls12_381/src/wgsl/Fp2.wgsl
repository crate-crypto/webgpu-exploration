struct Fp2 {
  c0: Fp,
  c1: Fp
}

fn Fp2_one() -> Fp2 {
  return Fp2(Fp_one(), Fp_zero());
}

fn Fp2_frobenius_map(fp2: Fp2) -> Fp2{
   return Fp2_conjugate(fp2);
}

fn Fp2_conjugate(fp2: Fp2) -> Fp2 {
    return Fp2(fp2.c0, Fp_neg(fp2.c1));
}

fn Fp2_mul_by_nonresidue(fp2: Fp2) -> Fp2 {
    return Fp2(Fp_sub(fp2.c0,fp2.c1), Fp_add(fp2.c0,fp2.c1));
}

fn Fp2_lexicographically_largest(fp2: Fp2) -> u32 { 
  var is_zero = 0u;
  if fp2.c1.value[0] == 0u {
      is_zero = 1u; 
  }

  return Fp_lexicographically_largest(fp2.c1) | (Fp_lexicographically_largest(fp2.c0) & is_zero);
}

fn Fp2_add(lhs: Fp2, rhs: Fp2) -> Fp2 {
  return Fp2(Fp_add(lhs.c0, rhs.c0),Fp_add(lhs.c1, rhs.c1));
} 

fn Fp2_sub(lhs: Fp2, rhs: Fp2) -> Fp2 {
  return Fp2(Fp_sub(lhs.c0, rhs.c0),Fp_sub(lhs.c1, rhs.c1));
} 

fn Fp2_neg(fp2 : Fp2) -> Fp2 {
  return Fp2(Fp_neg(fp2.c0), Fp_neg(fp2.c1));
} 

fn Fp2_mul(lhs: Fp2, rhs: Fp2 ) -> Fp2 {
// F_{p^2} x F_{p^2} multiplication implemented with operand scanning (schoolbook)
// computes the result as:
//
//   a·b = (a_0 b_0 + a_1 b_1 β) + (a_0 b_1 + a_1 b_0)i
//
// In BLS12-381's F_{p^2}, our β is -1, so the resulting F_{p^2} element is:
//
//   c_0 = a_0 b_0 - a_1 b_1
//   c_1 = a_0 b_1 + a_1 b_0
//
// Each of these is a "sum of products", which we can compute efficiently.

  return Fp2(
    Fp2_sum_of_products(array<Fp,2>(lhs.c0, Fp_neg(lhs.c1)), array<Fp,2>(rhs.c0, rhs.c1)),
    Fp2_sum_of_products(array<Fp,2>(lhs.c0,lhs.c1), array<Fp,2>(rhs.c1, rhs.c0))
);
}


fn Fp2_square(fp2: Fp2) -> Fp2 {
// Complex squaring:
//
// v0  = c0 * c1
// c0' = (c0 + c1) * (c0 + \beta*c1) - v0 - \beta * v0
// c1' = 2 * v0
//
// In BLS12-381's F_{p^2}, our \beta is -1 so we
// can modify this formula:
//
// c0' = (c0 + c1) * (c0 - c1)
// c1' = 2 * c0 * c1

  let a = Fp_add(fp2.c0, fp2.c1);
  let b = Fp_sub(fp2.c0, fp2.c1);
  let c = Fp_add(fp2.c0, fp2.c0);

  return Fp2(
    Fp_mul(a,b),
    Fp_mul(c,fp2.c1)
  );
}

fn Fp2_invert(fp2: Fp2) -> Fp2 {
  let tmp = Fp_invert(Fp_add(square(fp2.c0),square(fp2.c1)));

  return Fp2(
    Fp_mul(fp2.c0, tmp),
    Fp_mul(fp2.c1 * Fp_invert(tmp))
);
}

fn Fp2_sqrt(fp2: Fp2) -> Fp2 {
  // need to rework
  let val = Fp2_pow_vartime(fp2,array<u32,12>(
        0xffffeaaau,
        0xee7fbfffu,
        0xac54ffffu,
        0x07aaffffu,
        0x3dac3d89u,
        0xd9cc34a8u,
        0x3ce144afu,
        0xd91dd2e1u,
        0x90d2eb35u,
        0x92c6e9edu,
        0x8e5ff9a6u,
        0x0680447au,
  )); 

  let alpha = Fp2_mul(Fp2_square(val), fp2);

  let x0 = Fp2_mul(fp2,val);

  return Fp2(
    Fp_invert(x0.c1),
    x0.c0
  );
} 

fn Fp2_zero() -> Fp2 {
  return Fp2(Fp_zero(), Fp_zero());
} 

@compute
@workgroup_size(1,1,1)
fn Fp2_neg_test() {

    let fp1 = Fp(array<u32,12>(v_indices[0], v_indices[1], v_indices[2], v_indices[3], v_indices[4], v_indices[5], v_indices[6], v_indices[7], v_indices[8], v_indices[9], v_indices[10], v_indices[11]));

    let fp2 = Fp(array<u32,12>(v_indices[12], v_indices[13], v_indices[14], v_indices[15], v_indices[16], v_indices[17], v_indices[18], v_indices[19], v_indices[20], v_indices[21], v_indices[22], v_indices[23]));


  let a = Fp2(fp1,fp2);


   let added_value = Fp2_neg(a);
    v_indices[0] = added_value.c0.value[0];
    v_indices[1] = added_value.c0.value[1];
    v_indices[2] = added_value.c0.value[2];
    v_indices[3] = added_value.c0.value[3];
    v_indices[4] = added_value.c0.value[4];
    v_indices[5] = added_value.c0.value[5];
    v_indices[6] = added_value.c0.value[6];
    v_indices[7] = added_value.c0.value[7];
    v_indices[8] = added_value.c0.value[8];
    v_indices[9] = added_value.c0.value[9];
    v_indices[10] = added_value.c0.value[10];
    v_indices[11] = added_value.c0.value[11];

    v_indices[12] = added_value.c1.value[0];
    v_indices[13] = added_value.c1.value[1];
    v_indices[14] = added_value.c1.value[2];
    v_indices[15] = added_value.c1.value[3];
    v_indices[16] = added_value.c1.value[4];
    v_indices[17] = added_value.c1.value[5];
    v_indices[18] = added_value.c1.value[6];
    v_indices[19] = added_value.c1.value[7];
    v_indices[20] = added_value.c1.value[8];
    v_indices[21] = added_value.c1.value[9];
    v_indices[22] = added_value.c1.value[10];
    v_indices[23] = added_value.c1.value[11];

}

@compute
@workgroup_size(1,1,1)
fn Fp2_add_test() {

    let fp1 = Fp(array<u32,12>(v_indices[0], v_indices[1], v_indices[2], v_indices[3], v_indices[4], v_indices[5], v_indices[6], v_indices[7], v_indices[8], v_indices[9], v_indices[10], v_indices[11]));

    let fp2 = Fp(array<u32,12>(v_indices[12], v_indices[13], v_indices[14], v_indices[15], v_indices[16], v_indices[17], v_indices[18], v_indices[19], v_indices[20], v_indices[21], v_indices[22], v_indices[23]));

    let fp3 = Fp(array<u32,12>(v_indices[24], v_indices[25], v_indices[26], v_indices[27], v_indices[28], v_indices[29], v_indices[30], v_indices[31], v_indices[32], v_indices[33], v_indices[34], v_indices[35]));
    let fp4 = Fp(array<u32,12>(v_indices[36], v_indices[37], v_indices[38], v_indices[39], v_indices[40], v_indices[41], v_indices[42], v_indices[43], v_indices[44], v_indices[45], v_indices[46], v_indices[47]));

  let a = Fp2(fp1,fp2);
  let b = Fp2(fp3,fp4);


   let added_value = Fp2_add(a, b);
    v_indices[0] = added_value.c0.value[0];
    v_indices[1] = added_value.c0.value[1];
    v_indices[2] = added_value.c0.value[2];
    v_indices[3] = added_value.c0.value[3];
    v_indices[4] = added_value.c0.value[4];
    v_indices[5] = added_value.c0.value[5];
    v_indices[6] = added_value.c0.value[6];
    v_indices[7] = added_value.c0.value[7];
    v_indices[8] = added_value.c0.value[8];
    v_indices[9] = added_value.c0.value[9];
    v_indices[10] = added_value.c0.value[10];
    v_indices[11] = added_value.c0.value[11];

    v_indices[12] = added_value.c1.value[0];
    v_indices[13] = added_value.c1.value[1];
    v_indices[14] = added_value.c1.value[2];
    v_indices[15] = added_value.c1.value[3];
    v_indices[16] = added_value.c1.value[4];
    v_indices[17] = added_value.c1.value[5];
    v_indices[18] = added_value.c1.value[6];
    v_indices[19] = added_value.c1.value[7];
    v_indices[20] = added_value.c1.value[8];
    v_indices[21] = added_value.c1.value[9];
    v_indices[22] = added_value.c1.value[10];
    v_indices[23] = added_value.c1.value[11];

}

@compute
@workgroup_size(1,1,1)
fn Fp2_sub_test() {

    let fp1 = Fp(array<u32,12>(v_indices[0], v_indices[1], v_indices[2], v_indices[3], v_indices[4], v_indices[5], v_indices[6], v_indices[7], v_indices[8], v_indices[9], v_indices[10], v_indices[11]));

    let fp2 = Fp(array<u32,12>(v_indices[12], v_indices[13], v_indices[14], v_indices[15], v_indices[16], v_indices[17], v_indices[18], v_indices[19], v_indices[20], v_indices[21], v_indices[22], v_indices[23]));

    let fp3 = Fp(array<u32,12>(v_indices[24], v_indices[25], v_indices[26], v_indices[27], v_indices[28], v_indices[29], v_indices[30], v_indices[31], v_indices[32], v_indices[33], v_indices[34], v_indices[35]));
    let fp4 = Fp(array<u32,12>(v_indices[36], v_indices[37], v_indices[38], v_indices[39], v_indices[40], v_indices[41], v_indices[42], v_indices[43], v_indices[44], v_indices[45], v_indices[46], v_indices[47]));

  let a = Fp2(fp1,fp2);
  let b = Fp2(fp3,fp4);


   let sub_value = Fp2_sub(a, b);
    v_indices[0] = sub_value.c0.value[0];
    v_indices[1] = sub_value.c0.value[1];
    v_indices[2] = sub_value.c0.value[2];
    v_indices[3] = sub_value.c0.value[3];
    v_indices[4] = sub_value.c0.value[4];
    v_indices[5] = sub_value.c0.value[5];
    v_indices[6] = sub_value.c0.value[6];
    v_indices[7] = sub_value.c0.value[7];
    v_indices[8] = sub_value.c0.value[8];
    v_indices[9] = sub_value.c0.value[9];
    v_indices[10] = sub_value.c0.value[10];
    v_indices[11] = sub_value.c0.value[11];

    v_indices[12] = sub_value.c1.value[0];
    v_indices[13] = sub_value.c1.value[1];
    v_indices[14] = sub_value.c1.value[2];
    v_indices[15] = sub_value.c1.value[3];
    v_indices[16] = sub_value.c1.value[4];
    v_indices[17] = sub_value.c1.value[5];
    v_indices[18] = sub_value.c1.value[6];
    v_indices[19] = sub_value.c1.value[7];
    v_indices[20] = sub_value.c1.value[8];
    v_indices[21] = sub_value.c1.value[9];
    v_indices[22] = sub_value.c1.value[10];
    v_indices[23] = sub_value.c1.value[11];

}

