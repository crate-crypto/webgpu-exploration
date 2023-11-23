struct G1Affine {
  x: Fp,
  y: Fp,
  Infinity: u32
}

struct G1Projective {
  x: Fp,
  y: Fp,
  z: Fp
} 
fn G1Projective_generator() -> G1Projective {
  return G1Projective(
  Fp(
array<u32,12>(
        0xfd530c16u,
        0x5cb38790u,
        0x9976fff5u,
        0x7817fc67u,
        0x143ba1c1u,
        0x154f95c7u,
        0xf3d0e747u,
        0xf0ae6acdu,
        0x21dbf440u,
        0xedce6eccu,
        0x9e0bfb75u,
        0x12017741u,)
),
  Fp(
array<u32,12>(
        0x0ce72271u,
        0xbaac93d5u,
        0x7918fd8eu,
        0x8c22631au,
        0x570725ceu,
        0xdd595f13u,
        0x50405194u,
        0x51ac5829u,
        0xad0059c0u,
        0x0e1c8c3fu,
        0x5008a26au,
        0x0bbc3efcu,)
),
  Fp_one()
);
}

fn G1Projective_conditional_select(a: G1Projective, b: G1Projective, choice: u32) -> G1Projective {
    return G1Projective(
      Fp_conditional_select(a.x, b.x, choice),
      Fp_conditional_select(a.y, b.y, choice),
      Fp_conditional_select(a.z, b.z, choice),
    );
}


fn G1Projective_identity() -> G1Projective {
  return G1Projective(
      Fp_zero(),   
      Fp_one(),
      Fp_zero()
  );
}

// g1 projective to g1affine

fn G1_neg(g1: G1Affine) -> G1Affine {
  return G1Affine(g1.x, Fp_conditional_select(Fp_neg(g1.y),Fp_one(),0u),0u);
}

fn Fp_mul_by_3b(a: Fp) -> Fp {

  var b = Fp_add(a,a);
  b = Fp_add(b,a);

  return Fp_add(Fp_add(b,b),b);
}

fn G1Projective_add(lhs: G1Projective, rhs: G1Projective) -> G1Projective {
  var t0 = Fp_mul(lhs.x, rhs.x);
  var t1 = Fp_mul(lhs.y, rhs.y);
  var t2 = Fp_mul(lhs.z, rhs.z);
  var t3 = Fp_add(lhs.x, lhs.y);
  var t4 = Fp_add(rhs.x, rhs.y);

  t3 = Fp_mul(t3 , t4);
  t4 = Fp_add(t0 , t1);
  t3 = Fp_sub(t3 , t4);
  t4 = Fp_add(lhs.y , lhs.z);
  var x3 = Fp_add(rhs.y , rhs.z);
  t4 = Fp_mul(t4 , x3);
  x3 = Fp_add(t1 , t2);
  t4 = Fp_sub(t4 , x3);
  x3 = Fp_add(lhs.x , lhs.z);
  var y3 = Fp_add(rhs.x , rhs.z);
  x3 = Fp_mul(x3 , y3);
  y3 = Fp_add(t0 , t2);
  y3 = Fp_sub(x3 ,y3);
  x3 = Fp_add(t0 , t0);
  t0 = Fp_add(x3 , t0);
  t2 = Fp_mul_by_3b(t2);
  var z3 = Fp_add(t1 , t2);
  t1 = Fp_sub(t1 , t2);
  y3 = Fp_mul_by_3b(y3);
  x3 = Fp_mul(t4 , y3);
  t2 = Fp_mul(t3 , t1);
  x3 = Fp_sub(t2 , x3);
  y3 = Fp_mul(y3 , t0);
  t1 = Fp_mul(t1 , z3);
  y3 = Fp_add(t1 , y3);
  t0 = Fp_mul(t0 , t3);
  z3 = Fp_mul(z3 , t4);
  z3 = Fp_add(z3 , t0);
  return G1Projective(x3,y3,z3);
}

fn G1Projective_sub(lhs: G1Projective, rhs: G1Projective) -> G1Projective {
  return G1Projective_add(lhs, G1Projective_neg(rhs));
}



fn G1Projective_add_mixed(lhs: G1Projective, rhs: G1Affine) -> G1Projective {
  var t0 = Fp_mul(lhs.x, rhs.x);
  var t1 = Fp_mul(lhs.y, rhs.y);
  var t3 = Fp_add(lhs.x, rhs.y);
  var t4 = Fp_add(lhs.x, rhs.y);

  t3 = Fp_mul(t3,t4);
  t4 = Fp_add(t0,t1);
  t3 = Fp_sub(t3,t4);
  t4 = Fp_add(t0,t1);
  t3 = Fp_sub(t3,t4);
  t4 = Fp_mul(rhs.y,lhs.z);
  t4 = Fp_add(t4, lhs.y);
  var y3 = Fp_mul(rhs.x, lhs.z);
  y3 = Fp_add(y3, lhs.x);
  var x3 = Fp_add(t0,t0);
  t0 = Fp_add(x3,t0);
  var t2 = Fp_mul_by_3b(lhs.z);
  var z3 = Fp_add(t1,t2);
  t1 = Fp_sub(t1,t2);
  y3 = Fp_mul_by_3b(y3);
  x3 = Fp_mul(t4,y3);
  t2 = Fp_mul(t3,t1);
  x3 = Fp_sub(t2,x3);
  y3 = Fp_mul(y3,t0);
  t1 = Fp_mul(t1,z3);
  y3 = Fp_add(t1,y3);
  t0 = Fp_mul(t0,t3);
  z3 = Fp_mul(z3,t4);
  z3 = Fp_add(z3,t0);

  let tmp =  G1Projective(x3,y3,z3);

  return G1Projective_conditional_select(tmp, lhs, G1_is_identity(rhs));

}

//todo
//fn G1Projective_multiply(lhs: G1Projective, rhs: G1Projective) -> G1Projective {
 // let acc = G1Projective_identity();

  //todo
//}

fn G1_sub(lhs: G1Affine, rhs: G1Projective) -> G1Projective {
  return G1Projective_add_mixed(G1Projective_neg(rhs),lhs);
}

fn G1_sub_first_Projective(lhs: G1Projective, rhs: G1Affine) -> G1Projective {
  return G1Projective_add_mixed(lhs, G1_neg(rhs));
}

fn G1_identity() -> G1Affine {
  return G1Affine(Fp_zero(), Fp_one(), 1u);
}

fn G1_generator() -> G1Affine {
  return G1Affine(
    Fp(array<u32,12>(
                    0xfd530c16u,
                    0x5cb38790u,
                    0x9976fff5u,
                    0x7817fc67u,
                    0x143ba1c1u,
                    0x154f95c7u,
                    0xf3d0e747u,
                    0xf0ae6acdu,
                    0x21dbf440u,
                    0xedce6eccu,
                    0x9e0bfb75u,
                    0x12017741u,
)),
    Fp(array<u32,12>(
                    0x0ce72271u,
                    0xbaac93d5u,
                    0x7918fd8eu,
                    0x8c22631au,
                    0x570725ceu,
                    0xdd595f13u,
                    0x50405194u,
                    0x51ac5829u,
                    0xad0059c0u,
                    0x0e1c8c3fu,
                    0x5008a26au,   
                    0x0bbc3efcu,
)),
  0u
  );
}

fn G1_is_identity(g1: G1Affine) -> u32 {
  return g1.Infinity;
}

fn G1Projective_is_identity(g1: G1Projective) -> u32 {
  return Fp_is_zero(g1.z);
}


fn G1Affine_to_G1Projective(g1: G1Affine) -> G1Projective{
  return G1Projective(
      g1.x,
      g1.y,
      Fp_conditional_select(Fp_one(), Fp_zero() ,0u)
  );
}

// use little endian
fn u64_mod2(u64: array<u32,2>) -> u32 {
    return u64[0] & 1u;
}

fn u64_shift_left_by_one(u64: array<u32,2>) -> array<u32,2> {
    let zero_index_shift_value = u64[0] & 0x80000000u;
    return array<u32,2>(u64[0] << 1u, (u64[1] << 1u) | zero_index_shift_value);
}

fn u64_shift_right_by_one(u64: array<u32,2>) -> array<u32,2> {
    let first_index_shift_value = u64[1] & 1u;
      return array<u32,2>((u64[0] >> 1u) | (first_index_shift_value << 31u),u64[1] >> 1u);
}

fn G1Projective_double(g1: G1Projective) -> G1Projective {
  var t0 = square(g1.y);
  var z3 = Fp_add(t0,t0) ;
  z3 =  Fp_add(z3,z3);
  z3 = Fp_add(z3,z3);
  var t1 = Fp_mul(g1.y,g1.z);
  var t2 = square(g1.z);
  t2 = Fp_mul_by_3b(t2);
  var x3 = Fp_mul(t2,z3);
  var y3 = Fp_add(t0,t2);
  z3 = Fp_mul(t1,z3);
  t1 = Fp_add(t2,t2);
  t2 = Fp_add(t1,t2);
  t0 = Fp_sub(t0,t2);
  y3 = Fp_mul(t0,y3);
  y3 = Fp_add(x3,y3);
  t1 = Fp_mul(g1.x,g1.y);
  x3 = Fp_mul(t0,t1);
  x3 = Fp_add(x3,x3);

  let tmp = G1Projective(x3,y3,z3);


  return G1Projective_conditional_select(tmp, G1Projective_identity(),G1Projective_is_identity(g1));
}

fn G1Projective_neg(g1: G1Projective) -> G1Projective {
  return G1Projective(
      g1.x,   
      Fp_neg(g1.y),
      g1.z
  );
}

fn G1Projective_mul_by_x(g1: G1Projective) -> G1Projective {
  var xself = G1Projective_identity();
  
  var x = u64_shift_right_by_one(BLS_X);
  var tmp = g1;
  
  while (x[0]!=0u & x[1]!=0u){
      tmp = G1Projective_double(tmp);
      
      if u64_mod2(x) == 1u {
          xself = G1Projective_add(xself,tmp);
      } 
      x = u64_shift_right_by_one(x);
  }
  
  if BLS_X_IS_NEGATIVE { 
      xself = G1Projective_neg(xself);
   }
  return xself;
}

fn G1Projective_clear_cofactor(g1: G1Projective) -> G1Projective {
  return G1Projective_sub(g1, G1Projective_mul_by_x(g1));
}

fn G1Projective_is_on_cruve(g1: G1Projective) -> u32 {
  // TODO
  return 1u;
}

fn G1_is_torsion_free(g1: G1Affine) -> u32 {
 let minus_x_squared_times_p = G1Affine_to_G1Projective(g1);
  // TODO
  return 1u;
}

// need some form of u8 representation from u32

fn G1Projective_multiply(lhs: G1Projective, rhs: array<u32,8> ) {
 let acc = G1Projective_identity();
}

