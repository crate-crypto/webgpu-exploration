// G2 Affine
struct G2Affine {
  x: Fp2, 
  y: Fp2,
  infinity: u32
}

struct G2Projective {
  x: Fp2,
  y: Fp2,
  z: Fp2
}

fn Fp2_conditional_select(a: Fp2, b: Fp2 , choice: u32) -> Fp2 {
  return Fp2(
      Fp_conditional_select(a.c0, b.c0, choice),
      Fp_conditional_select(a.c1,b.c1,choice)
  );
}

fn Fp2_is_zero(p: Fp2) -> u32{
  return u32(Fp_is_zero(p.c0) & Fp_is_zero(p.c1));
}

fn G2Affine_conditional_select(a: G2Affine, b: G2Affine, choice: u32) -> G2Affine {
  return G2Affine(
    Fp2_conditional_select(a.x, b.x,choice),
    Fp2_conditional_select(a.y, b.y,choice),
    u32_conditional_select(a.infinity, b.infinity,choice),
  );
}

fn G2Affine_from_G2Projective(p: G2Projective) -> G2Affine {
  let zinv = Fp2_invert(p.z);
  
  let x = Fp2_mul(p.x , zinv);
  let y = Fp2_mul(p.y, zinv);

  let tmp = G2Affine(x,y,0u);
  
  return G2Affine_conditional_select(tmp, G2Affine_identity(), Fp2_is_zero(zinv));
}

fn G2Affine_identity() -> G2Affine {
    return G2Affine(Fp2_zero(), Fp2_one(), 1u);
}

const B = Fp2(
  Fp(
    array<u32,12>(
        0x000cfff3u,
        0xaa270000u,
        0xfc34000au,
        0x53cc0032u,
        0x6b0a807fu,
        0x478fe97au,
        0xe6ba24d7u,
        0xb1d37ebeu,
        0xbf78ab2fu,
        0x8ec9733bu,
        0x3d83de7eu,
        0x09d64551u,) 
  ),
  Fp(
    array<u32,12>(
        0x000cfff3u,
        0xaa270000u,
        0xfc34000au,
        0x53cc0032u,
        0x6b0a807fu,
        0x478fe97au,
        0xe6ba24d7u,
        0xb1d37ebeu,
        0xbf78ab2fu,
        0x8ec9733bu,
        0x3d83de7eu,
        0x09d64551u,)
  ),
);

//const B3 = Fp2_add(B,Fp2_add(B,B));
fn G2Affine_generator() -> G2Affine {
  return G2Affine(
    Fp2(Fp(array<u32,12>(
        0x02940a10u,
        0xf5f28fa2u,
        0x87b4961au,
        0xb3f5fb26u,
        0x3e2ae580u,
        0xa1a893b5u,
        0x1a3caee9u,
        0x9894999du,
        0x1863366bu,
        0x6f67b763u,
        0x4350bcd7u,
        0x05819192u,
    )),
    Fp(array<u32,12>(
        0x9e23f606u,
        0xa5a9c075u,
        0xbccd60c3u,
        0xaaa0c59du,
        0xe2867806u,
        0x3bb17e18u,
        0x8541b367u,
        0x1b1ab6ccu,
        0xf2158547u,
        0xc2b6ed0eu,
        0x7360edf3u,
        0x11922a09u,
    ))),
    Fp2(Fp(array<u32,12>(
        0x60494c4au,
        0x4c730af8u,
        0x5e369c5au,
        0x597cfa1fu,
        0xaa0a635au,
        0xe7e6856cu,
        0x6e0d495fu,
        0xbbefb5e9u,
        0xf0ef25a2u,
        0x07d3a975u,
        0x7e80dae5u,
        0x0083fd8eu,
    )),
    Fp(array<u32,12>(
        0xdf64b05du,
        0xadc0fc92u,
        0x2b1461dcu,
        0x18aa270au,
        0x3be4eba0u,
        0x86adac6au,
        0xc93da33au,
        0x79495c4eu,
        0xa43ccaedu,
        0xe7175850u,
        0x63de1bf2u,
        0x0b2bc2a1u)
    )),
  0u
);
}

fn G2Affine_is_identity(p: G2Affine) -> u32 {
   return p.infinity;
}

fn G2Projective_psi(g2: G2Projective) -> G2Projective {

    let psi_coeff_x = Fp2(
      Fp_zero(),
      Fp(array<u32,12>(
          0x867545c3u,
          0x890dc9e4u,
          0x3285a5d5u,
          0x2af32253u,
          0x309b7e2cu,
          0x50880866u,
          0x7e881024u,
          0xa20d1b8cu,
          0xe2db9068u,
          0x14e4f04fu,
          0x1564853au,
          0x14e56d3fu,
      ))
    );
    let psi_coeff_y = Fp2(Fp(
      array<u32,12>(
          0xa55c9ad1u,
          0x3e2f585du,
          0x86c18183u,
          0x4294213du,
          0x8b623732u,
          0x382844c8u,
          0x19103e18u,
          0x92ad2afdu,
          0xac7cf0b9u,
          0x1d794e4fu,
          0x7d825ec8u,
          0x0bd592fcu)),
      Fp(array<u32,12>(
          0x5aa30fdau,
          0x7bcfa7a2u,
          0x2a927e7cu,
          0xdc17dec1u,
          0x6b4ebef1u,
          0x2f088dd8u,
          0xda74d4a7u,
          0xd1ca2087u,
          0x96cebc1du,
          0x2da25966u,
          0xbbfd87d2u,
          0x0e2b7eedu)
          ),
          );
    return G2Projective(       
      Fp2_frobenius_map(g2.x),
      Fp2_frobenius_map(g2.y),
      Fp2_frobenius_map(g2.z),
    );
}

fn G2Projective_psi2(g2: G2Projective) -> G2Projective {

let psi2_coeff_x = Fp2(
  Fp(array<u32,12>(
        0x8671f071u,
        0xcd03c9e4u,
        0x1fcda5d2u,
        0x5dab2246u,
        0xd3851b95u,
        0x587042afu,
        0x01bacb9eu,
        0x8eb60ebeu,
        0x83d050d2u,
        0x03f97d6eu,
        0x54638741u,
        0x18f02065u)),
  Fp_zero());

  return G2Projective(
    Fp2_mul(g2.x,psi2_coeff_x),
    Fp2_neg(g2.y),
    Fp2_zero()
  );
}

fn Fp_ct_eq(lhs: Fp, rhs: Fp) -> u32 {
    return 
        u32(( lhs.value[0] == rhs.value[0] ) & 
        ( lhs.value[1] ==  rhs.value[1] ) & 
        ( lhs.value[2] ==  rhs.value[2] ) & 
        ( lhs.value[3] ==  rhs.value[3] ) & 
        ( lhs.value[4] ==  rhs.value[4] ) & 
        ( lhs.value[5] ==  rhs.value[5] ) & 
        ( lhs.value[6] ==  rhs.value[6] ) & 
        ( lhs.value[7] ==  rhs.value[7] ) & 
        ( lhs.value[8] ==  rhs.value[8] ) & 
        ( lhs.value[9] ==  rhs.value[9] ) & 
        ( lhs.value[10] == rhs.value[10] ) & 
        ( lhs.value[11] ==  rhs.value[11] ));
}

fn Fp2_ct_eq(lhs: Fp2, rhs: Fp2) -> u32 {
  return u32(Fp_ct_eq(lhs.c0, rhs.c0) & Fp_ct_eq(lhs.c1, rhs.c1));
}

fn Fp6_ct_eq(lhs: Fp6, rhs: Fp6) -> u32 {
  return u32(Fp2_ct_eq(lhs.c0, rhs.c0) & Fp2_ct_eq(lhs.c1, rhs.c1) & Fp2_ct_eq(lhs.c2, rhs.c2) );
}

fn Fp12_ct_eq(lhs: Fp12, rhs: Fp12) -> u32 {
  return u32(Fp6_ct_eq(lhs.c0, rhs.c0) & Fp6_ct_eq(lhs.c1, rhs.c1));
}

fn G1Projective_ct_eq(lhs: G1Projective, rhs: G1Projective) -> u32{
  let x1 = Fp_mul(lhs.x,rhs.z);
  let x2 = Fp_mul(rhs.x,lhs.z);

  let y1 = Fp_mul(lhs.y, rhs.z);
  let y2 = Fp_mul(rhs.y, lhs.z);

  let self_is_zero = Fp_is_zero(lhs.z);
  let other_is_zero = Fp_is_zero(rhs.z);

  return u32((self_is_zero & other_is_zero) | 
  ((u32_negation(self_is_zero)) & (u32_negation(other_is_zero)) & Fp_ct_eq(x1,x2) & Fp_ct_eq(y1,y2)));
}


// util function
fn convert_to_bool(val: u32) -> bool {
  if val == 0u { 
    return false;
  }else{
    return true;
  }
}

fn u32_negation(val:u32) -> u32 {

//  if val == 0u { 
//    return 1u;
//  }else if val == 1u{
//    return 0u;
//  }
  // return None here 
// todo: do something for unsupported, maybe introduce a new kind of Option for wgsl

  if val == 0u { 
    return 1u;
  }else{
    return 0u;
  }
}

fn G2Projective_ct_eq(lhs: G2Projective, rhs: G2Projective ) -> u32 {
  let x1 = Fp2_mul(lhs.x, rhs.z); 
  let x2 = Fp2_mul(rhs.x, lhs.z);
  
  let y1 = Fp2_mul(lhs.y, rhs.z);
  let y2 = Fp2_mul(rhs.y, lhs.z);

  let self_is_zero = Fp2_is_zero(lhs.z);

  let other_is_zero = Fp2_is_zero(rhs.z);

  return u32((self_is_zero & other_is_zero) | ((u32_negation(self_is_zero)) & (u32_negation(other_is_zero)) & Fp2_ct_eq(x1,x2) & Fp2_ct_eq(y1,y2)));
}

fn G2Affine_to_G2Projective(g1: G2Affine) -> G2Projective {
  return G2Projective(
    g1.x,
    g1.y,
    Fp2_conditional_select(Fp2_one(), Fp2_zero(), g1.infinity), 
  ); 
}

fn G2Projective_identity() -> G2Projective {
  return G2Projective(
      Fp2_zero(),
      Fp2_one(),
      Fp2_zero());
}

fn Fp2_mul_by_3b(x: Fp2) -> Fp2 {
  return Fp2_mul(x, Fp2_add(Fp2_add(B,B),B));
} 

fn G2Projective_is_identity(g2: G2Projective) -> u32 {
  return Fp2_is_zero(g2.z);
}

fn G2Projective_double(g2: G2Projective) -> G2Projective {
  // implement G2Condiional select, , square, mul_by_3b
    var t0 = Fp2_square(g2.y);
    var z3 = Fp2_add(t0,t0) ;
    z3 =  Fp2_add(z3,z3);
    z3 = Fp2_add(z3,z3);
    var t1 = Fp2_mul(g2.y,g2.z);
    var t2 = Fp2_square(g2.z);
    t2 = Fp2_mul_by_3b(t2);
    var x3 = Fp2_mul(t2,z3);
    var y3 = Fp2_add(t0,t2);
    z3 = Fp2_mul(t1,z3);
    t1 = Fp2_add(t2,t2);
    t2 = Fp2_add(t1,t2);
    t0 = Fp2_sub(t0,t2);
    y3 = Fp2_mul(t0,y3);
    y3 = Fp2_add(x3,y3);
    t1 = Fp2_mul(g2.x,g2.y);
    x3 = Fp2_mul(t0,t1);
    x3 = Fp2_add(x3,x3);

    let tmp = G2Projective(x3,y3,z3);


    return G2Projective_conditional_select(tmp, G2Projective_identity(),G2Projective_is_identity(g2));
}

fn G2Projective_add(lhs : G2Projective, rhs: G2Projective ) -> G2Projective {
  var t0 = Fp2_mul(lhs.x, rhs.x);
  var t1 = Fp2_mul(lhs.y, rhs.y);
  var t2 = Fp2_mul(lhs.z, rhs.z);
  var t3 = Fp2_add(lhs.x, lhs.y);
  var t4 = Fp2_add(rhs.x, rhs.y);

  t3 = Fp2_mul(t3 , t4);
  t4 = Fp2_add(t0 , t1);
  t3 = Fp2_sub(t3 , t4);
  t4 = Fp2_add(lhs.y , lhs.z);
  var x3 = Fp2_add(rhs.y , rhs.z);
  t4 = Fp2_mul(t4 , x3);
  x3 = Fp2_add(t1 , t2);
  t4 = Fp2_sub(t4 , x3);
  x3 = Fp2_add(lhs.x , lhs.z);
  var y3 = Fp2_add(rhs.x , rhs.z);
  x3 = Fp2_mul(x3 , y3);
  y3 = Fp2_add(t0 , t2);
  y3 = Fp2_sub(x3 ,y3);
  x3 = Fp2_add(t0 , t0);
  t0 = Fp2_add(x3 , t0);
  t2 = Fp2_mul_by_3b(t2);
  var z3 = Fp2_add(t1 , t2);
  t1 = Fp2_sub(t1 , t2);
  y3 = Fp2_mul_by_3b(y3);
  x3 = Fp2_mul(t4 , y3);
  t2 = Fp2_mul(t3 , t1);
  x3 = Fp2_sub(t2 , x3);
  y3 = Fp2_mul(y3 , t0);
  t1 = Fp2_mul(t1 , z3);
  y3 = Fp2_add(t1 , y3);
  t0 = Fp2_mul(t0 , t3);
  z3 = Fp2_mul(z3 , t4);
  z3 = Fp2_add(z3 , t0);
  return G2Projective(x3,y3,z3);

}

fn G2Projective_conditional_select(a: G2Projective, b: G2Projective, choice: u32) -> G2Projective {
    return G2Projective(
      Fp2_conditional_select(a.x, b.x, choice),
      Fp2_conditional_select(a.y, b.y, choice),
      Fp2_conditional_select(a.z, b.z, choice),
    );
}

fn G2Projective_add_mixed(lhs: G2Projective, rhs: G2Affine) -> G2Projective {
  var t0 = Fp2_mul(lhs.x, rhs.x);
  var t1 = Fp2_mul(lhs.y, rhs.y);
  var t3 = Fp2_add(lhs.x, rhs.y);
  var t4 = Fp2_add(lhs.x, rhs.y);

  t3 = Fp2_mul(t3,t4);
  t4 = Fp2_add(t0,t1);
  t3 = Fp2_sub(t3,t4);
  t4 = Fp2_add(t0,t1);
  t3 = Fp2_sub(t3,t4);
  t4 = Fp2_mul(rhs.y,lhs.z);
  t4 = Fp2_add(t4, lhs.y);
  var y3 = Fp2_mul(rhs.x, lhs.z);
  y3 = Fp2_add(y3, lhs.x);
  var x3 = Fp2_add(t0,t0);
  t0 = Fp2_add(x3,t0);
  var t2 = Fp2_mul_by_3b(lhs.z);
  var z3 = Fp2_add(t1,t2);
  t1 = Fp2_sub(t1,t2);
  y3 = Fp2_mul_by_3b(y3);
  x3 = Fp2_mul(t4,y3);
  t2 = Fp2_mul(t3,t1);
  x3 = Fp2_sub(t2,x3);
  y3 = Fp2_mul(y3,t0);
  t1 = Fp2_mul(t1,z3);
  y3 = Fp2_add(t1,y3);
  t0 = Fp2_mul(t0,t3);
  z3 = Fp2_mul(z3,t4);
  z3 = Fp2_add(z3,t0);

  let tmp =  G2Projective(x3,y3,z3);

  // TODO
  //G2Projective_is_identity(rhs))

  return G2Projective_conditional_select(tmp, lhs, 0u);
}

fn G2Projective_neg(g2: G2Projective) -> G2Projective{
  return G2Projective(
    g2.x,Fp2_neg(g2.y),g2.z
  );
}

fn G2Projective_mul_by_x(a : G2Projective ) -> G2Projective {
  var xself = G2Projective_identity();

  var x = u64_shift_right_by_one(BLS_X);
  var acc = a;
  while x[0] != 0u {
    acc = G2Projective_double(acc);
    //wip
    if u64_mod2(x) == 1u {
        xself = G2Projective_add(xself, acc);
    }
    x = u64_shift_right_by_one(x);
  }
  if BLS_X_IS_NEGATIVE {
      xself = G2Projective_neg(xself);
  }
  return xself;
}

fn G2Projective_is_torsion_free(g1:G2Affine) -> u32{

    let p =  G2Affine_to_G2Projective(g1);

    return G2Projective_ct_eq(G2Projective_psi(p),G2Projective_mul_by_x(p));
}

