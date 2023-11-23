
struct Fp12 {
  c0: Fp6,
  c1: Fp6
}

fn Fp12_zero() -> Fp12 {
  return Fp12(Fp6_zero(), Fp6_zero());
} 

fn Fp12_one() -> Fp12 {
  return Fp12(Fp6_one(), Fp6_zero());
} 

fn mul_by_014(c0: Fp2, c1: Fp2, c4: Fp2) -> Fp12 {
}

fn Fp12_frobenius_map(fp12: Fp12) -> Fp12 {
  var c0 = Fp6_frobenius_map(fp12.c0);
  var c1 = Fp6_frobenius_map(fp12.c1);
  
  var c1 = Fp6_mul(c1, Fp6(Fp2(Fp(array<u32,12>(
                    0xb319d465u,
                    0x07089552u,
                    0xb50a8313u,
                    0xc6695f92u,
                    0xd117228fu,
                    0x97e83cccu,
                    0xb2dc29eeu,
                    0xa35baecau,
                    0x5daace4du,
                    0x1ce393eau,
                    0xb0fb66ebu,
                    0x08f2220fu,
)),Fp(array<u32,12>(
                    0x4ce5d646u,
                    0xb2f66aadu,
                    0xfc497cecu,
                    0x5842a06bu,
                    0x2599d394u,
                    0xcf4895d4u,
                    0x40a8e8d0u,
                    0xc11b9cbau,
                    0xe5a0de89u,
                    0x2e3813cbu,
                    0x88847fafu,
                    0x110eefdau,
))),Fp2_zero(), Fp2_zero()));

  return Fp12(c0,c1);

} 

fn Fp12_mul(lhs: Fp12, rhs: Fp12) -> Fp12 {
  let aa = Fp6_mul(lhs.c0, rhs.c0);
  let bb = Fp6_mul(lhs.c1, rhs.c1);
  let o = Fp6_add(rhs.c0, rhs.c1);

  var c1 = Fp6_add(lhs.c1, lhs.c0);
  c1 = Fp6_mul(c1,o);
  c1 = Fp6_sub(c1,aa);
  c1 = Fp6_sub(c1,bb);

  var c0 = Fp6_mul_by_nonresidue(bb);
  c0 = Fp6_add(c0, aa);

  return Fp12 (c0, c1);
}

fn Fp12_add(lhs: Fp12, rhs: Fp12) -> Fp12 {
  return Fp12(Fp6_add(lhs.c0, rhs.c0), Fp6_add(lhs.c1, rhs.c1));
}

fn Fp12_sub(lhs: Fp12, rhs: Fp12) -> Fp12 {
  return Fp12(Fp6_sub(lhs.c0, rhs.c0), Fp6_sub(lhs.c1, rhs.c1));
}

fn Fp12_neg(fp12: Fp12) -> Fp12 {
  return Fp12(Fp6_neg(fp12.c0),Fp6_neg(fp12.c1));
}

