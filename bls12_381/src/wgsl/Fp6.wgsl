//fp6

struct Fp6 {
  c0: Fp2,
  c1: Fp2,
  c2: Fp2
}

fn Fp6_zero() -> Fp6 {
  return Fp6(Fp2_zero(), Fp2_zero(), Fp2_zero());
}

fn Fp6_one() -> Fp6 {
  return Fp6(Fp2_one(), Fp2_zero(), Fp2_zero());
}

fn Fp6_frobenius_map(fp6: Fp6) -> Fp6 {
  var c0 = Fp2_frobenius_map(fp6.c0);
  var c1 = Fp2_frobenius_map(fp6.c1);
  var c2 = Fp2_frobenius_map(fp6.c2);

  // c1 = c1 * (u+1)^((p-1)/3)
  c1 = Fp2_mul(c1,
              Fp2(Fp_zero()),
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
                    0x18f02065u,
      )));

  c2 = Fp2_mul(c2,
              Fp2( Fp(array<u32,12>(
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
      ),Fp_zero())));

      return Fp6(c0,c1,c2);
  }

fn Fp6_sum_of_products(a:array<Fp,6> , b: array<Fp,6>) -> Fp {
    var u: array<u32, 12> = array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u,0u, 0u, 0u, 0u, 0u, 0u);
    var t: array<u32, 13> = array<u32, 13>(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],u[8],u[9],u[10],u[11],0u);



    var t0 = mac(t[0], a[0].value[0], b[0].value[0], 0u);
    t[0] = t0[0];
    var t1 = mac(t[1], a[0].value[0], b[0].value[1], t0[1]);
    t[1] = t1[0];
    var t2 = mac(t[2], a[0].value[0], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    var t3 = mac(t[3], a[0].value[0], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    var t4 = mac(t[4], a[0].value[0], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    var t5 = mac(t[5], a[0].value[0], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    var t6 = mac(t[6], a[0].value[0], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    var t7 = mac(t[7], a[0].value[0], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    var t8 = mac(t[8], a[0].value[0], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    var t9 = mac(t[9], a[0].value[0], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    var t10 = mac(t[10], a[0].value[0], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    var t11 = mac(t[11], a[0].value[0], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    var t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  

    var k = t[0] * INV;
    var f = mac(t[0], k, MODULUS[0], 0u);
    var r1 = mac(t[1], k, MODULUS[1], f[1]);
    var r2 = mac(t[2], k, MODULUS[2], r1[1]);
    var r3 = mac(t[3], k, MODULUS[3], r2[1]);
    var r4 = mac(t[4], k, MODULUS[4], r3[1]);
    var r5 = mac(t[5], k, MODULUS[5], r4[1]);
    var r6 = mac(t[6], k, MODULUS[6], r5[1]);
    var r7 = mac(t[7], k, MODULUS[7], r6[1]);
    var r8 = mac(t[8], k, MODULUS[8], r7[1]);
    var r9 = mac(t[9], k, MODULUS[9], r8[1]);
    var r10 = mac(t[10], k, MODULUS[10], r9[1]);
    var r11 = mac(t[11], k, MODULUS[11], r10[1]);

    var r12 = adc(t[12], 0u, r11[1]);

    
    let final_fp = Fp(array<u32,12>(
        r1[0],
        r2[0],
        r3[0],
        r4[0],
        r5[0],
        r6[0],
        r7[0],
        r8[0],
        r9[0],
        r10[0],
        r11[0],
        r12[0]
    ));

    return subtract_p(final_fp);
}


fn Fp6_mul_interleaved(a: Fp6, b: Fp6) -> Fp6 {
  let b10_p_b11 = Fp_add(b.c1.c0,f.c1.c1);
  let b10_m_b11 = Fp_sub(b.c1.c0, b.c1.c1);
  let b20_p_b21 = Fp_add(b.c2.c0, b.c2.c1);
  let b20_m_b21 = Fp_sub(b.c2.c0, b.c2.c1);

  return Fp6(
    Fp2(
      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, -a.c0.c1, a.c1.c0, -a.c1.c1, a.c2.c0, -a.c2.c1),
        array<Fp,6>(b.c0.c0, b.c0.c1, b20_m_b21, b20_p_b21, b10_m_b11, b10_p_b11),
      ),
      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, a.c0.c1, a.c1.c0, a.c1.c1, a.c2.c0, a.c2.c1),
        array<Fp,6>(b.c0.c0, b.c0.c1, b20_p_b21, b20_m_b21, b10_p_b11, b10_m_b11),
      ),
    ),

    Fp2(
      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, -a.c0.c1, a.c1.c0, -a.c1.c1, a.c2.c0, -a.c2.c1),
        array<Fp,6>(b.c1.c0, b.c1.c1, b.c0.c0, b.c0.c1, b20_m_b21, b20_p_b21),
      ),
      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, a.c0.c1, a.c1.c0, a.c1.c1, a.c2.c0, a.c2.c1),
        array<Fp,6>(b.c1.c0, b.c1.c1, b.c0.c1, b.c0.c0, b10_p_b21, b10_m_b21),
      ),
    ),

    Fp2(
      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, -a.c0.c1, a.c1.c0, -a.c1.c1, a.c2.c0, -a.c2.c1),
        array<Fp,6>(b.c2.c0, b.c2.c1, b.c1.c0, b.c1.c1, b.c0.c0, b.c0.c1),
      ),
      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, a.c0.c1, a.c1.c0, a.c1.c1, a.c2.c0, a.c2.c1),
        array<Fp,6>(b.c2.c1, b.c2.c0, b.c1.c1, b.c1.c0, b.c0.c1, b.c0.c0),
      ),
    ),
  );
}

fn Fp6_add(lhs: Fp6, rhs: Fp6) -> Fp6 {
    return Fp6(
        Fp2_add(lhs.c0, rhs.c0),
        Fp2_add(lhs.c1, rhs.c1),
        Fp2_add(lhs.c2, rhs.c2)
    );
}

fn Fp6_sub(lhs: Fp6, rhs: Fp6) -> Fp6 {
    return Fp6(
        Fp2_sub(lhs.c0, rhs.c0),
        Fp2_sub(lhs.c1, rhs.c1),
        Fp2_sub(lhs.c2, rhs.c2)
    );
}

fn Fp6_mul(lhs: Fp6, rhs: Fp6) -> Fp6 {
  return Fp6_mul_interleaved(lhs, rhs);
}

fn Fp6_invert(fp6: Fp6) -> Fp6 {
    var c0 = Fp2_mul_by_nonresidue(Fp2_mul(fp6.c0,fp6.c1));
    c0 = Fp2_sub(Fp2_square(fp6.c0),c0);
    
    var c1 = Fp2_mul_by_nonresidue(Fp2_square(fp2.c2));
    c1 = Fp2_sub(c1, Fp2_mul(fp6.c0, fp6.c1));
    
    var c2 = Fp2_square(fp2.c1);
    c2 = Fp2_sub(c2, Fp2_mul(fp6.c0, fp6.c2));

    var tmp = Fp2_mul_by_nonresidue(Fp2_add(Fp2_mul(fp2.c1,c2), Fp2_mul(fp2.c2,c1)));
    tmp = Fp2_invert(Fp2_sum(tmp, Fp2_mul(fp6.c0,c0)));

    return Fp6 (
      Fp2_mul(t,c0),
      Fp2_mul(t,c1),
      Fp2_mul(t,c2),
    );
}

