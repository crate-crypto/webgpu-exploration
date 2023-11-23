@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience

struct Fp {
    value: array<u32,12>
}
// BLS parameter x is -0xd201000000010000
const BLS_X = array<u32,2>(
  0x00010000u,
  0xd2010000u
);

const BLS_X_IS_NEGATIVE = false;
// p = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
// little endian
// for i in range(381,0,-16):
//     modulus.append(hex(int(bin(p)[2:][i-16 if (i-16) > 0 else 0 :i],base=2))))
// 
const MODULUS = array<u32,12>(
    0xffffaaabu,
    0xb9feffffu,
    0xb153ffffu,
    0x1eabfffeu,
    0xf6b0f624u,
    0x6730d2a0u,
    0xf38512bfu,
    0x64774b84u,
    0x434bacd7u,
    0x4b1ba7b6u,
    0x397fe69au,
    0x1a0111eau,
);


// 2 ^ 384 mod p
const R = Fp(array<u32,12>(
    0x0002fffdu,
    0x76090000u,
    0xc40c0002u,
    0xebf4000bu,
    0x53c758bau,
    0x5f489857u,
    0x70525745u,
    0x77ce5853u,
    0xa256ec6du,
    0x5c071a97u,
    0xfa80e493u,
    0x15f65ec3u
));


// 2 ^ (384*2) mod p
const R2 = Fp(array<u32,12>(
    0x1c341746u,
    0xf4df1f34u,
    0x09d104f1u,
    0x0a76e6a6u,
    0x4c95b6d5u,
    0x8de5476cu,
    0x939d83c0u,
    0x67eb88a9u,
    0xb519952du,
    0x9a793e85u,
    0x92cae3aau,
    0x11988fe5u,
));


// 2 ^ (384*3) mod p
const R3 = Fp(array<u32,12>(
    0xd94ca1e0u,
    0xed48ac6bu,
    0x03a7adf8u,
    0x315f831eu,
    0x615e29ddu,
    0x9a53352au,
    0x921e1761u,
    0x34c04e5eu,
    0x65724728u,
    0x2512d435u,
    0x91755d4du,
    0x0aa63460u,
));






// -(p^{-1} mod 2 ^ 32) mod 2 ^ 32
//  p with respect to 2 ^32 can be found with extended Eucledian algorithm
// p * p ^ {-1} + 2 ^ 32 * n' = 1 as gcd(p,2^32) = 1

// >>> gcdExtended(p,2**32)
// (1, 196611, -183218565085364330533720803313833161286265889208537591468314938759776294891409139001262666995330922939384922111)

// because inverse in montgomery form is - p ^ {-1} mod 2 ^ 32 
// >>> a = -196611

// since the modulo field 2 ^32 doesn't have negative numbers you have to get its equivalent positive representation
// i.e add by 2^32 until you get positive number 
// >>> while a < 0:
// ...     a += 2 ** 32
// ... 
// >>> a
// 4294770685
// >>> hex(a)
// '0xfffcfffd'
const INV: u32 = 0xfffcfffdu; 

fn Fp_one() -> Fp {
  return R; 
}

fn Fp_zero() -> Fp {
  return Fp(array<u32,12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u));
}

// multiply operation
fn multiply(a: u32, b: u32) -> array<u32, 2> {
  // Split a and b into lower and upper 16 bits
    let a_low: u32 = a & 0xFFFFu;
    let a_high: u32 = a >> 16u;
    let b_low: u32 = b & 0xFFFFu;
    let b_high: u32 = b >> 16u;

    //      16 16
    //    x 16 16
    //     --------
    // t2   t1  t0
    // t22  t11  X
    //-------------  
    // t3 (t1+t2) t0

    var t0 = a_low * b_low;
    var t1 = a_low * b_high + (t0 >> 16u);
    var t2 = t1 >> 16u;

    t1 = (t1 & 0xFFFFu) + a_high * b_low;
    t2 = t2 + a_high * b_high + (t1 >> 16u);


    let f = ((t1 & 0xffffu) << 16u) + (t0 & 0xffffu);

    return array<u32, 2>(f, t2);
}

// 32 bit addition
fn sum(a: u32, b: u32) -> array<u32,2> {
    let a_31bit = a & 0x7fffffffu;
    let b_31bit = b & 0x7fffffffu;

    let sum = a + b;

    let sum_31bit = a_31bit + b_31bit;

    let msb_a = a >> 31u;
    let msb_b = b >> 31u;
    let msb_sum = sum_31bit >> 31u;
    let carry = (msb_a & msb_b) | (msb_a & msb_sum) | (msb_b & msb_sum);

    return array<u32,2>(sum, carry);
}


// a + (b * c ) + carry 
// returns new result and new carry over
fn mac(a: u32, b: u32, c: u32, cm : u32) -> array<u32,2> {
    let bc_multiply = multiply(b, c);
    let bc_total = sum(bc_multiply[0], cm);


    let a_bc_sum = sum(a, bc_total[0]);

    let carry = bc_total[1] + bc_multiply[1] + a_bc_sum[1];

    return array<u32,2>(a_bc_sum[0], carry);
}

// a + b + carry
// returns the result and new carry over
fn adc(a: u32, b: u32, carry: u32) -> array<u32,2> {
    let ab = sum(a, b);
    let abc = sum(ab[0], carry);

    let ck = ab[1] + abc[1];

    return array<u32,2>(abc[0], ck);
}

// https://shorturl.at/mCIPT
// a - (b + borrow)
fn sbb(a: u32, b: u32, borrow: u32) -> array<u32,2> {
    let b32: u32 = (borrow >> 31u);
    let borr = b + b32;
    let ret = a - borr;
    var ret_borr = 0u;
    if b + (borrow >> 31u) > a {
        ret_borr = 4294967295u;
    }
    return array<u32,2>(ret, ret_borr);
}

fn Fp_add(lhs: Fp, rhs: Fp) -> Fp {
    let d0 = adc(lhs.value[0], rhs.value[0], 0u);
    let d1 = adc(lhs.value[1], rhs.value[1], d0[1]);
    let d2 = adc(lhs.value[2], rhs.value[2], d1[1]);
    let d3 = adc(lhs.value[3], rhs.value[3], d2[1]);
    let d4 = adc(lhs.value[4], rhs.value[4], d3[1]);
    let d5 = adc(lhs.value[5], rhs.value[5], d4[1]);
    let d6 = adc(lhs.value[6], rhs.value[6], d5[1]);
    let d7 = adc(lhs.value[7], rhs.value[7], d6[1]);
    let d8 = adc(lhs.value[8], rhs.value[8], d7[1]);
    let d9 = adc(lhs.value[9], rhs.value[9], d8[1]);
    let d10 = adc(lhs.value[10], rhs.value[10], d9[1]);
    let d11 = adc(lhs.value[11], rhs.value[11], d10[1]);

    let final_fp = Fp(array<u32,12>(
        d0[0],
        d1[0],
        d2[0],
        d3[0],
        d4[0],
        d5[0],
        d6[0],
        d7[0],
        d8[0],
        d9[0],
        d10[0],
        d11[0]
    ));
    // return final_fp;
    return subtract_p(final_fp);
}


fn Fp_neg(data: Fp) -> Fp {
    let r1_a = sbb(MODULUS[0], data.value[0], 0u);
    let r2_a = sbb(MODULUS[1], data.value[1], r1_a[1]);
    let r3_a = sbb(MODULUS[2], data.value[2], r2_a[1]);
    let r4_a = sbb(MODULUS[3], data.value[3], r3_a[1]);
    let r5_a = sbb(MODULUS[4], data.value[4], r4_a[1]);
    let r6_a = sbb(MODULUS[5], data.value[5], r5_a[1]);
    let r7_a = sbb(MODULUS[6], data.value[6], r6_a[1]);
    let r8_a = sbb(MODULUS[7], data.value[7], r7_a[1]);
    let r9_a = sbb(MODULUS[8], data.value[8], r8_a[1]);
    let r10_a = sbb(MODULUS[9], data.value[9], r9_a[1]);
    let r11_a = sbb(MODULUS[10], data.value[10], r10_a[1]);
    let r12_a = sbb(MODULUS[11], data.value[11], r11_a[1]);

    let m_v = (data.value[0] | data.value[1] | data.value[2] | data.value[3] | data.value[4] | data.value[5] | data.value[6] | data.value[7] | data.value[8] | data.value[9] | data.value[10] | data.value[11]);
    var mask = 0u;
    if m_v != 0u {
        // u32::MAX 
        mask = mask - 1u;
    }

    return Fp(
        array<u32,12>(
            r1_a[0] & mask,
            r2_a[0] & mask,
            r3_a[0] & mask,
            r4_a[0] & mask,
            r5_a[0] & mask,
            r6_a[0] & mask,
            r7_a[0] & mask,
            r8_a[0] & mask,
            r9_a[0] & mask,
            r10_a[0] & mask,
            r11_a[0] & mask,
            r12_a[0] & mask,
        )
    );
}

fn Fp_sub(lhs: Fp, rhs: Fp) -> Fp {
    return Fp_add(lhs, Fp_neg(rhs));
}

fn montgomery_reduce(
    // n = 12 , 2*n-1 = 23
    t0: u32,
    t1: u32,
    t2: u32,
    t3: u32,
    t4: u32,
    t5: u32,
    t6: u32,
    t7: u32,
    t8: u32,
    t9: u32,
    t10: u32,
    t11: u32,
    t12: u32,
    t13: u32,
    t14: u32,
    t15: u32,
    t16: u32,
    t17: u32,
    t18: u32,
    t19: u32,
    t20: u32,
    t21: u32,
    t22: u32,
    t23: u32,
) -> Fp {
    //for i = 0 to n-1 do
    //  ki <- (ti*N') mod b
    //  t <- t + kiNb^{i}
    // t <- t/R (will be shift right)
    // if t >= N then t <- t-N
    // return t

    var k = t0 * INV;
    var f = mac(t0, k, MODULUS[0], 0u);
    var r1 = mac(t1, k, MODULUS[1], f[1]);
    var r2 = mac(t2, k, MODULUS[2], r1[1]);
    var r3 = mac(t3, k, MODULUS[3], r2[1]);
    var r4 = mac(t4, k, MODULUS[4], r3[1]);
    var r5 = mac(t5, k, MODULUS[5], r4[1]);
    var r6 = mac(t6, k, MODULUS[6], r5[1]);
    var r7 = mac(t7, k, MODULUS[7], r6[1]);
    var r8 = mac(t8, k, MODULUS[8], r7[1]);
    var r9 = mac(t9, k, MODULUS[9], r8[1]);
    var r10 = mac(t10, k, MODULUS[10], r9[1]);
    var r11 = mac(t11, k, MODULUS[11], r10[1]);

    var r12 = adc(t12, 0u, r11[1]);
    var rem_carry = r12[1];

    k = r1[0] * INV;
    f = mac(r1[0], k, MODULUS[0], 0u);
    r2 = mac(r2[0], k, MODULUS[1], f[1]);
    r3 = mac(r3[0], k, MODULUS[2], r2[1]);
    r4 = mac(r4[0], k, MODULUS[3], r3[1]);
    r5 = mac(r5[0], k, MODULUS[4], r4[1]);
    r6 = mac(r6[0], k, MODULUS[5], r5[1]);
    r7 = mac(r7[0], k, MODULUS[6], r6[1]);
    r8 = mac(r8[0], k, MODULUS[7], r7[1]);
    r9 = mac(r9[0], k, MODULUS[8], r8[1]);
    r10 = mac(r10[0], k, MODULUS[9], r9[1]);
    r11 = mac(r11[0], k, MODULUS[10], r10[1]);
    r12 = mac(r12[0], k, MODULUS[11], r11[1]);

    var r13 = adc(t13, rem_carry, r12[1]);
    rem_carry = r13[1];

    k = r2[0] * INV;
    f = mac(r2[0], k, MODULUS[0], 0u);
    r3 = mac(r3[0], k, MODULUS[1], f[1]);
    r4 = mac(r4[0], k, MODULUS[2], r3[1]);
    r5 = mac(r5[0], k, MODULUS[3], r4[1]);
    r6 = mac(r6[0], k, MODULUS[4], r5[1]);
    r7 = mac(r7[0], k, MODULUS[5], r6[1]);
    r8 = mac(r8[0], k, MODULUS[6], r7[1]);
    r9 = mac(r9[0], k, MODULUS[7], r8[1]);
    r10 = mac(r10[0], k, MODULUS[8], r9[1]);
    r11 = mac(r11[0], k, MODULUS[9], r10[1]);
    r12 = mac(r12[0], k, MODULUS[10], r11[1]);
    r13 = mac(r13[0], k, MODULUS[11], r12[1]);

    var r14 = adc(t14, rem_carry, r13[1]);
    rem_carry = r14[1];

    k = r3[0] * INV;
    f = mac(r3[0], k, MODULUS[0], 0u);
    r4 = mac(r4[0], k, MODULUS[1], f[1]);
    r5 = mac(r5[0], k, MODULUS[2], r4[1]);
    r6 = mac(r6[0], k, MODULUS[3], r5[1]);
    r7 = mac(r7[0], k, MODULUS[4], r6[1]);
    r8 = mac(r8[0], k, MODULUS[5], r7[1]);
    r9 = mac(r9[0], k, MODULUS[6], r8[1]);
    r10 = mac(r10[0], k, MODULUS[7], r9[1]);
    r11 = mac(r11[0], k, MODULUS[8], r10[1]);
    r12 = mac(r12[0], k, MODULUS[9], r11[1]);
    r13 = mac(r13[0], k, MODULUS[10], r12[1]);
    r14 = mac(r14[0], k, MODULUS[11], r13[1]);

    var r15 = adc(t15, rem_carry, r14[1]);
    rem_carry = r15[1];

    k = r4[0] * INV;
    f = mac(r4[0], k, MODULUS[0], 0u);
    r5 = mac(r5[0], k, MODULUS[1], f[1]);
    r6 = mac(r6[0], k, MODULUS[2], r5[1]);
    r7 = mac(r7[0], k, MODULUS[3], r6[1]);
    r8 = mac(r8[0], k, MODULUS[4], r7[1]);
    r9 = mac(r9[0], k, MODULUS[5], r8[1]);
    r10 = mac(r10[0], k, MODULUS[6], r9[1]);
    r11 = mac(r11[0], k, MODULUS[7], r10[1]);
    r12 = mac(r12[0], k, MODULUS[8], r11[1]);
    r13 = mac(r13[0], k, MODULUS[9], r12[1]);
    r14 = mac(r14[0], k, MODULUS[10], r13[1]);
    r15 = mac(r15[0], k, MODULUS[11], r14[1]);

    var r16 = adc(t16, rem_carry, r15[1]);
    rem_carry = r16[1];


    k = r5[0] * INV;
    f = mac(r5[0], k, MODULUS[0], 0u);
    r6 = mac(r6[0], k, MODULUS[1], r5[1]);
    r7 = mac(r7[0], k, MODULUS[2], r6[1]);
    r8 = mac(r8[0], k, MODULUS[3], r7[1]);
    r9 = mac(r9[0], k, MODULUS[4], r8[1]);
    r10 = mac(r10[0], k, MODULUS[5], r9[1]);
    r11 = mac(r11[0], k, MODULUS[6], r10[1]);
    r12 = mac(r12[0], k, MODULUS[7], r11[1]);
    r13 = mac(r13[0], k, MODULUS[8], r12[1]);
    r14 = mac(r14[0], k, MODULUS[9], r13[1]);
    r15 = mac(r15[0], k, MODULUS[10], r14[1]);
    r16 = mac(r16[0], k, MODULUS[11], r15[1]);

    var r17 = adc(t17, rem_carry, r16[1]);
    rem_carry = r17[1];


    k = r6[0] * INV;
    f = mac(r6[0], k, MODULUS[0], 0u);
    r7 = mac(r7[0], k, MODULUS[1], r6[1]);
    r8 = mac(r8[0], k, MODULUS[2], r7[1]);
    r9 = mac(r9[0], k, MODULUS[3], r8[1]);
    r10 = mac(r10[0], k, MODULUS[4], r9[1]);
    r11 = mac(r11[0], k, MODULUS[5], r10[1]);
    r12 = mac(r12[0], k, MODULUS[6], r11[1]);
    r13 = mac(r13[0], k, MODULUS[7], r12[1]);
    r14 = mac(r14[0], k, MODULUS[8], r13[1]);
    r15 = mac(r15[0], k, MODULUS[9], r14[1]);
    r16 = mac(r16[0], k, MODULUS[10], r15[1]);
    r17 = mac(r17[0], k, MODULUS[11], r16[1]);

    var r18 = adc(t18, rem_carry, r17[1]);
    rem_carry = r18[1];

    k = r7[0] * INV;
    f = mac(r7[0], k, MODULUS[0], 0u);
    r8 = mac(r8[0], k, MODULUS[1], r7[1]);
    r9 = mac(r9[0], k, MODULUS[2], r8[1]);
    r10 = mac(r10[0], k, MODULUS[3], r9[1]);
    r11 = mac(r11[0], k, MODULUS[4], r10[1]);
    r12 = mac(r12[0], k, MODULUS[5], r11[1]);
    r13 = mac(r13[0], k, MODULUS[6], r12[1]);
    r14 = mac(r14[0], k, MODULUS[7], r13[1]);
    r15 = mac(r15[0], k, MODULUS[8], r14[1]);
    r16 = mac(r16[0], k, MODULUS[9], r15[1]);
    r17 = mac(r17[0], k, MODULUS[10], r16[1]);
    r18 = mac(r18[0], k, MODULUS[11], r17[1]);

    var r19 = adc(t19, rem_carry, r18[1]);
    rem_carry = r19[1];

    k = r8[0] * INV;
    f = mac(r8[0], k, MODULUS[0], 0u);
    r9 = mac(r9[0], k, MODULUS[1], r8[1]);
    r10 = mac(r10[0], k, MODULUS[2], r9[1]);
    r11 = mac(r11[0], k, MODULUS[3], r10[1]);
    r12 = mac(r12[0], k, MODULUS[4], r11[1]);
    r13 = mac(r13[0], k, MODULUS[5], r12[1]);
    r14 = mac(r14[0], k, MODULUS[6], r13[1]);
    r15 = mac(r15[0], k, MODULUS[7], r14[1]);
    r16 = mac(r16[0], k, MODULUS[8], r15[1]);
    r17 = mac(r17[0], k, MODULUS[9], r16[1]);
    r18 = mac(r18[0], k, MODULUS[10], r17[1]);
    r19 = mac(r19[0], k, MODULUS[11], r18[1]);

    var r20 = adc(t20, rem_carry, r19[1]);
    rem_carry = r20[1];

    k = r9[0] * INV;
    f = mac(r9[0], k, MODULUS[0], 0u);
    r10 = mac(r10[0], k, MODULUS[1], r9[1]);
    r11 = mac(r11[0], k, MODULUS[2], r10[1]);
    r12 = mac(r12[0], k, MODULUS[3], r11[1]);
    r13 = mac(r13[0], k, MODULUS[4], r12[1]);
    r14 = mac(r14[0], k, MODULUS[5], r13[1]);
    r15 = mac(r15[0], k, MODULUS[6], r14[1]);
    r16 = mac(r16[0], k, MODULUS[7], r15[1]);
    r17 = mac(r17[0], k, MODULUS[8], r16[1]);
    r18 = mac(r18[0], k, MODULUS[9], r17[1]);
    r19 = mac(r19[0], k, MODULUS[10], r18[1]);
    r20 = mac(r20[0], k, MODULUS[11], r19[1]);

    var r21 = adc(t21, rem_carry, r20[1]);
    rem_carry = r21[1];

    k = r10[0] * INV;
    f = mac(r10[0], k, MODULUS[0], 0u);
    r11 = mac(r11[0], k, MODULUS[1], r10[1]);
    r12 = mac(r12[0], k, MODULUS[2], r11[1]);
    r13 = mac(r13[0], k, MODULUS[3], r12[1]);
    r14 = mac(r14[0], k, MODULUS[4], r13[1]);
    r15 = mac(r15[0], k, MODULUS[5], r14[1]);
    r16 = mac(r16[0], k, MODULUS[6], r15[1]);
    r17 = mac(r17[0], k, MODULUS[7], r16[1]);
    r18 = mac(r18[0], k, MODULUS[8], r17[1]);
    r19 = mac(r19[0], k, MODULUS[9], r18[1]);
    r20 = mac(r20[0], k, MODULUS[10], r19[1]);
    r21 = mac(r21[0], k, MODULUS[11], r20[1]);

    var r22 = adc(t22, rem_carry, r21[1]);
    rem_carry = r22[1];


    k = r11[0] * INV;
    f = mac(r11[0], k, MODULUS[0], 0u);
    r12 = mac(r12[0], k, MODULUS[1], r11[1]);
    r13 = mac(r13[0], k, MODULUS[2], r12[1]);
    r14 = mac(r14[0], k, MODULUS[3], r13[1]);
    r15 = mac(r15[0], k, MODULUS[4], r14[1]);
    r16 = mac(r16[0], k, MODULUS[5], r15[1]);
    r17 = mac(r17[0], k, MODULUS[6], r16[1]);
    r18 = mac(r18[0], k, MODULUS[7], r17[1]);
    r19 = mac(r19[0], k, MODULUS[8], r18[1]);
    r20 = mac(r20[0], k, MODULUS[9], r19[1]);
    r21 = mac(r21[0], k, MODULUS[10], r20[1]);
    r22 = mac(r22[0], k, MODULUS[11], r21[1]);

    var r23 = adc(t23, rem_carry, r22[1]);

    // implement subtract p
    let fp_d = Fp(array<u32,12>(
        r12[0],
        r13[0],
        r14[0],
        r15[0],
        r16[0],
        r17[0],
        r18[0],
        r19[0],
        r20[0],
        r21[0],
        r22[0],
        r23[0]
    ));

    return subtract_p(fp_d);
}

fn subtract_p(data: Fp) -> Fp {

    let r1_a = sbb(data.value[0], MODULUS[0], 0u);
    let r2_a = sbb(data.value[1], MODULUS[1], r1_a[1]);
    let r3_a = sbb(data.value[2], MODULUS[2], r2_a[1]);
    let r4_a = sbb(data.value[3], MODULUS[3], r3_a[1]);
    let r5_a = sbb(data.value[4], MODULUS[4], r4_a[1]);
    let r6_a = sbb(data.value[5], MODULUS[5], r5_a[1]);
    let r7_a = sbb(data.value[6], MODULUS[6], r6_a[1]);
    let r8_a = sbb(data.value[7], MODULUS[7], r7_a[1]);
    let r9_a = sbb(data.value[8], MODULUS[8], r8_a[1]);
    let r10_a = sbb(data.value[9], MODULUS[9], r9_a[1]);
    let r11_a = sbb(data.value[10], MODULUS[10], r10_a[1]);
    let r12_a = sbb(data.value[11], MODULUS[11], r11_a[1]);

    let r0 = (data.value[0] & r12_a[1]) | (r1_a[0] & ~r12_a[1]);
    let r1 = (data.value[1] & r12_a[1]) | (r2_a[0] & ~r12_a[1]);
    let r2 = (data.value[2] & r12_a[1]) | (r3_a[0] & ~r12_a[1]);
    let r3 = (data.value[3] & r12_a[1]) | (r4_a[0] & ~r12_a[1]);
    let r4 = (data.value[4] & r12_a[1]) | (r5_a[0] & ~r12_a[1]);
    let r5 = (data.value[5] & r12_a[1]) | (r6_a[0] & ~r12_a[1]);
    let r6 = (data.value[6] & r12_a[1]) | (r7_a[0] & ~r12_a[1]);
    let r7 = (data.value[7] & r12_a[1]) | (r8_a[0] & ~r12_a[1]);
    let r8 = (data.value[8] & r12_a[1]) | (r9_a[0] & ~r12_a[1]);
    let r9 = (data.value[9] & r12_a[1]) | (r10_a[0] & ~r12_a[1]);
    let r10 = (data.value[10] & r12_a[1]) | (r11_a[0] & ~r12_a[1]);
    let r11 = (data.value[11] & r12_a[1]) | (r12_a[0] & ~r12_a[1]);

    return Fp(array<u32,12>(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11));
    // return Fp(array<u32,12>(r1_a[1], r2_a[0], r3_a[0], r4_a[0], r5_a[0], r6_a[0], r7_a[0], r8_a[0], r9_a[0], r10_a[0], r11_a[0], r12_a[0]));
}

fn pow_vartime(data: Fp, by: array<u32,12>) -> Fp {
    var res = R;
    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[0] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[1] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[2] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[3] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[4] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[4] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[5] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[6] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[6] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[7] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[8] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }


    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[9] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[10] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }

    for (var j = 32u; j > 0u; j--) {
        res = square(res);

        if ((by[11] >> j) & 1u) == 1u {
            res = Fp_mul(res, data);
        }
    }
    return res;
} 

fn Fp_sqrt(data: Fp) -> Fp {

    // p + 1 // 4
    let sqrt = pow_vartime(data, array<u32,12>(
        0xffffeaabu,
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

    return sqrt;
}

fn Fp_invert(data: Fp) -> Fp {
    // exponentiate by p-2
    let d = array<u32,12>(
        0xffffaaa9u,
        0xb9feffffu,
        0xb153ffffu,
        0x1eabfffeu,
        0xf6b0f624u,
        0x6730d2a0u,
        0xf38512bfu,
        0x64774b84u,
        0x434bacd7u,
        0x4b1ba7b6u,
        0x397fe69au,
        0x1a0111eau,
    );
    let in = pow_vartime(data, d);
    return in;
}

fn square(data: Fp) -> Fp {
    var t1 = mac(0u, data.value[0], data.value[1], 0u);
    var t2 = mac(0u, data.value[0], data.value[2], t1[1]);
    var t3 = mac(0u, data.value[0], data.value[3], t2[1]);
    var t4 = mac(0u, data.value[0], data.value[4], t3[1]);
    var t5 = mac(0u, data.value[0], data.value[5], t4[1]);
    var t6 = mac(0u, data.value[0], data.value[6], t5[1]);
    var t7 = mac(0u, data.value[0], data.value[7], t6[1]);
    var t8 = mac(0u, data.value[0], data.value[8], t7[1]);
    var t9 = mac(0u, data.value[0], data.value[9], t8[1]);
    var t10 = mac(0u, data.value[0], data.value[10], t9[1]);
    var t11 = mac(0u, data.value[0], data.value[11], t10[1]);
    var temp = t11[1];


    t3 = mac(t3[0], data.value[1], data.value[2], 0u);
    t4 = mac(t4[0], data.value[1], data.value[3], t3[1]);
    t5 = mac(t5[0], data.value[1], data.value[4], t4[1]);
    t6 = mac(t6[0], data.value[1], data.value[5], t5[1]);
    t7 = mac(t7[0], data.value[1], data.value[6], t6[1]);
    t8 = mac(t8[0], data.value[1], data.value[7], t7[1]);
    t9 = mac(t9[0], data.value[1], data.value[8], t8[1]);
    t10 = mac(t10[0], data.value[1], data.value[9], t9[1]);
    t11 = mac(t11[0], data.value[1], data.value[10], t10[1]);
    var t12 = mac(temp, data.value[1], data.value[11], t11[1]);
    temp = t12[1];

    t5 = mac(t5[0], data.value[2], data.value[3], 0u);
    t6 = mac(t6[0], data.value[2], data.value[4], t5[1]);
    t7 = mac(t7[0], data.value[2], data.value[5], t6[1]);
    t8 = mac(t8[0], data.value[2], data.value[6], t7[1]);
    t9 = mac(t9[0], data.value[2], data.value[7], t8[1]);
    t10 = mac(t10[0], data.value[2], data.value[8], t9[1]);
    t11 = mac(t11[0], data.value[2], data.value[9], t10[1]);
    t12 = mac(t12[0], data.value[2], data.value[10], t11[1]);
    var t13 = mac(temp, data.value[2], data.value[11], t12[1]);
    temp = t13[1];

    t7 = mac(t7[0], data.value[3], data.value[4], 0u);
    t8 = mac(t8[0], data.value[3], data.value[5], t7[1]);
    t9 = mac(t9[0], data.value[3], data.value[6], t8[1]);
    t10 = mac(t10[0], data.value[3], data.value[7], t9[1]);
    t11 = mac(t11[0], data.value[3], data.value[8], t10[1]);
    t12 = mac(t12[0], data.value[3], data.value[9], t11[1]);
    t13 = mac(t13[0], data.value[3], data.value[10], t12[1]);
    var t14 = mac(temp, data.value[3], data.value[11], t13[1]);
    temp = t14[1];

    t9 = mac(t9[0], data.value[4], data.value[5], t8[1]);
    t10 = mac(t10[0], data.value[4], data.value[6], t9[1]);
    t11 = mac(t11[0], data.value[4], data.value[7], t10[1]);
    t12 = mac(t12[0], data.value[4], data.value[8], t11[1]);
    t13 = mac(t13[0], data.value[4], data.value[9], t12[1]);
    t14 = mac(t14[0], data.value[4], data.value[10], t13[1]);
    var t15 = mac(temp, data.value[4], data.value[11], t14[1]);
    temp = t15[1];

    t11 = mac(t11[0], data.value[5], data.value[6], t10[1]);
    t12 = mac(t12[0], data.value[5], data.value[7], t11[1]);
    t13 = mac(t13[0], data.value[5], data.value[8], t12[1]);
    t14 = mac(t14[0], data.value[5], data.value[9], t13[1]);
    t15 = mac(t15[0], data.value[5], data.value[10], t14[1]);
    var t16 = mac(temp, data.value[5], data.value[11], t15[1]);
    temp = t16[1];

    t13 = mac(t13[0], data.value[6], data.value[7], t12[1]);
    t14 = mac(t14[0], data.value[6], data.value[8], t13[1]);
    t15 = mac(t15[0], data.value[6], data.value[9], t14[1]);
    t16 = mac(t16[0], data.value[6], data.value[10], t15[1]);
    var t17 = mac(temp, data.value[6], data.value[11], t16[1]);
    temp = t17[1];

    t15 = mac(t15[0], data.value[7], data.value[8], t14[1]);
    t16 = mac(t16[0], data.value[7], data.value[9], t15[1]);
    t17 = mac(t17[0], data.value[7], data.value[10], t16[1]);
    var t18 = mac(temp, data.value[7], data.value[11], t17[1]);
    temp = t18[1];

    t17 = mac(t17[0], data.value[7], data.value[9], t16[1]);
    t18 = mac(t18[0], data.value[7], data.value[10], t17[1]);
    var t19 = mac(temp, data.value[7], data.value[11], t18[1]);
    temp = t19[1];

    t19 = mac(t19[0], data.value[8], data.value[10], t18[1]);
    var t20 = mac(temp, data.value[8], data.value[11], t19[1]);
    var d21 = t20[1];

    var t21 = mac(t20[1], data.value[9], data.value[11], 0u);

    var d23 = t21[1] >> 31u;
    var d22 = (t21[1] << 1u) | (t21[0] >> 63u);
    d21 = (t21[0] << 1u) | (t20[0] >> 63u);
    var d20 = (t20[0] << 1u) | (t19[0] >> 63u);
    var d19 = (t19[0] << 1u) | (t18[0] >> 63u);
    var d18 = (t18[0] << 1u) | (t17[0] >> 63u);
    var d17 = (t17[0] << 1u) | (t16[0] >> 63u);
    var d16 = (t16[0] << 1u) | (t15[0] >> 63u);
    var d15 = (t15[0] << 1u) | (t14[0] >> 63u);
    var d14 = (t14[0] << 1u) | (t13[0] >> 63u);
    var d13 = (t13[0] << 1u) | (t12[0] >> 63u);
    var d12 = (t12[0] << 1u) | (t11[0] >> 63u);
    var d11 = (t11[0] << 1u) | (t10[0] >> 63u);
    var d10 = (t10[0] << 1u) | (t9[0] >> 63u);
    var d9 = (t9[0] << 1u) | (t8[0] >> 63u);
    var d8 = (t8[0] << 1u) | (t7[0] >> 63u);
    var d7 = (t7[0] << 1u) | (t6[0] >> 63u);
    var d6 = (t6[0] << 1u) | (t5[0] >> 63u);
    var d5 = (t5[0] << 1u) | (t4[0] >> 63u);
    var d4 = (t4[0] << 1u) | (t3[0] >> 63u);
    var d3 = (t3[0] << 1u) | (t2[0] >> 63u);
    var d2 = (t2[0] << 1u) | (t1[0] >> 63u);
    var d1 = (t1[0] << 1u);

    var f0 = mac(0u, data.value[0], data.value[0], 0u);
    var f1 = adc(d1, 0u, f0[1]);
    var f2 = mac(d2, data.value[1], data.value[1], f1[1]);
    var f3 = adc(d3, 0u, f2[1]);
    var f4 = mac(d4, data.value[2], data.value[2], f3[1]);
    var f5 = adc(d5, 0u, f4[1]);
    var f6 = mac(d6, data.value[3], data.value[3], f5[1]);
    var f7 = adc(d7, 0u, f6[1]);
    var f8 = mac(d8, data.value[4], data.value[4], f7[1]);
    var f9 = adc(d9, 0u, f8[1]);
    var f10 = mac(d10, data.value[5], data.value[5], f9[1]);
    var f11 = adc(d11, 0u, f10[1]);
    var f12 = mac(d12, data.value[6], data.value[6], f11[1]);
    var f13 = adc(d13, 0u, f12[1]);
    var f14 = mac(d14, data.value[7], data.value[7], f13[1]);
    var f15 = adc(d15, 0u, f14[1]);
    var f1o6 = mac(d16, data.value[8], data.value[8], f15[1]);
    var f17 = adc(d17, 0u, f1o6[1]);
    var f18 = mac(d18, data.value[9], data.value[9], f17[1]);
    var f19 = adc(d19, 0u, f18[1]);
    var f20 = mac(d20, data.value[10], data.value[10], f19[1]);
    var f21 = adc(d21, 0u, f20[1]);
    var f22 = mac(d22, data.value[11], data.value[11], f21[1]);
    var f23 = adc(d23, 0u, f22[1]);

    return montgomery_reduce(f0[0], f1[0], f2[0], f3[0], f4[0], f5[0], f6[0], f7[0], f8[0], f9[0], f10[0], f11[0], f12[0], f13[0], f14[0], f15[0], f1o6[0], f17[0], f18[0], f19[0], f20[0], f21[0], f22[0], f23[0]);
}


fn Fp_mul(lhs: Fp, rhs: Fp) -> Fp {
    var t0 = mac(0u, lhs.value[0], rhs.value[0], 0u);
    var t1 = mac(0u, lhs.value[0], rhs.value[1], t0[1]);
    var t2 = mac(0u, lhs.value[0], rhs.value[2], t1[1]);
    var t3 = mac(0u, lhs.value[0], rhs.value[3], t2[1]);
    var t4 = mac(0u, lhs.value[0], rhs.value[4], t3[1]);
    var t5 = mac(0u, lhs.value[0], rhs.value[5], t4[1]);
    var t6 = mac(0u, lhs.value[0], rhs.value[6], t5[1]);
    var t7 = mac(0u, lhs.value[0], rhs.value[7], t6[1]);
    var t8 = mac(0u, lhs.value[0], rhs.value[8], t7[1]);
    var t9 = mac(0u, lhs.value[0], rhs.value[9], t8[1]);
    var t10 = mac(0u, lhs.value[0], rhs.value[10], t9[1]);
    var t11 = mac(0u, lhs.value[0], rhs.value[11], t10[1]);
    var temp = t11[1];

    t1 = mac(t1[0], lhs.value[1], rhs.value[0], 0u);
    t2 = mac(t2[0], lhs.value[1], rhs.value[1], t1[1]);
    t3 = mac(t3[0], lhs.value[1], rhs.value[2], t2[1]);
    t4 = mac(t4[0], lhs.value[1], rhs.value[3], t3[1]);
    t5 = mac(t5[0], lhs.value[1], rhs.value[4], t4[1]);
    t6 = mac(t6[0], lhs.value[1], rhs.value[5], t5[1]);
    t7 = mac(t7[0], lhs.value[1], rhs.value[6], t6[1]);
    t8 = mac(t8[0], lhs.value[1], rhs.value[7], t7[1]);
    t9 = mac(t9[0], lhs.value[1], rhs.value[8], t8[1]);
    t10 = mac(t10[0], lhs.value[1], rhs.value[9], t9[1]);
    t11 = mac(t11[0], lhs.value[1], rhs.value[10], t10[1]);
    var t12 = mac(temp, lhs.value[1], rhs.value[11], t11[1]);
    temp = t12[1];

    t2 = mac(t2[0], lhs.value[2], rhs.value[0], 0u);
    t3 = mac(t3[0], lhs.value[2], rhs.value[1], t2[1]);
    t4 = mac(t4[0], lhs.value[2], rhs.value[2], t3[1]);
    t5 = mac(t5[0], lhs.value[2], rhs.value[3], t4[1]);
    t6 = mac(t6[0], lhs.value[2], rhs.value[4], t5[1]);
    t7 = mac(t7[0], lhs.value[2], rhs.value[5], t6[1]);
    t8 = mac(t8[0], lhs.value[2], rhs.value[6], t7[1]);
    t9 = mac(t9[0], lhs.value[2], rhs.value[7], t8[1]);
    t10 = mac(t10[0], lhs.value[2], rhs.value[8], t9[1]);
    t11 = mac(t11[0], lhs.value[2], rhs.value[9], t10[1]);
    t12 = mac(t12[0], lhs.value[2], rhs.value[10], t11[1]);
    var t13 = mac(temp, lhs.value[2], rhs.value[11], t12[1]);
    temp = t13[1];

    t3 = mac(t3[0], lhs.value[3], rhs.value[0], t2[1]);
    t4 = mac(t4[0], lhs.value[3], rhs.value[1], t3[1]);
    t5 = mac(t5[0], lhs.value[3], rhs.value[2], t4[1]);
    t6 = mac(t6[0], lhs.value[3], rhs.value[3], t5[1]);
    t7 = mac(t7[0], lhs.value[3], rhs.value[4], t6[1]);
    t8 = mac(t8[0], lhs.value[3], rhs.value[5], t7[1]);
    t9 = mac(t9[0], lhs.value[3], rhs.value[6], t8[1]);
    t10 = mac(t10[0], lhs.value[3], rhs.value[7], t9[1]);
    t11 = mac(t11[0], lhs.value[3], rhs.value[8], t10[1]);
    t12 = mac(t12[0], lhs.value[3], rhs.value[9], t11[1]);
    t13 = mac(t13[0], lhs.value[3], rhs.value[10], t12[1]);
    var t14 = mac(temp, lhs.value[3], rhs.value[11], t13[1]);
    temp = t14[1];

    t4 = mac(t4[0], lhs.value[4], rhs.value[0], 0u);
    t5 = mac(t5[0], lhs.value[4], rhs.value[1], t4[1]);
    t6 = mac(t6[0], lhs.value[4], rhs.value[2], t5[1]);
    t7 = mac(t7[0], lhs.value[4], rhs.value[3], t6[1]);
    t8 = mac(t8[0], lhs.value[4], rhs.value[4], t7[1]);
    t9 = mac(t9[0], lhs.value[4], rhs.value[5], t8[1]);
    t10 = mac(t10[0], lhs.value[4], rhs.value[6], t9[1]);
    t11 = mac(t11[1], lhs.value[4], rhs.value[7], t10[1]);
    t12 = mac(t12[1], lhs.value[4], rhs.value[8], t11[1]);
    t13 = mac(t13[1], lhs.value[4], rhs.value[9], t12[1]);
    t14 = mac(t14[1], lhs.value[4], rhs.value[10], t13[1]);
    var t15 = mac(temp, lhs.value[4], rhs.value[11], t14[1]);
    temp = t15[1];

    t5 = mac(t5[0], lhs.value[5], rhs.value[0], 0u);
    t6 = mac(t6[0], lhs.value[5], rhs.value[1], t5[1]);
    t7 = mac(t7[0], lhs.value[5], rhs.value[2], t6[1]);
    t8 = mac(t8[0], lhs.value[5], rhs.value[3], t7[1]);
    t9 = mac(t9[0], lhs.value[5], rhs.value[4], t8[1]);
    t10 = mac(t10[0], lhs.value[5], rhs.value[5], t9[1]);
    t11 = mac(t11[1], lhs.value[5], rhs.value[6], t10[1]);
    t12 = mac(t12[1], lhs.value[5], rhs.value[7], t11[1]);
    t13 = mac(t13[1], lhs.value[5], rhs.value[8], t12[1]);
    t14 = mac(t14[1], lhs.value[5], rhs.value[9], t13[1]);
    t15 = mac(t15[1], lhs.value[5], rhs.value[10], t14[1]);
    var t16 = mac(temp, lhs.value[5], rhs.value[11], t15[1]);
    temp = t16[1];

    t6 = mac(t6[0], lhs.value[6], rhs.value[0], 0u);
    t7 = mac(t7[0], lhs.value[6], rhs.value[1], t6[1]);
    t8 = mac(t8[0], lhs.value[6], rhs.value[2], t7[1]);
    t9 = mac(t9[0], lhs.value[6], rhs.value[3], t8[1]);
    t10 = mac(t10[0], lhs.value[6], rhs.value[4], t9[1]);
    t11 = mac(t11[1], lhs.value[6], rhs.value[5], t10[1]);
    t12 = mac(t12[1], lhs.value[6], rhs.value[6], t11[1]);
    t13 = mac(t13[1], lhs.value[6], rhs.value[7], t12[1]);
    t14 = mac(t14[1], lhs.value[6], rhs.value[8], t13[1]);
    t15 = mac(t15[1], lhs.value[6], rhs.value[9], t14[1]);
    t16 = mac(t16[1], lhs.value[6], rhs.value[10], t15[1]);
    var t17 = mac(temp, lhs.value[6], rhs.value[11], t16[1]);
    temp = t17[1];

    t7 = mac(t7[0], lhs.value[7], rhs.value[0], 0u);
    t8 = mac(t8[0], lhs.value[7], rhs.value[1], t7[1]);
    t9 = mac(t9[0], lhs.value[7], rhs.value[2], t8[1]);
    t10 = mac(t10[0], lhs.value[7], rhs.value[3], t9[1]);
    t11 = mac(t11[0], lhs.value[7], rhs.value[4], t10[1]);
    t12 = mac(t12[0], lhs.value[7], rhs.value[5], t11[1]);
    t13 = mac(t13[0], lhs.value[7], rhs.value[6], t12[1]);
    t14 = mac(t14[0], lhs.value[7], rhs.value[7], t13[1]);
    t15 = mac(t15[0], lhs.value[7], rhs.value[8], t14[1]);
    t16 = mac(t16[0], lhs.value[7], rhs.value[9], t15[1]);
    t17 = mac(t17[0], lhs.value[7], rhs.value[10], t16[1]);
    var t18 = mac(temp, lhs.value[7], rhs.value[11], t17[1]);
    temp = t18[1];


    t8 = mac(t8[0], lhs.value[8], rhs.value[0], 0u);
    t9 = mac(t9[0], lhs.value[8], rhs.value[1], t8[1]);
    t10 = mac(t10[0], lhs.value[8], rhs.value[2], t9[1]);
    t11 = mac(t11[0], lhs.value[8], rhs.value[3], t10[1]);
    t12 = mac(t12[0], lhs.value[8], rhs.value[4], t11[1]);
    t13 = mac(t13[0], lhs.value[8], rhs.value[5], t12[1]);
    t14 = mac(t14[0], lhs.value[8], rhs.value[6], t13[1]);
    t15 = mac(t15[0], lhs.value[8], rhs.value[7], t14[1]);
    t16 = mac(t16[0], lhs.value[8], rhs.value[8], t15[1]);
    t17 = mac(t17[0], lhs.value[8], rhs.value[9], t16[1]);
    t18 = mac(t18[0], lhs.value[8], rhs.value[10], t17[1]);
    var t19 = mac(temp, lhs.value[8], rhs.value[11], t18[1]);
    temp = t19[1];

    // Continue the pattern for t9 to t23

    t9 = mac(t9[0], lhs.value[9], rhs.value[0], 0u);
    t10 = mac(t10[0], lhs.value[9], rhs.value[1], t9[1]);
    t11 = mac(t11[0], lhs.value[9], rhs.value[2], t10[1]);
    t12 = mac(t12[0], lhs.value[9], rhs.value[3], t11[1]);
    t13 = mac(t13[0], lhs.value[9], rhs.value[4], t12[1]);
    t14 = mac(t14[0], lhs.value[9], rhs.value[5], t13[1]);
    t15 = mac(t15[0], lhs.value[9], rhs.value[6], t14[1]);
    t16 = mac(t16[0], lhs.value[9], rhs.value[7], t15[1]);
    t17 = mac(t17[0], lhs.value[9], rhs.value[8], t16[1]);
    t18 = mac(t18[0], lhs.value[9], rhs.value[9], t17[1]);
    t19 = mac(t19[0], lhs.value[9], rhs.value[10], t18[1]);
    var t20 = mac(temp, lhs.value[9], rhs.value[11], t19[1]);
    temp = t20[1];

    t10 = mac(t10[0], lhs.value[10], rhs.value[0], 0u);
    t11 = mac(t11[0], lhs.value[10], rhs.value[1], t10[1]);
    t12 = mac(t12[0], lhs.value[10], rhs.value[2], t11[1]);
    t13 = mac(t13[0], lhs.value[10], rhs.value[3], t12[1]);
    t14 = mac(t14[0], lhs.value[10], rhs.value[4], t13[1]);
    t15 = mac(t15[0], lhs.value[10], rhs.value[5], t14[1]);
    t16 = mac(t16[0], lhs.value[10], rhs.value[6], t15[1]);
    t17 = mac(t17[0], lhs.value[10], rhs.value[7], t16[1]);
    t18 = mac(t18[0], lhs.value[10], rhs.value[8], t17[1]);
    t19 = mac(t19[0], lhs.value[10], rhs.value[9], t18[1]);
    t20 = mac(t20[0], lhs.value[10], rhs.value[10], t19[1]);
    var t21 = mac(temp, lhs.value[10], rhs.value[11], t20[1]);
    temp = t21[1];

    // Continue the pattern for t11 to t23

    t11 = mac(t11[0], lhs.value[11], rhs.value[0], 0u);
    t12 = mac(t12[0], lhs.value[11], rhs.value[1], t11[1]);
    t13 = mac(t13[0], lhs.value[11], rhs.value[2], t12[1]);
    t14 = mac(t14[0], lhs.value[11], rhs.value[3], t13[1]);
    t15 = mac(t15[0], lhs.value[11], rhs.value[4], t14[1]);
    t16 = mac(t16[0], lhs.value[11], rhs.value[5], t15[1]);
    t17 = mac(t17[0], lhs.value[11], rhs.value[6], t16[1]);
    t18 = mac(t18[0], lhs.value[11], rhs.value[7], t17[1]);
    t19 = mac(t19[0], lhs.value[11], rhs.value[8], t18[1]);
    t20 = mac(t20[0], lhs.value[11], rhs.value[9], t19[1]);
    t21 = mac(t21[0], lhs.value[11], rhs.value[10], t20[1]);
    var t22 = mac(temp, lhs.value[11], rhs.value[11], t21[1]);



    return montgomery_reduce(
        t0[0],
        t1[0],
        t2[0],
        t3[0],
        t4[0],
        t5[0],
        t6[0],
        t7[0],
        t8[0],
        t9[0],
        t10[0],
        t11[0],
        t12[0],
        t13[0],
        t14[0],
        t15[0],
        t16[0],
        t17[0],
        t18[0],
        t19[0],
        t20[0],
        t21[0],
        t22[0],
        t22[1]
    );
}

fn Fp_lexicographically_largest(fp: Fp) -> u32{
  let tmp = montgomery_reduce(
    fp.value[0],
    fp.value[1],
    fp.value[2],
    fp.value[3],
    fp.value[4],
    fp.value[5],
    fp.value[6],
    fp.value[7],
    fp.value[8],
    fp.value[9],
    fp.value[10],
    fp.value[11],
    0u, 0u, 0u, 0u, 0u, 0u,
    0u, 0u, 0u, 0u, 0u, 0u
);

let a0 = sbb(tmp.value[0], 0xffffd556u, 0u);
let a1 = sbb(tmp.value[1], 0xdcff7fffu,a0[1]);
let a2 = sbb(tmp.value[2], 0x58a9ffffu,a1[1]);
let a3 = sbb(tmp.value[3], 0xf55ffffu,a2[1]);
let a4 = sbb(tmp.value[4], 0x7b587b12u,a3[1]);
let a5 = sbb(tmp.value[5], 0xb3986950u,a4[1]);
let a6 = sbb(tmp.value[6], 0x79c2895fu,a5[1]);
let a7 = sbb(tmp.value[7], 0xb23ba5c2u,a6[1]);
let a8 = sbb(tmp.value[8], 0x21a5d66bu,a7[1]);
let a9 = sbb(tmp.value[9], 0x258dd3dbu,a8[1]);
let a10 = sbb(tmp.value[10], 0x1cbff34du,a9[1]);
let a11 = sbb(tmp.value[11], 0xd0088f5u,a10[1]);

// only a11[1] is important, as if the element
// was smaller it will produce a borrow value 
// 0xffff_ffff, otherwise it will be zero
// we can return u32 

if (a11[1] & 1u) == 1u {
    return 0u;
  }else {
    return 1u;
  }

}

// local invocation is like 0 to x for every workgroup
@compute
@workgroup_size(32,1,1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {

    v_indices[global_id.x] = v_indices[global_id.x] * 4294967295u;
}



/// Tests
@compute
@workgroup_size(1,1,1)
fn sum_test() {
    let a = sum(v_indices[0], v_indices[1]);
    v_indices[0] = a[0];
    v_indices[1] = a[1];
}

@compute
@workgroup_size(1,1,1)
fn adc_test() {
    let a = adc(v_indices[0], v_indices[1], v_indices[2]);
    v_indices[0] = a[0];
    v_indices[1] = a[1];
}

@compute
@workgroup_size(1,1,1)
fn multiply_test() {
    let a = multiply(v_indices[0], v_indices[1]);
    v_indices[0] = a[0];
    v_indices[1] = a[1];
}

@compute
@workgroup_size(1,1,1)
fn mac_test() {
    // let a = mac(v_indices[0], v_indices[1], v_indices[2], v_indices[3]);
    let a = multiply(v_indices[1], v_indices[2]);
    v_indices[0] = a[0];
    v_indices[1] = a[1];
}

@compute
@workgroup_size(1,1,1)
fn sbb_test() {
    let a = sbb(v_indices[0], v_indices[1], v_indices[2]);
    v_indices[0] = a[0];
    v_indices[1] = a[1];
}

@compute
@workgroup_size(1,1,1)
fn add_test() {

    let fp1 = Fp(array<u32,12>(v_indices[0], v_indices[1], v_indices[2], v_indices[3], v_indices[4], v_indices[5], v_indices[6], v_indices[7], v_indices[8], v_indices[9], v_indices[10], v_indices[11]));
    let fp2 = Fp(array<u32,12>(v_indices[12], v_indices[13], v_indices[14], v_indices[15], v_indices[16], v_indices[17], v_indices[18], v_indices[19], v_indices[20], v_indices[21], v_indices[22], v_indices[23]));


    let added_value = Fp_add(fp1, fp2);
    v_indices[0] = added_value.value[0];
    v_indices[1] = added_value.value[1];
    v_indices[2] = added_value.value[2];
    v_indices[3] = added_value.value[3];
    v_indices[4] = added_value.value[4];
    v_indices[5] = added_value.value[5];
    v_indices[6] = added_value.value[6];
    v_indices[7] = added_value.value[7];
    v_indices[8] = added_value.value[8];
    v_indices[9] = added_value.value[9];
    v_indices[10] = added_value.value[10];
    v_indices[11] = added_value.value[11];
}

@compute
@workgroup_size(1,1,1)
fn fp_subtract_test() {
    let fp1 = Fp(array<u32,12>(v_indices[0], v_indices[1], v_indices[2], v_indices[3], v_indices[4], v_indices[5], v_indices[6], v_indices[7], v_indices[8], v_indices[9], v_indices[10], v_indices[11]));
    let fp2 = Fp(array<u32,12>(v_indices[12], v_indices[13], v_indices[14], v_indices[15], v_indices[16], v_indices[17], v_indices[18], v_indices[19], v_indices[20], v_indices[21], v_indices[22], v_indices[23]));


    let added_value = Fp_sub(fp1, fp2);
    v_indices[0] = added_value.value[0];
    v_indices[1] = added_value.value[1];
    v_indices[2] = added_value.value[2];
    v_indices[3] = added_value.value[3];
    v_indices[4] = added_value.value[4];
    v_indices[5] = added_value.value[5];
    v_indices[6] = added_value.value[6];
    v_indices[7] = added_value.value[7];
    v_indices[8] = added_value.value[8];
    v_indices[9] = added_value.value[9];
    v_indices[10] = added_value.value[10];
    v_indices[11] = added_value.value[11];
}


@compute
@workgroup_size(1,1,1)
fn fp_neg_test() {
    let fp1 = Fp(array<u32,12>(v_indices[0], v_indices[1], v_indices[2], v_indices[3], v_indices[4], v_indices[5], v_indices[6], v_indices[7], v_indices[8], v_indices[9], v_indices[10], v_indices[11]));


    let added_value = Fp_neg(fp1);
    v_indices[0] = added_value.value[0];
    v_indices[1] = added_value.value[1];
    v_indices[2] = added_value.value[2];
    v_indices[3] = added_value.value[3];
    v_indices[4] = added_value.value[4];
    v_indices[5] = added_value.value[5];
    v_indices[6] = added_value.value[6];
    v_indices[7] = added_value.value[7];
    v_indices[8] = added_value.value[8];
    v_indices[9] = added_value.value[9];
    v_indices[10] = added_value.value[10];
    v_indices[11] = added_value.value[11];
}

@compute
@workgroup_size(1,1,1)
fn subtract_p_test() {

    let fp1 = Fp(array<u32,12>(v_indices[0], v_indices[1], v_indices[2], v_indices[3], v_indices[4], v_indices[5], v_indices[6], v_indices[7], v_indices[8], v_indices[9], v_indices[10], v_indices[11]));

    let subtract = subtract_p(fp1);

    v_indices[0] = subtract.value[0];
    v_indices[1] = subtract.value[1];
    v_indices[2] = subtract.value[2];
    v_indices[3] = subtract.value[3];
    v_indices[4] = subtract.value[4];
    v_indices[5] = subtract.value[5];
    v_indices[6] = subtract.value[6];
    v_indices[7] = subtract.value[7];
    v_indices[8] = subtract.value[8];
    v_indices[9] = subtract.value[9];
    v_indices[10] = subtract.value[10];
    v_indices[11] = subtract.value[11];
}


@compute
@workgroup_size(1,1,1)
fn fp_multiply_test() {
    let fp1 = Fp(array<u32,12>(v_indices[0], v_indices[1], v_indices[2], v_indices[3], v_indices[4], v_indices[5], v_indices[6], v_indices[7], v_indices[8], v_indices[9], v_indices[10], v_indices[11]));
    let fp2 = Fp(array<u32,12>(v_indices[12], v_indices[13], v_indices[14], v_indices[15], v_indices[16], v_indices[17], v_indices[18], v_indices[19], v_indices[20], v_indices[21], v_indices[22], v_indices[23]));


    let added_value = Fp_mul(fp1, fp2);
    v_indices[0] = added_value.value[0];
    v_indices[1] = added_value.value[1];
    v_indices[2] = added_value.value[2];
    v_indices[3] = added_value.value[3];
    v_indices[4] = added_value.value[4];
    v_indices[5] = added_value.value[5];
    v_indices[6] = added_value.value[6];
    v_indices[7] = added_value.value[7];
    v_indices[8] = added_value.value[8];
    v_indices[9] = added_value.value[9];
    v_indices[10] = added_value.value[10];
    v_indices[11] = added_value.value[11];
}




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


fn Fp2_sum_of_products(a:array<Fp,2> , b: array<Fp,2>) -> Fp {
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


    t0 = mac(t[0], a[1].value[0], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[0], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[0], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[0], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[0], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[0], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[0], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[0], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[0], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[0], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[0], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[0], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[1], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[1], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[1], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[1], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[1], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[1], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[1], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[1], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[1], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[1], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[1], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[1], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[1], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[1], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[1], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[1], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[1], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[1], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[1], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[1], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[1], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[1], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[1], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[1], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[2], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[2], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[2], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[2], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[2], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[2], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[2], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[2], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[2], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[2], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[2], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[2], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[2], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[2], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[2], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[2], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[2], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[2], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[2], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[2], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[2], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[2], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[2], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[2], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[3], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[3], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[3], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[3], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[3], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[3], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[3], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[3], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[3], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[3], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[3], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[3], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[3], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[3], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[3], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[3], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[3], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[3], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[3], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[3], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[3], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[3], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[3], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[3], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[4], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[4], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[4], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[4], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[4], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[4], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[4], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[4], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[4], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[4], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[4], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[4], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[4], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[4], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[4], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[4], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[4], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[4], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[4], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[4], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[4], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[4], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[4], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[4], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[5], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[5], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[5], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[5], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[5], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[5], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[5], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[5], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[5], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[5], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[5], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[5], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[5], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[5], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[5], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[5], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[5], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[5], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[5], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[5], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[5], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[5], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[5], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[5], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[6], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[6], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[6], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[6], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[6], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[6], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[6], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[6], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[6], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[6], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[6], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[6], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[6], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[6], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[6], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[6], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[6], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[6], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[6], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[6], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[6], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[6], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[6], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[6], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[7], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[7], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[7], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[7], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[7], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[7], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[7], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[7], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[7], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[7], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[7], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[7], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[7], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[7], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[7], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[7], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[7], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[7], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[7], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[7], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[7], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[7], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[7], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[7], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[8], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[8], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[8], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[8], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[8], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[8], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[8], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[8], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[8], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[8], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[8], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[8], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[8], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[8], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[8], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[8], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[8], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[8], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[8], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[8], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[8], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[8], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[8], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[8], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[9], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[9], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[9], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[9], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[9], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[9], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[9], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[9], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[9], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[9], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[9], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[9], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[9], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[9], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[9], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[9], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[9], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[9], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[9], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[9], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[9], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[9], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[9], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[9], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[10], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[10], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[10], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[10], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[10], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[10], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[10], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[10], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[10], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[10], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[10], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[10], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[10], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[10], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[10], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[10], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[10], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[10], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[10], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[10], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[10], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[10], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[10], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[10], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[11], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[11], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[11], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[11], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[11], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[11], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[11], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[11], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[11], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[11], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[11], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[11], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[11], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[11], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[11], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[11], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[11], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[11], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[11], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[11], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[11], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[11], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[11], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[11], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
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


fn Fp2_pow_vartime(fp2: Fp2 , by: array<u32,12>) -> Fp2{
    var res = Fp2_one();


    res = Fp2_square(res);
    if (by[11] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[11] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[10] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[9] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[8] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[7] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[6] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[5] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[4] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[3] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[2] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[1] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 63u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 62u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 61u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 60u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 59u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 58u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 57u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 56u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 55u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 54u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 53u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 52u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 51u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 50u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 49u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 48u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 47u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 46u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 45u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 44u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 43u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 42u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 41u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 40u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 39u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 38u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 37u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 36u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 35u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 34u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 33u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 32u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 31u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 30u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 29u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 28u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 27u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 26u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 25u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 24u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 23u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 22u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 21u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 20u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 19u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 18u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 17u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 16u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 15u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 14u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 13u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 12u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 11u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 10u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 9u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 8u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 7u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 6u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 5u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 4u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 3u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 2u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 1u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }
              

    res = Fp2_square(res);
    if (by[0] >> 0u & 1u) == 1u {
        res = Fp2_mul(res, fp2);
    }

    return res;              
}

fn Fp2_invert(fp2: Fp2) -> Fp2 {
  let tmp = Fp_invert(Fp_add(square(fp2.c0),square(fp2.c1)));

  return Fp2(
    Fp_mul(fp2.c0, tmp),
    Fp_mul(fp2.c1,Fp_invert(tmp))
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

@compute
@workgroup_size(1,1,1)
fn Fp6_add_test() {


  let fp0 = Fp(array<u32,12>(v_indices[0], v_indices[1], v_indices[2], v_indices[3], v_indices[4], v_indices[5], v_indices[6], v_indices[7], v_indices[8], v_indices[9], v_indices[10], v_indices[11]));



  let fp1 = Fp(array<u32,12>(v_indices[12], v_indices[13], v_indices[14], v_indices[15], v_indices[16], v_indices[17], v_indices[18], v_indices[19], v_indices[20], v_indices[21], v_indices[22], v_indices[23]));



  let fp2 = Fp(array<u32,12>(v_indices[24], v_indices[25], v_indices[26], v_indices[27], v_indices[28], v_indices[29], v_indices[30], v_indices[31], v_indices[32], v_indices[33], v_indices[34], v_indices[35]));



  let fp3 = Fp(array<u32,12>(v_indices[36], v_indices[37], v_indices[38], v_indices[39], v_indices[40], v_indices[41], v_indices[42], v_indices[43], v_indices[44], v_indices[45], v_indices[46], v_indices[47]));



  let fp4 = Fp(array<u32,12>(v_indices[48], v_indices[49], v_indices[50], v_indices[51], v_indices[52], v_indices[53], v_indices[54], v_indices[55], v_indices[56], v_indices[57], v_indices[58], v_indices[59]));



  let fp5 = Fp(array<u32,12>(v_indices[60], v_indices[61], v_indices[62], v_indices[63], v_indices[64], v_indices[65], v_indices[66], v_indices[67], v_indices[68], v_indices[69], v_indices[70], v_indices[71]));



  let fp6 = Fp(array<u32,12>(v_indices[72], v_indices[73], v_indices[74], v_indices[75], v_indices[76], v_indices[77], v_indices[78], v_indices[79], v_indices[80], v_indices[81], v_indices[82], v_indices[83]));



  let fp7 = Fp(array<u32,12>(v_indices[84], v_indices[85], v_indices[86], v_indices[87], v_indices[88], v_indices[89], v_indices[90], v_indices[91], v_indices[92], v_indices[93], v_indices[94], v_indices[95]));



  let fp8 = Fp(array<u32,12>(v_indices[96], v_indices[97], v_indices[98], v_indices[99], v_indices[100], v_indices[101], v_indices[102], v_indices[103], v_indices[104], v_indices[105], v_indices[106], v_indices[107]));



  let fp9 = Fp(array<u32,12>(v_indices[108], v_indices[109], v_indices[110], v_indices[111], v_indices[112], v_indices[113], v_indices[114], v_indices[115], v_indices[116], v_indices[117], v_indices[118], v_indices[119]));



  let fp10 = Fp(array<u32,12>(v_indices[120], v_indices[121], v_indices[122], v_indices[123], v_indices[124], v_indices[125], v_indices[126], v_indices[127], v_indices[128], v_indices[129], v_indices[130], v_indices[131]));



  let fp11 = Fp(array<u32,12>(v_indices[132], v_indices[133], v_indices[134], v_indices[135], v_indices[136], v_indices[137], v_indices[138], v_indices[139], v_indices[140], v_indices[141], v_indices[142], v_indices[143]));



  let a = Fp6(Fp2(fp1,fp2), Fp2(fp3,fp4), Fp2(fp5,fp6));
  let b = Fp6(Fp2(fp7,fp8), Fp2(fp8,fp9), Fp2(fp10,fp11));


   let added_value = Fp6_add(a,b);
    v_indices[0] = added_value.c0.c0.value[0];
    v_indices[1] = added_value.c0.c0.value[1];
    v_indices[2] = added_value.c0.c0.value[2];
    v_indices[3] = added_value.c0.c0.value[3];
    v_indices[4] = added_value.c0.c0.value[4];
    v_indices[5] = added_value.c0.c0.value[5];
    v_indices[6] = added_value.c0.c0.value[6];
    v_indices[7] = added_value.c0.c0.value[7];
    v_indices[8] = added_value.c0.c0.value[8];
    v_indices[9] = added_value.c0.c0.value[9];
    v_indices[10] = added_value.c0.c0.value[10];
    v_indices[11] = added_value.c0.c0.value[11];

    v_indices[12] = added_value.c0.c1.value[0];
    v_indices[13] = added_value.c0.c1.value[1];
    v_indices[14] = added_value.c0.c1.value[2];
    v_indices[15] = added_value.c0.c1.value[3];
    v_indices[16] = added_value.c0.c1.value[4];
    v_indices[17] = added_value.c0.c1.value[5];
    v_indices[18] = added_value.c0.c1.value[6];
    v_indices[19] = added_value.c0.c1.value[7];
    v_indices[20] = added_value.c0.c1.value[8];
    v_indices[21] = added_value.c0.c1.value[9];
    v_indices[22] = added_value.c0.c1.value[10];
    v_indices[23] = added_value.c0.c1.value[11];


}



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
              Fp2(Fp_zero(),
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
      ))));

  c2 = Fp2_mul(c2,
              Fp2(Fp(array<u32,12>(
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
      )),Fp_zero()));

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
  let b10_p_b11 = Fp_add(b.c1.c0, b.c1.c1);
  let b10_m_b11 = Fp_sub(b.c1.c0, b.c1.c1);
  let b20_p_b21 = Fp_add(b.c2.c0, b.c2.c1);
  let b20_m_b21 = Fp_sub(b.c2.c0, b.c2.c1);

  return Fp6(
    Fp2(
      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, Fp_neg(a.c0.c1), a.c1.c0, Fp_neg(a.c1.c1), a.c2.c0, Fp_neg(a.c2.c1)),
        array<Fp,6>(b.c0.c0, b.c0.c1, b20_m_b21, b20_p_b21, b10_m_b11, b10_p_b11),
      ),
      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, a.c0.c1, a.c1.c0, a.c1.c1, a.c2.c0, a.c2.c1),
        array<Fp,6>(b.c0.c0, b.c0.c1, b20_p_b21, b20_m_b21, b10_p_b11, b10_m_b11),
      ),
    ),

    Fp2(

      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, Fp_neg(a.c0.c1), a.c1.c0, Fp_neg(a.c1.c1), a.c2.c0, Fp_neg(a.c2.c1)),
        array<Fp,6>(b.c1.c0, b.c1.c1, b.c0.c0, b.c0.c1, b20_m_b21, b20_p_b21),
      ),

      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, a.c0.c1, a.c1.c0, a.c1.c1, a.c2.c0, a.c2.c1),
        array<Fp,6>(b.c1.c0, b.c1.c1, b.c0.c1, b.c0.c0, b20_p_b21, b20_m_b21),
      ),
    ),

    Fp2(

      Fp6_sum_of_products(
        array<Fp,6>(a.c0.c0, Fp_neg(a.c0.c1), a.c1.c0, Fp_neg(a.c1.c1), a.c2.c0, Fp_neg(a.c2.c1)),
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
    
    var c1 = Fp2_mul_by_nonresidue(Fp2_square(fp6.c2));
    c1 = Fp2_sub(c1, Fp2_mul(fp6.c0, fp6.c1));
    
    var c2 = Fp2_square(fp6.c1);
    c2 = Fp2_sub(c2, Fp2_mul(fp6.c0, fp6.c2));

    var tmp = Fp2_mul_by_nonresidue(Fp2_add(Fp2_mul(fp6.c1,c2), Fp2_mul(fp6.c2,c1)));
    tmp = Fp2_invert(Fp2_add(tmp, Fp2_mul(fp6.c0,c0)));

    return Fp6 (
      Fp2_mul(tmp,c0),
      Fp2_mul(tmp,c1),
      Fp2_mul(tmp,c2),
    );
}

fn Fp6_square(fp6: Fp6) -> Fp6 {
  let s0 = Fp2_square(fp6.c0);
  let ab = Fp2_mul(fp6.c0, fp6.c1);
  let s1 = Fp2_add(ab, ab);
  let s2 = Fp2_square(Fp2_add(fp6.c2, Fp2_sub(fp6.c0,fp6.c1)));
  let bc = Fp2_mul(fp6.c1, fp6.c2);
  let s3 = Fp2_add(bc, bc);
  let s4 = Fp2_square(fp6.c2);

  return Fp6 (
    Fp2_add(Fp2_mul_by_nonresidue(s3),s0),
    Fp2_add(Fp2_mul_by_nonresidue(s4),s1),
    Fp2_sub(Fp2_sub(Fp2_add(Fp2_add(s2,s3),s1),s0),s4)
  );
}

fn Fp6_neg(fp6: Fp6) -> Fp6 {
    return Fp6(
        Fp2_neg(fp6.c0), 
        Fp2_neg(fp6.c1), 
        Fp2_neg(fp6.c2), 
    );
}

fn Fp6_mul_by_nonresidue(fp6: Fp6) -> Fp6 {
  return Fp6(Fp2_mul_by_nonresidue(fp6.c2), fp6.c0, fp6.c1);
} 

fn Fp2_neg(fp2: Fp2) -> Fp2 {
    return Fp2(
        Fp_neg(fp2.c0), 
        Fp_neg(fp2.c1), 
    );
}


// fp12

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

//fn mul_by_014(c0: Fp2, c1: Fp2, c4: Fp2) -> Fp12 {
//}

fn Fp12_frobenius_map(fp12: Fp12) -> Fp12 {
  var c0 = Fp6_frobenius_map(fp12.c0);
  var c1 = Fp6_frobenius_map(fp12.c1);
  
  c1 = Fp6_mul(c1, Fp6(Fp2(Fp(array<u32,12>(
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

fn u32_conditional_select(a: u32, b: u32, choice: u32) -> u32 {
    if choice == 0u {
      return a;
    }else{
      return b;
  }
}

// conditionally select
fn Fp_conditional_select(a: Fp, b: Fp, choice: u32) -> Fp{
    return Fp(
        array<u32,12>(
          u32_conditional_select(a.value[0],b.value[0],choice),
          u32_conditional_select(a.value[1],b.value[1],choice),
          u32_conditional_select(a.value[2],b.value[2],choice),
          u32_conditional_select(a.value[3],b.value[3],choice),
          u32_conditional_select(a.value[4],b.value[4],choice),
          u32_conditional_select(a.value[5],b.value[5],choice),
          u32_conditional_select(a.value[6],b.value[6],choice),
          u32_conditional_select(a.value[7],b.value[7],choice),
          u32_conditional_select(a.value[8],b.value[8],choice),
          u32_conditional_select(a.value[9],b.value[9],choice),
          u32_conditional_select(a.value[10],b.value[10],choice),
          u32_conditional_select(a.value[11],b.value[11],choice),
        )
      );
}

fn Fp_is_zero(fp: Fp) -> u32 {
  return 
        u32(( fp.value[0] == 0u ) & 
        ( fp.value[1] == 0u ) & 
        ( fp.value[2] == 0u ) & 
        ( fp.value[3] == 0u ) & 
        ( fp.value[4] == 0u ) & 
        ( fp.value[5] == 0u ) & 
        ( fp.value[6] == 0u ) & 
        ( fp.value[7] == 0u ) & 
        ( fp.value[8] == 0u ) & 
        ( fp.value[9] == 0u ) & 
        ( fp.value[10] == 0u ) & 
        ( fp.value[11] == 0u ));
}

// G1
const G1_B: Fp = Fp(
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
);

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

/// Returns true if this point is on the curve. This should always return
/// true unless an "unchecked" API was used.
fn G1Projective_is_on_cruve(g1: G1Projective) -> u32 {
  // Y^2 Z = X^3 + b Z^3

  return Fp_ct_eq((Fp_mul(square(g1.y), g1.z)),Fp_add(Fp_mul(square(g1.x), g1.x),Fp_mul(square(g1.z),Fp_mul(g1.z,G1_B)))) | Fp_is_zero(g1.z);
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

fn G1Affine_generator() -> G1Affine{
  return G1Affine(
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
    0u
  );
}

fn G1Affine_identity() -> G1Affine {
  return G1Affine(
    Fp_zero(),
    Fp_one(),
    1u
  );
}

// G1 Tests
@compute
@workgroup_size(1,1,1)
fn G1Projective_test_generator_is_on_curve() {
    let a = G1Projective_is_on_cruve(G1Projective_generator());
  
    v_indices[0] = a;
}

@compute
@workgroup_size(1,1,1)
fn G1Projective_test_identity_is_on_curve() {
    let a = G1Projective_is_on_cruve(G1Projective_identity());
  
    v_indices[0] = a;
}

@compute
@workgroup_size(1,1,1)
fn G1Projective_test_equality() {
    let a = G1Projective_ct_eq(G1Projective_identity(),G1Projective_generator());
  
    v_indices[0] = a;
}

@compute
@workgroup_size(1,1,1)
fn G1Projective_test_conditionally_select_affine_should_select_first() {
  let a = G1Projective_generator();
  let b = G1Projective_identity();

  v_indices[0] = G1Projective_ct_eq(G1Projective_conditional_select(a,b,0u),a);

}


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


