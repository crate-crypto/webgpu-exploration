const zxc: u32 = 0u;

fn computeNP0(x: u32) -> u32 { // +
  var inv = x;

  inv=inv*(inv*x+14u);
  inv=inv*(inv*x+2u);
  inv=inv*(inv*x+2u);
  inv=inv*(inv*x+2u);

  return inv;
}

fn mp_zero(x: ptr<function, array<u32, limbs>>) { // +
  for(var i=0; i < i32(limbs); i++)
  {
    (*x)[i]=0u;
  }
}

fn mp_copy(dst: ptr<function, array<u32, limbs>>, src: array<u32, limbs>) { // +
  // it is not possible to index over the array that is passed to the function, so I make a copy of it
  var src_copy: array<u32, limbs> = src;
  for(var i=0; i < i32(limbs); i++)
  {
    (*dst)[i]=src_copy[i];
  }
}

fn mp_logical_or(a: array<u32, limbs>) -> u32 { // +
  var lor = a[0];
  var a_copy: array<u32, limbs> = a;
  for(var i=1;i < i32(limbs); i++)
  {
    lor = lor | a_copy[i];
  }
  return lor;
} 

fn funnelshift_rc(lo: u32, hi: u32, shift: u32) -> u32 { // +
  let concat: u32 = (hi << 32u) | (lo);
  let shifted: u32 = concat >> min(shift, 32u);
  return shifted;
}

fn mp_shift_right(r: ptr<function, array<u32, limbs>>, x: array<u32, limbs>, bits: u32) { // +
  var x_copy: array<u32, limbs> = x;

  for(var i=0;i < i32(limbs) - 1; i++)
  {
    (*r)[i] = funnelshift_rc(x_copy[i], x_copy[i+1], bits);
  }
  (*r)[i32(limbs) - 1] = funnelshift_rc(x_copy[i32(limbs) - 1], 0u, bits);
}

fn funnelshift_lc(lo: u32, hi: u32, shift: u32) -> u32 {
  let concat: u32 = (hi << 32u) | lo;
  let shifted: u32 = concat << min(shift, 32u);
  return shifted >> 32u;
}

fn mp_shift_left(r: ptr<function, array<u32, limbs>>, x: array<u32, limbs>, bits: u32) { // +
  var x_copy: array<u32, limbs> = x;
  for(var i=i32(limbs) - 1;i > 0; i--)
  {
    (*r)[i] = funnelshift_lc(x_copy[i - 1], x_copy[i], bits);
  }
  (*r)[0]=funnelshift_lc(0u, x_copy[0], bits);
}

fn mp_add(limbs_size: u32, r: ptr<function, array<u32, limbs>>, a: array<u32, limbs>, b: array<u32, limbs>) { // +
  var chain: chain_t;
  var a_copy: array<u32, limbs> = a;
  var b_copy: array<u32, limbs> = b;
  for(var i=0;i < i32(limbs_size); i++)
  {
    (*r)[i] = add_chain_t(&chain, a_copy[i], b_copy[i]);
  }
}

fn mp_add_carry(r: ptr<function, array<u32, limbs>>, a: array<u32, limbs>, b: array<u32, limbs>) -> bool { // +
  var chain: chain_t;
  var a_copy: array<u32, limbs> = a;
  var b_copy: array<u32, limbs> = b;
  for(var i=0;i < i32(limbs); i++)
  {
    (*r)[i] = add_chain_t(&chain, a_copy[i], b_copy[i]);
  }
  return getCarry(&chain);
}

fn mp_sub(limbs_size: u32, r: ptr<function, array<u32, limbs>>, a: array<u32, limbs>, b: array<u32, limbs>) { // +
  var chain: chain_t;
  var a_copy: array<u32, limbs> = a;
  var b_copy: array<u32, limbs> = b;
  for(var i=0;i < i32(limbs_size); i++)
  {
    (*r)[i] = sub_chain_t(&chain, a_copy[i], b_copy[i]);
  }
}

fn mp_sub_carry(limbs_size: u32, r: ptr<function, array<u32, limbs>>, a: array<u32, limbs>, b: array<u32, limbs>) -> bool { // +
  var chain: chain_t;
  var a_copy: array<u32, limbs> = a;
  var b_copy: array<u32, limbs> = b;
  for(var i=0;i < i32(limbs_size); i++)
  {
    (*r)[i] = sub_chain_t(&chain, a_copy[i], b_copy[i]);
  }
  return getCarry(&chain);
}

fn mp_comp_eq(a: array<u32, limbs>, b: array<u32, limbs>) -> bool { // +
  // name `match` is a reserved keyword
  var match_ = a[0] ^ b[0];
  var a_copy: array<u32, limbs> = a;
  var b_copy: array<u32, limbs> = b;

  for(var i=1;i < i32(limbs); i++)
  {
    match_=match_ | (a_copy[i] ^ b_copy[i]);
  }
  return match_==0u;
}

fn mp_comp_ge(a: array<u32, limbs>, b: array<u32, limbs>) -> bool { // +
  var chain: chain_t;
  var a_copy: array<u32, limbs> = a;
  var b_copy: array<u32, limbs> = b;

  for(var i=0;i < i32(limbs); i++)
  {
    sub_chain_t(&chain, a_copy[i], b_copy[i]);
  }
  return getCarry(&chain);
}

fn mp_comp_gt(a: array<u32, limbs>, b: array<u32, limbs>) -> bool { // +
  var chain: chain_t;
  var a_copy: array<u32, limbs> = a;
  var b_copy: array<u32, limbs> = b;

  for(var i=0;i < i32(limbs); i++)
  {
    sub_chain_t(&chain, a_copy[i], b_copy[i]);
  }
  return !getCarry(&chain);
}

fn mp_select(r: ptr<function, array<u32, limbs>>, abSelect: bool, a: array<u32, limbs>, b: array<u32, limbs>) { // +
  var a_copy: array<u32, limbs> = a;
  var b_copy: array<u32, limbs> = b;
  for(var i=0;i < i32(limbs); i++)
  {
    if abSelect {
      (*r)[i] = a_copy[i];
    }
    else {
      (*r)[i] = b_copy[i];
    }
  }
}

fn mp_mul_red_cl(evenOdd: ptr<function, array<WideNumber, limbs>>, a: array<u32, limbs>, b: array<u32, limbs>, n: array<u32, limbs>) -> bool {
  var even: array<WideNumber, limbs>;
  var odd: array<WideNumber, limbs>;

  var chain: chain_t;
  var carry: bool = false;
  var lo: u32 = 0u;
  var q: u32;
  var c1: u32;
  var c2: u32;

  for (var i : u32 = 0u; i < limbs/2u; i = i + 1u) {
    even[i] = make_wide1(0u, 0u);
    odd[i] = make_wide1(0u, 0u);
  }

  var a_copy = a;
  var b_copy = b;
  var n_copy = n;

  for(var i = 0; i < i32(limbs); i+=2) {
    if i!=0 {
      reset2(&chain, carry);
      lo = add_chain_t(&chain, lo, ulow2(even[0]));
      carry = (add_chain_t(&chain, 0u, 0u) != 0u);
      even[0] = make_wide1(lo, uhigh2(even[0]));
    }

    reset1(&chain);
    for(var j = 0; j < i32(limbs); j+=2){
      even[j/2] = madwide2(&chain, a_copy[i], b_copy[j], even[j/2]);
    }
    c1 = add_chain_t(&chain, 0u, 0u);

    reset1(&chain);
    for(var j = 0; j < i32(limbs); j+=2){
      odd[j/2] = madwide2(&chain, a_copy[i], b_copy[j+1], odd[j/2]);
    }

    q = qTerm(ulow2(even[0]));

    reset1(&chain);
    for(var j = 0; j < i32(limbs); j+=2){
      odd[j/2] = madwide2(&chain, q, n_copy[j+1], odd[j/2]);
    }

    reset1(&chain);
    even[0]=madwide2(&chain, q, n[0], even[0]);
    lo=uhigh2(even[0]);
    for(var j = 2; j < i32(limbs); j+=2){
      even[j/2 - 1] = madwide2(&chain, q, n_copy[j], even[j/2]);
    }
    c1=add_chain_t(&chain, c1, 0u);

    reset1(&chain);
    lo=add_chain_t(&chain, lo, ulow2(odd[0]));
    carry=(add_chain_t(&chain, 0u, 0u)!=0u);
    odd[0]=make_wide1(lo, uhigh2(odd[0]));

    reset1(&chain);
    for(var j = 0; j < i32(limbs); j+=2){
      odd[j/2] = madwide2(&chain, a_copy[i+1], b_copy[j], odd[j/2]);
    }
    c2=add_chain_t(&chain, 0u, 0u);

    q = qTerm(ulow2(odd[0]));

    reset1(&chain);
    odd[0]=madwide2(&chain, q, n[0], odd[0]);
    lo=uhigh2(odd[0]);
    for(var j = 2; j < i32(limbs); j+=2){
      odd[j/2 - 1] = madwide2(&chain, q, n_copy[j], odd[j/2]);
    }
    c2 = add_chain_t(&chain, c2, 0u);

    odd[i32(limbs)/2 - 1]=make_wide1(0u, 0u);
    even[i32(limbs)/2 - 1]=make_wide1(c1, c2);

    reset1(&chain);
    for(var j = 0; j < i32(limbs); j+=2){
      even[j/2] = madwide2(&chain, a_copy[i+1], b_copy[j+1], even[j/2]);
    }

    reset1(&chain);
    for(var j = 0; j < i32(limbs); j+=2){
      even[j/2] = madwide2(&chain, q, n_copy[j+1], even[j/2]);
    }
  }

  reset2(&chain, carry);
  lo=add_chain_t(&chain, lo, ulow2(even[0]));
  carry=(add_chain_t(&chain, 0u, 0u)!=0u);
  even[0]=make_wide1(lo, uhigh2(even[0]));

  return carry;
}

fn mp_sqr_red_cl(evenOdd: ptr<function, array<WideNumber, limbs>>, temp: ptr<function, array<u32, limbs>>, a: array<u32, limbs>, n: array<u32, limbs>) -> bool {
  var even: array<WideNumber, limbs>;
  var odd: array<WideNumber, limbs>;

  var chain: chain_t;
  var carry: bool = false;
  var lo: u32 = 0u;
  var q: u32;
  var c1: u32;
  var c2: u32;
  var low: u32;
  var high: u32;

  // This routine can be used when a+n < R (i.e. it doesn't carry out).  Hence the name cl for carryless.
  // Only works with an even number of limbs.

  mp_zero(temp);

  for(var i = 0; i < i32(limbs)/2; i++) {
    even[i]=make_wide1(0u, 0u);
    odd[i]=make_wide1(0u, 0u);
  }

  var a_copy = a;
  var n_copy = n;

  // do odds
  for(var j = i32(limbs) - 1; j > 0; j-=2) {
    reset1(&chain);
    for(var i = 0; i < i32(limbs) - j; i++) {
      (*evenOdd)[j/2 + i + 1]=madwide2(&chain, a_copy[i], a_copy[i + j], (*evenOdd)[j/2 + i + 1]);
    }
  }

  // shift right
  for(var i = 0; i < i32(limbs) - 1; i++) {
    (*evenOdd)[i]=make_wide1(uhigh2((*evenOdd)[i]), ulow2((*evenOdd)[i+1]));
  }
  (*evenOdd)[i32(limbs) - 1]=make_wide1(uhigh2((*evenOdd)[i32(limbs) - 1]), 0u);

  // do evens
  for(var j = i32(limbs) - 2; j > 0; j-=2) {
    reset1(&chain);
    for(var i = 0; i < i32(limbs) - j; i++) {
      (*evenOdd)[j/2+i]=madwide2(&chain, a_copy[i], a_copy[i+j], (*evenOdd)[j/2+i]);
    }
    if add_chain_t(&chain, 0u, 0u)!=0u {
      (*temp)[i32(limbs)-j] = 2u;
    }
    else {
      (*temp)[i32(limbs)-j] = 0u;
    }
  }

  // double
  reset1(&chain);
  for(var i = 0; i < i32(limbs); i++) { 
    low=add_chain_t(&chain, ulow2((*evenOdd)[i]), ulow2((*evenOdd)[i]));
    high=add_chain_t(&chain, uhigh2((*evenOdd)[i]), uhigh2((*evenOdd)[i]));
    (*evenOdd)[i]=make_wide1(low, high);
  }

  // add diagonals
  reset1(&chain);
  for(var i = 0; i < i32(limbs); i++) { 
    (*evenOdd)[i]=madwide2(&chain, a_copy[i], a_copy[i], (*evenOdd)[i]);
  }

  // add high part of wide to b...
  reset1(&chain);
  for(var i = 0; i < i32(limbs); i+=2) { 
    (*temp)[i]=add_chain_t(&chain, ulow2((*evenOdd)[i32(limbs)/2+i/2]), (*temp)[i]);
    (*temp)[i+1]=add_chain_t(&chain, uhigh2((*evenOdd)[i32(limbs)/2+i/2]), (*temp)[i+1]);
  }

  for(var i = 0; i < i32(limbs) / 2; i+=2) { 
    odd[i]=make_wide1(0u, 0u);
  }

  // now we need to reduce
  for(var i = 0; i < i32(limbs) / 2; i++) { 
    if i!=0 {
      // integrate lo
      reset1(&chain);
      lo=add_chain_t(&chain, lo, ulow2(even[0]));
      carry=(add_chain_t(&chain, 0u, 0u)!=0u);
      even[0]=make_wide1(lo, uhigh2(even[0]));
    }

    q=qTerm(ulow2(even[0]));

    // shift even by 64 bits
    reset1(&chain);
    even[0]=madwide2(&chain, q, n_copy[0], even[0]);
    lo=uhigh2(even[0]);
    for(var j = 2; j < i32(limbs); j+=2) {
      even[j/2 - 1]=madwide2(&chain, q, n_copy[j], even[j/2]);
    }
    c1=add_chain_t(&chain, 0u, 0u);

    reset1(&chain);
    for(var j = 0; j < i32(limbs); j+=2) {
      odd[j/2]=madwide2(&chain, q, n_copy[j+1], odd[j/2]);
    }

    // second half

    // integrate lo
    reset1(&chain);
    lo=add_chain_t(&chain, lo, ulow2(odd[0]));
    carry=(add_chain_t(&chain, 0u, 0u)!=0u);
    odd[0]=make_wide1(lo, uhigh2(odd[0]));

    q=qTerm(ulow2(odd[0]));

    // shift odd by 64 bits
    reset1(&chain);
    odd[0]=madwide2(&chain, q, n_copy[0], odd[0]);
    lo=uhigh2(odd[0]);
    for(var j = 2; j < i32(limbs); j+=2) {
      odd[j/2 - 1]=madwide2(&chain, q, n_copy[j], odd[j/2]);
    }
    odd[i32(limbs)/2 - 1].first = 0u;
    odd[i32(limbs)/2 - 1].second = 0u;
    c2=add_chain_t(&chain, 0u, 0u);

    reset1(&chain);
    for(var j = 0; j < i32(limbs) - 2; j+=2) {
      even[j/2]=madwide2(&chain, q, n_copy[j+1], even[j/2]);
    }
    even[i32(limbs)/2 - 1]=madwide2(&chain, q, n_copy[i32(limbs) - 1], make_wide1(c1, c2));
  }

  reset1(&chain);
  for(var i = 0; i < i32(limbs); i+=2) {
    low=add_chain_t(&chain, ulow2(even[i/2]), (*temp)[i]);
    high=add_chain_t(&chain, uhigh2(even[i/2]), (*temp)[i+1]);
    even[i/2]=make_wide1(low, high);
  }

  reset1(&chain);
  lo=add_chain_t(&chain, lo, ulow2(even[0]));
  carry=(add_chain_t(&chain, 0u, 0u)!=0u);
  even[0]=make_wide1(lo, uhigh2(even[0]));

  return carry;
}

fn mp_merge_cl(r: ptr<function, array<u32, limbs>>, evenOdd: array<WideNumber, limbs>, carry: bool) { // +
  var chain: chain_t;
  initialize_chain_t2(&chain, carry);

  (*r)[0]=ulow2(evenOdd[0]);

  var evenOdd_copy = evenOdd;
  for(var i = 0; i < i32(limbs) / 2 - 1; i++) {
    (*r)[2*i+1]=add_chain_t(&chain, uhigh2(evenOdd_copy[i]), ulow2(evenOdd_copy[i32(limbs)/2 + i]));
    (*r)[2*i+2]=add_chain_t(&chain, ulow2(evenOdd_copy[i+1]), uhigh2(evenOdd_copy[i32(limbs)/2 + i]));
  }
  (*r)[i32(limbs) - 1]=add_chain_t(&chain, uhigh2(evenOdd_copy[i32(limbs)/2 - 1]), 0u);
}