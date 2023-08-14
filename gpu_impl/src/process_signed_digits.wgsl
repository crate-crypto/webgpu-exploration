fn slice23(sliced: ptr<function, array<u32, 12>>, packed: array<u32, limbs>) { // +
  (*sliced)[0]=packed[0] & 0x7FFFFFu;
  (*sliced)[1]=funnelshift_rc(packed[0], packed[1], 23u) & 0x7FFFFFu;
  (*sliced)[2]=funnelshift_rc(packed[1], packed[2], 14u) & 0x7FFFFFu;
  (*sliced)[3]=(packed[2]>>5u) & 0x7FFFFFu;
  (*sliced)[4]=funnelshift_rc(packed[2], packed[3], 28u) & 0x7FFFFFu;
  (*sliced)[5]=funnelshift_rc(packed[3], packed[4], 19u) & 0x7FFFFFu;
  (*sliced)[6]=funnelshift_rc(packed[4], packed[5], 10u) & 0x7FFFFFu;
  (*sliced)[7]=(packed[5]>>1u) & 0x7FFFFFu;
  (*sliced)[8]=funnelshift_rc(packed[5], packed[6], 24u) & 0x7FFFFFu;
  (*sliced)[9]=funnelshift_rc(packed[6], packed[7], 15u) & 0x7FFFFFu;
  (*sliced)[10]=(packed[7]>>6u) & 0x7FFFFFu;
  (*sliced)[11]=0u;
}

fn sub_psd(res: ptr<function, array<u32, limbs>>, a: array<u32, limbs>, b: array<u32, limbs>) -> bool { // +
  return mp_sub_carry(8u, res, a, b);
}

fn addN_psd(res: ptr<function, array<u32, limbs>>, x: array<u32, limbs>) { // +
  var localN: array<u32, limbs>;

  localN[0]=0x00000001u;
  localN[1]=0x0a118000u;
  localN[2]=0xd0000001u;
  localN[3]=0x59aa76feu;
  localN[4]=0x5c37b001u;
  localN[5]=0x60b44d1eu;
  localN[6]=0x9a2ca556u;
  localN[7]=0x12ab655eu; 

  mp_add(8u, res, localN, x);
}

fn negate(res: ptr<function, array<u32, limbs>>, x: array<u32, limbs>) { // +
  var localN: array<u32, limbs>;

  localN[0]=0x00000001u;
  localN[1]=0x0a118000u;
  localN[2]=0xd0000001u;
  localN[3]=0x59aa76feu;
  localN[4]=0x5c37b001u;
  localN[5]=0x60b44d1eu;
  localN[6]=0x9a2ca556u;
  localN[7]=0x12ab655eu; 

  mp_sub(8u, res, localN, x);
}

fn ballot_sync(mask: u32, predicate: bool) -> u32 { // +
  var predicate_mask: u32;

  if predicate {
    predicate_mask = 0xffffffffu;
  }
  else {
    predicate_mask = 0u;
  }
  return (mask & predicate_mask);
}

fn processSignedDigitsKernel(processedScalarData: array<u32, limbs>, scalarData: ptr<function, array<u32, limbs>>, points: u32, thread: Thread) { // +
  var warpThread = thread.threadIdx.x & 0x1Fu;
  var warp = thread.threadIdx.x>>5u;

  var current: u32 = 0u;
  var distributed: u32 = 0u;
  var mask: u32 = (1u<<thread.threadIdx.x)>>1u;
  var estimate: u32;

  var packed: array<u32, limbs>;
  var nTerm: array<u32, limbs>;
  var sliced: array<u32, 12>;

  var carry: bool;
  var neg: bool;

  var base: array<u32, limbs>;

  var transpose: array<u32, limbs>;

  if(thread.threadIdx.x==0u) 
  {
    distributed=0x00000001u;
  }
  if(thread.threadIdx.x==1u) 
  {
    distributed=0x0a118000u;
  }
  if(thread.threadIdx.x==2u) 
  {
    distributed=0xd0000001u;
  }
  if(thread.threadIdx.x==3u) 
  {
    distributed=0x59aa76feu;
  }
  if(thread.threadIdx.x==4u) 
  {
    distributed=0x5c37b001u;
  }
  if(thread.threadIdx.x==5u) 
  {
    distributed=0x60b44d1eu;
  }
  if(thread.threadIdx.x==6u) 
  {
    distributed=0x9a2ca556u;
  }
  if(thread.threadIdx.x==7u) 
  {
    distributed=0x12ab655eu;
  } 

  if(thread.threadIdx.x < 8u) {
    store_shared_u32(thread.threadIdx.x*4u + 264u, current); // 8448 -> 264
  } 

  for(var i = 1; i < 15; i++) {
    current=uadd_cc(current, distributed);
    carry=(uadd_cc(0u, 0u)!=0u);

    if (ballot_sync(0xFFFFFFFFu, carry) & mask)!=0u {
      current++;
    }
    if thread.threadIdx.x<8u {
      store_shared_u32(u32(i)*1u + thread.threadIdx.x*4u + 264u, current); // 8448 -> 264, 32 -> 1
    }
  }
  for(var i = 0; i<i32(thread.blockIdx.x); i++) {
    if (i + 8*32) > i32(limbs) {
      (*scalarData)[i] = (*scalarData)[(i + 8*32) % 13];
    }
    else {
      (*scalarData)[i] = (*scalarData)[i + 8*32]; // 256 -> 8
    }
  }
  
  for(var i=0;i<8;i++) {
    if (i*8 + i32(thread.threadIdx.x) > i32(limbs)) || (i*8 + i32(thread.threadIdx.x))*4 > i32(limbs) {
      transpose[(i*8 + i32(thread.threadIdx.x)) % 13] = (*scalarData)[((i*8 + i32(thread.threadIdx.x))*4) % 13]; // 256 -> 8
    }
    else {
      transpose[i*8 + i32(thread.threadIdx.x)] = (*scalarData)[(i*8 + i32(thread.threadIdx.x))*4]; // 256 -> 8
    }
  }

  var p0 = packed[0];
  var p1 = packed[1];
  var p2 = packed[2];
  var p3 = packed[3];
  var p4 = packed[4];
  var p5 = packed[5];
  var p6 = packed[6];
  var p7 = packed[7];


  load_shared_u4_2(&p0, &p1, &p2, &p3, thread.threadIdx.x*32u);
  load_shared_u4_2(&p4, &p5, &p6, &p7, thread.threadIdx.x*32u + 16u);

  packed[0] = p0;
  packed[1] = p1;
  packed[2] = p2;
  packed[3] = p3;
  packed[4] = p4;
  packed[5] = p5;
  packed[6] = p6;
  packed[7] = p7;

  estimate = packed[7] * 0x0Eu;

  var n0 = nTerm[0];
  var n1 = nTerm[1];
  var n2 = nTerm[2];
  var n3 = nTerm[3];
  var n4 = nTerm[4];
  var n5 = nTerm[5];
  var n6 = nTerm[6];
  var n7 = nTerm[7];

  load_shared_u4_2(&n0, &n1, &n2, &n3, estimate*32u + 16u);
  load_shared_u4_2(&n4, &n5, &n6, &n7, estimate*32u + 16u + 16u);

  nTerm[0] = n0;
  nTerm[1] = n1;
  nTerm[2] = n2;
  nTerm[3] = n3;
  nTerm[4] = n4;
  nTerm[5] = n5;
  nTerm[6] = n6;
  nTerm[7] = n7;

  if !sub_psd(&packed, packed, nTerm) {
    addN_psd(&packed, packed);
  }

  neg=(packed[7]>=0x10000000u);

  if neg {
    negate(&packed, packed);
  }

  slice23(&sliced, packed);

  for(var i = 0; i < 11; i++) {
    if sliced[i]<=0x00400000u {
      if neg && sliced[i]!=0u {
        sliced[i]+=0x00800000u;
      }
    }
    else if sliced[i]<0x00800000u {
      sliced[i]=(sliced[i] ^ 0x7FFFFFu) + 1u;
      if !neg {
        sliced[i]+=0x00800000u;
      }
      sliced[i+1]++;
    }
    else {
      sliced[i]=0u;
      sliced[i+1]++;
    }
    sliced[i]=compress(sliced[i], thread);  
  }

  var processedScalarData_copy = processedScalarData;
  if warpThread<24u {
    for(var i = 0; i < 11; i++) {
      for(var j = 0; j < 3*i32(points); j++) {
        if j > i32(limbs) {
          base[j % 13] = processedScalarData_copy[i];
        }
        else {
          base[j] = processedScalarData_copy[i];
        }
      }
      if warpThread*4u + warp*3u + thread.blockIdx.x*24u > limbs {
        base[(warpThread*4u + warp*3u + thread.blockIdx.x*24u) % 13u] = sliced[i];
      }
      else {
        base[warpThread*4u + warp*3u + thread.blockIdx.x*24u] = sliced[i]; // 768 -> 24, 96 -> 3
      }
    }
  }
}