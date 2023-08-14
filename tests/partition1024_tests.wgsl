@compute
@workgroup_size(1)
fn nextPage_test() {
  var countersPtr: array<WideNumber, limbs> = array<WideNumber, limbs> (
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
  );
  
  v_indices[0] = nextPage(countersPtr);
}

@compute
@workgroup_size(1)
fn initializeShared_test() {
  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );
  
  initializeShared(thread);

  for(var i = 0; i<200; i++) {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn shared_copy_bytes_test() {
  var global = array<u32, limbs>(
    0u, 1u, 2u , 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u
  );

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );
  
  shared_copy_bytes(&global, 10u, 12u, thread);

  for(var i = 0; i<12; i++) {
    v_indices[i] = global[i];
  }
}

@compute
@workgroup_size(1)
fn clz_test() {
  v_indices[0] = clz(11u);
}

@compute
@workgroup_size(1)
fn cleanup1_test() {
  var pagesPtr = array<u32, limbs>(
    0u, 1u, 2u , 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u
  );

  var sizesPtr = array<u32, limbs>(
    0u, 1u, 2u , 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u
  );

  var countersPtr: array<WideNumber, limbs> = array<WideNumber, limbs> (
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
  );

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  var writeBytes = 15u;
  
  cleanup1(pagesPtr, &sizesPtr, &countersPtr, 0u, 0u, &writeBytes, thread);

  for(var i = 0; i<12; i++) {
    v_indices[i] = countersPtr[i].first;
    v_indices[i + 12] = countersPtr[i].second;
  }
}

@compute
@workgroup_size(1)
fn processWrites_test() {
  var pagesPtr = array<u32, limbs>(
    0u, 1u, 2u , 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u
  );

  var sizesPtr = array<u32, limbs>(
    0u, 1u, 2u , 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u
  );

  var countersPtr: array<WideNumber, limbs> = array<WideNumber, limbs> (
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
  );

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  var writeRequired = true;
  
  processWrites(pagesPtr, &sizesPtr, &countersPtr, &writeRequired, 10u, 0u, thread);

  for(var i = 0; i<12; i++) {
    v_indices[i] = countersPtr[i].first;
    v_indices[i + 12] = countersPtr[i].second;
  }
}

@compute
@workgroup_size(1)
fn partition1024Kernel_test() {
  var pagesPtr = array<u32, limbs>(
    0u, 1u, 2u , 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u
  );

  var sizesPtr = array<u32, limbs>(
    0u, 1u, 2u , 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u
  );

  var countersPtr: array<WideNumber, limbs> = array<WideNumber, limbs> (
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
    WideNumber(100u, 200u), WideNumber(100u, 200u), WideNumber(100u, 200u),
  );

  var processedScalarsPtr = array<u32, limbs>(
    0u, 1u, 2u , 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u
  );

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  var writeRequired = true;
  
  partition1024Kernel(pagesPtr, &sizesPtr, &countersPtr, processedScalarsPtr, 100u, thread);

  for(var i = 0; i<12; i++) {
    v_indices[i] = countersPtr[i].first;
    v_indices[i + 12] = countersPtr[i].second;
  }
}

@compute
@workgroup_size(1)
fn partition1024Kernel_call_test() {
  var pagesPtr = array<u32, limbs> (
    v_indices[4], v_indices[5], v_indices[6], v_indices[7], 
    v_indices[8], v_indices[9], v_indices[10], v_indices[11],
    v_indices[12], v_indices[13], v_indices[14], v_indices[15]
  );

  var sizesPtr = array<u32, limbs> (
    v_indices[16], v_indices[17], v_indices[18], v_indices[19], 
    v_indices[20], v_indices[21], v_indices[22], v_indices[23],
    v_indices[24], v_indices[25], v_indices[26], v_indices[27]
  );

  var wn1 = WideNumber(v_indices[28], v_indices[29]);
  var wn2 = WideNumber(v_indices[30], v_indices[31]);
  var wn3 = WideNumber(v_indices[32], v_indices[33]);
  var wn4 = WideNumber(v_indices[34], v_indices[35]);
  var wn5 = WideNumber(v_indices[36], v_indices[37]);
  var wn6 = WideNumber(v_indices[38], v_indices[39]);
  var wn7 = WideNumber(v_indices[40], v_indices[41]);
  var wn8 = WideNumber(v_indices[42], v_indices[43]);
  var wn9 = WideNumber(v_indices[44], v_indices[45]);
  var wn10 = WideNumber(v_indices[46], v_indices[47]);
  var wn11 = WideNumber(v_indices[48], v_indices[49]);
  var wn12 = WideNumber(v_indices[50], v_indices[51]);

  var countersPtr = array<WideNumber, limbs> (
    wn1, wn2, wn3, wn4, wn5, wn6, wn7, wn8, wn9, wn10, wn11, wn12
  );

  var processedScalarsPtr = array<u32, limbs> (
    v_indices[52], v_indices[53], v_indices[54], v_indices[55], 
    v_indices[56], v_indices[57], v_indices[58], v_indices[59],
    v_indices[60], v_indices[61], v_indices[62], v_indices[63]
  );

  var points = v_indices[64];

  var thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(v_indices[1], v_indices[1]),
    vec2<u32>(v_indices[0], v_indices[0]),
    v_indices[3],
  );

  partition1024Kernel(pagesPtr, &sizesPtr, &countersPtr, processedScalarsPtr, points, thread);

  for(var i = 0; i < 300; i++) {
    v_indices[i] = memory.data[i];
  }
}