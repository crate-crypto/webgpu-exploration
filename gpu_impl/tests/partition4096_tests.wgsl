@compute
@workgroup_size(1)
fn round128_test() {
  v_indices[0] = round128(v_indices[0]);
}

@compute
@workgroup_size(1)
fn initializeShared4096_test() {
  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  initializeShared4096(10u, thread);

  for(var i = 0; i<300; i++) {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn read640_test() {
  var data: array<u32, 5>;
  var pagesPtr = array<u32, limbs>(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u);
  var pageBase = array<u32, limbs>(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u);


  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  read640(&data, pagesPtr, &pageBase, 1u, thread);

  for(var i = 0; i<5; i++) {
    v_indices[i] = data[i];
  }
  for(var i = 0; i<12; i++) {
    v_indices[i+5] = pageBase[i];
  }
}

@compute
@workgroup_size(1)
fn shared_copy_u4_test() {
  var global: array<vec4<u32>, limbs>;


  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  shared_copy_u4(&global, 0u, 100u, thread);

  var j = 0;
  for(var i = 0; i<12; i+=4) {
    v_indices[i] = global[j].x;
    v_indices[i+1] = global[j].y;
    v_indices[i+2] = global[j].z;
    v_indices[i+3] = global[j].w;
    j++;
  }
}

@compute
@workgroup_size(1)
fn prefixSumBuckets_test() {
  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(512u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  prefixSumBuckets(0u, 100u, thread);

  for(var i = 0; i<300; i++) {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn sortMap_test() {
  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  sortMap(0u, 100u, thread);

  for(var i = 0; i<300; i++) {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn unpackData_test() {
  var lowBits: array<u32, 4>;
  var highBits: array<u32, 4>;
  var data = array<u32, 5>(10000u, 20000u, 300000u, 4u, 5u);

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  unpackData(&lowBits, &highBits, data, thread);

  for(var i = 0; i<4; i++) {
    v_indices[i] = lowBits[i];
    v_indices[i + 4] = highBits[i];
  }
}

@compute
@workgroup_size(1)
fn cleanupShared_test() {
  var scratchPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(50u, 2u),
    vec2<u32>(50u, 2u),
    vec2<u32>(100u, 2u),
    1u,
  );

  cleanupShared(12u, &scratchPtr, thread);

  for(var i = 0; i<300; i++) {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn writePointToShared_test() {
  var scratchPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  writePointToShared(10u, &scratchPtr, 1u, 2u, true, thread);

  for(var i = 0; i<300; i++) {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn partitionPagesToScratch_test() {
  var scratchPtr: array<u32, limbs>;

  var pagesPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var thread: Thread = Thread(
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    1u,
  );

  partitionPagesToScratch(1u, &scratchPtr, pagesPtr, 12u, thread);

  for(var i = 0; i<300; i++) {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn partitionScratchToPoints_test() {
  var scratchPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var pointsPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var thread: Thread = Thread(
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    1u,
  );

  partitionScratchToPoints(1u, &pointsPtr, scratchPtr, 10u, 12u, thread);

  for(var i = 0; i<12; i++) {
    v_indices[i] = pointsPtr[i];
  }
}

@compute
@workgroup_size(1)
fn countFromPages_test() {
  var pagesPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var thread: Thread = Thread(
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    1u,
  );

  countFromPages(1u, pagesPtr, 12000u, thread);

  for(var i = 0; i<300; i++) {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn partitionPagesToPoints_test() {
  var pointsPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var pagesPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var thread: Thread = Thread(
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    1u,
  );

  partitionPagesToPoints(1u, &pointsPtr, pagesPtr, 1000u, 12u, thread);

  for(var i = 0; i<12; i++) {
    v_indices[i] = pointsPtr[i];
  }
}

@compute
@workgroup_size(8)
fn partition4096Kernel_test(@builtin(global_invocation_id) global_id: vec3u) {
  var pointsPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var unsortedTriplePtr = array<vec4<u32>, limbs>(
    vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(0u, 1u, 2u, 3u),
    vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(0u, 1u, 2u, 3u),
    vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(0u, 1u, 2u, 3u),
    vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(0u, 1u, 2u, 3u),
  );

  var scratchPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var prefixSumSizesPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);
  var sizesPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var pagesPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var atomicsPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var thread: Thread = Thread(
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(100u, 2u),
    1u,
  );

  partition4096Kernel(&pointsPtr, unsortedTriplePtr, &scratchPtr, prefixSumSizesPtr,
                      sizesPtr, pagesPtr, atomicsPtr, 12u, thread, global_id);

  for(var i = 0; i<300; i++) {
    v_indices[i] = memory.data[i];
  }
}