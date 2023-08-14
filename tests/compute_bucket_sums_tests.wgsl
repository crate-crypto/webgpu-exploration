@compute
@workgroup_size(1)
fn copyCountsAndIndexes_test() {
  var arr = array<vec4<u32>, 3>(
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
  );

  v_indices[12] = copyCountsAndIndexes(1u, arr, 1u);

  for(var i = 0;i<12; i++)
  {
    v_indices[i] = memory.data[i+1];
  }
}

@compute
@workgroup_size(1)
fn copyPointIndexes_test() {
  var arr = array<vec4<u32>, limbs>(
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
    vec4<u32>(1u,2u,3u,4u),
  );

  var sequence = 1u;
  var pointIndexOffset = 1u;

  copyPointIndexes(&sequence, 0u, &pointIndexOffset, arr, 0u);

  //v_indices[12] = copyCountsAndIndexes(1u, arr, 1u);

  for(var i = 0;i<24; i++)
  {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn prefetch_test() {
  var pointsPtr: array<u32, limbs>;

  var storeOffset = 1u;
  var pointIndex = 1u;

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  prefetch(&storeOffset, pointIndex, pointsPtr, thread);

  //v_indices[12] = copyCountsAndIndexes(1u, arr, 1u);

  for(var i = 0;i<12; i++)
  {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn computeBucketSums_test() {
  var bucketsPtr: array<vec4<u32>, 12>;

  var pointsPtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  computeBucketSums(&bucketsPtr, pointsPtr, pointsPtr, pointsPtr, pointsPtr, thread);

  var j = 0;
  for(var i = 0;i<12; i++)
  {
    v_indices[j] = bucketsPtr[i].x;
    v_indices[j+1] = bucketsPtr[i].y;
    v_indices[j+2] = bucketsPtr[i].z;
    v_indices[j+3] = bucketsPtr[i].w;
    j+=4;
  }
}