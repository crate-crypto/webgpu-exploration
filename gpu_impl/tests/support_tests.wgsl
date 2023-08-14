@compute
@workgroup_size(1)
fn warpPrefixSum_test() {
  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  v_indices[0] = warpPrefixSum(1000u, 100u, thread);
}

@compute
@workgroup_size(1)
fn multiwarpPrefixSum1_test() {
  var thread: Thread = Thread(
    vec2<u32>(100u, 200u),
    vec2<u32>(100u, 200u),
    vec2<u32>(100u, 200u),
    vec2<u32>(100u, 200u),
    1u,
  );

  var shared_: array<u32, 32> = array<u32, 32>(0u, 32u, 0u, 0u, 0u, 0u, 0u, 0u, 
                                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

  v_indices[0] = multiwarpPrefixSum1(&shared_, 1000u, 100u, thread);
}

@compute
@workgroup_size(1)
fn multiwarpPrefixSum2_test() {
  var thread: Thread = Thread(
    vec2<u32>(100u, 200u),
    vec2<u32>(100u, 200u),
    vec2<u32>(100u, 200u),
    vec2<u32>(100u, 200u),
    1u,
  );

  v_indices[0] = multiwarpPrefixSum2(1u, 1000u, 100u, thread);
}

@compute
@workgroup_size(1)
fn udiv3_test() { 
  v_indices[0] = udiv3(v_indices[0]);
}

@compute
@workgroup_size(1)
fn udiv5_test() { 
  v_indices[0] = udiv5(v_indices[0]);
}

@compute
@workgroup_size(1)
fn compress_test() { 
  var thread: Thread = Thread(
    vec2<u32>(100u, 200u),
    vec2<u32>(100u, 200u),
    vec2<u32>(100u, 200u),
    vec2<u32>(100u, 200u),
    1u,
  );

  v_indices[0] = compress(v_indices[0], thread);
}