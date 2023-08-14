@compute
@workgroup_size(1)
fn histogramPrefixSumKernel_test() {
  var histogramPtr = array<u32, limbs>(1u, 200u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var unsortedTriplePtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  histogramPrefixSumKernel(&histogramPtr, unsortedTriplePtr, thread);

  for(var i = 0u; i < limbs; i++) {
    v_indices[i] = histogramPtr[i];
  }
}

@compute
@workgroup_size(1)
fn sortCountsKernel_test() {
  var sortedTriplePtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);
  var histogramPtr = array<u32, limbs>(1u, 200u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var unsortedTriplePtr = array<u32, limbs>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u);

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(100u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  sortCountsKernel(&sortedTriplePtr, histogramPtr, unsortedTriplePtr, thread);

  for(var i = 0u; i < limbs; i++) {
    v_indices[i] = sortedTriplePtr[i];
  }
}