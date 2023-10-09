@compute
@workgroup_size(8)
fn initializeCountersSizesAtomicsHistogramKernel_test(@builtin(global_invocation_id) global_id: vec3u) {
  var countersPtr: array<WideNumber, limbs>;
  var sizesPtr: array<u32, limbs>;
  var atomicsPtr: array<u32, limbs>;
  var histogramPtr: array<u32, limbs>;

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  initializeCountersSizesAtomicsHistogramKernel(&countersPtr, &sizesPtr, &atomicsPtr, &histogramPtr, thread, global_id);

  for(var i = 0; i<12; i++) {
    v_indices[i] = countersPtr[i].first;
    v_indices[i+12] = countersPtr[i].second;
    v_indices[i+24] = sizesPtr[i];
    v_indices[i+36] = atomicsPtr[i];
    v_indices[i+48] = histogramPtr[i];
  }
}

@compute
@workgroup_size(8)
fn sizesPrefixSumKernel_test(@builtin(global_invocation_id) global_id: vec3u) {
  var pagesPtr: array<u32, limbs>;
  var prefixSumSizesPtr: array<u32, limbs>;
  var sizesPtr: array<u32, limbs>;
  for(var i =0u;i<limbs;i++) {
    sizesPtr[i] = 1u;
  }
  var countersPtr: array<WideNumber, limbs>;
  var atomicsPtr: array<u32, limbs>;

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  sizesPrefixSumKernel(pagesPtr, &prefixSumSizesPtr, &sizesPtr, countersPtr, atomicsPtr, thread, global_id);

  for(var i = 0; i<12; i++) {
    v_indices[i] = prefixSumSizesPtr[i];
    v_indices[i+12] = sizesPtr[i];
  }
}