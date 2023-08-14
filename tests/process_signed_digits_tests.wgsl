@compute
@workgroup_size(1)
fn slice23_test() {
  var sliced = array<u32, 12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var packed = array<u32, limbs>(100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);

  slice23(&sliced, packed);

  for(var i = 0;i<12; i++)
  {
    v_indices[i] = sliced[i];
  }
}

@compute
@workgroup_size(1)
fn sub_psd_test() {
  var sliced = array<u32, limbs>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var a = array<u32, limbs>(100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var b = array<u32, limbs>(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u);

  v_indices[12] = u32(sub_psd(&sliced, a, b));

  for(var i = 0;i<12; i++)
  {
    v_indices[i] = sliced[i];
  }
}

@compute
@workgroup_size(1)
fn addN_psd_test() {
  var sliced = array<u32, limbs>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var a = array<u32, limbs>(100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);

  addN_psd(&sliced, a);

  for(var i = 0;i<12; i++)
  {
    v_indices[i] = sliced[i];
  }
}

@compute
@workgroup_size(1)
fn negate_test() {
  var sliced = array<u32, limbs>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var a = array<u32, limbs>(100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);

  negate(&sliced, a);

  for(var i = 0;i<12; i++)
  {
    v_indices[i] = sliced[i];
  }
}

@compute
@workgroup_size(1)
fn ballot_sync_test() {
  v_indices[0] = ballot_sync(v_indices[0], false);
}

@compute
@workgroup_size(1)
fn processSignedDigitsKernel_test() {
  var processedScalarData = array<u32, limbs>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var scalarData = array<u32, limbs>(100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );

  processSignedDigitsKernel(processedScalarData, &scalarData, 100u, thread);

  for(var i = 0;i<300; i++)
  {
    v_indices[i] = memory.data[i];
  }
}