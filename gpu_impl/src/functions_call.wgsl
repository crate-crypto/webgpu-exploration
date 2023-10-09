@compute
@workgroup_size(256)
fn processSignedDigitsKernel_call(@builtin(global_invocation_id) global_id: vec3u) {
  var thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(v_indices[1], v_indices[1]),
    vec2<u32>(v_indices[0], v_indices[0]),
    v_indices[3],
  );

  var processedScalarData = array<u32, limbs>(
    v_indices[4], v_indices[5], v_indices[6], 
    v_indices[7], v_indices[8], v_indices[9],
    v_indices[10], v_indices[11], v_indices[12],
    v_indices[13], v_indices[14], v_indices[15]
  );

  var scalarData = array<u32, limbs>(
    v_indices[16], v_indices[17], v_indices[18], 
    v_indices[19], v_indices[20], v_indices[21],
    v_indices[22], v_indices[23], v_indices[24],
    v_indices[25], v_indices[26], v_indices[27]
  );

  processSignedDigitsKernel(processedScalarData, &scalarData, v_indices[28], thread, global_id);

  for(var i = 0;i<300; i++)
  {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn initializeCountersSizesAtomicsHistogramKernel_call(@builtin(global_invocation_id) global_id: vec3u) {
  var wn1 = WideNumber(v_indices[4], v_indices[5]);
  var wn2 = WideNumber(v_indices[6], v_indices[7]);
  var wn3 = WideNumber(v_indices[8], v_indices[9]);
  var wn4 = WideNumber(v_indices[10], v_indices[11]);
  var wn5 = WideNumber(v_indices[12], v_indices[13]);
  var wn6 = WideNumber(v_indices[14], v_indices[15]);
  var wn7 = WideNumber(v_indices[16], v_indices[17]);
  var wn8 = WideNumber(v_indices[18], v_indices[19]);
  var wn9 = WideNumber(v_indices[20], v_indices[21]);
  var wn10 = WideNumber(v_indices[22], v_indices[23]);
  var wn11 = WideNumber(v_indices[24], v_indices[25]);
  var wn12 = WideNumber(v_indices[26], v_indices[27]);

  var countersPtr = array<WideNumber, limbs> (
    wn1, wn2, wn3, wn4, wn5, wn6, wn7, wn8, wn9, wn10, wn11, wn12
  );
  var sizesPtr = array<u32, limbs> (
    v_indices[28], v_indices[29], v_indices[30], v_indices[31], 
    v_indices[32], v_indices[33], v_indices[34], v_indices[35],
    v_indices[36], v_indices[37], v_indices[38], v_indices[39]
  );
  var atomicsPtr = array<u32, limbs> (
    v_indices[40], v_indices[41], v_indices[42], v_indices[43],
    v_indices[44], v_indices[45], v_indices[46], v_indices[47], 
    v_indices[48], v_indices[49], v_indices[50], v_indices[51]
  );
  var histogramPtr = array<u32, limbs> (
    v_indices[52], v_indices[53], v_indices[54], v_indices[55],
    v_indices[56], v_indices[57], v_indices[58], v_indices[59],
    v_indices[60], v_indices[61], v_indices[62], v_indices[63], 
  );

  var thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(v_indices[1], v_indices[1]),
    vec2<u32>(v_indices[0], v_indices[0]),
    v_indices[3],
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
@workgroup_size(1)
fn partition1024Kernel_call(@builtin(global_invocation_id) global_id: vec3u) {
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

  partition1024Kernel(pagesPtr, &sizesPtr, &countersPtr, processedScalarsPtr, points, thread, global_id);

  for(var i = 0; i < 300; i++) {
    v_indices[i] = memory.data[i];
  }
}

@compute
@workgroup_size(8)
fn sizesPrefixSumKernel_call(@builtin(global_invocation_id) global_id: vec3u) {
  var pagesPtr = array<u32, limbs> (
    v_indices[4], v_indices[5], v_indices[6], v_indices[7], 
    v_indices[8], v_indices[9], v_indices[10], v_indices[11],
    v_indices[12], v_indices[13], v_indices[14], v_indices[15]
  );

  var prefixSumSizesPtr = array<u32, limbs> (
    v_indices[16], v_indices[17], v_indices[18], v_indices[19], 
    v_indices[20], v_indices[21], v_indices[22], v_indices[23],
    v_indices[24], v_indices[25], v_indices[26], v_indices[27]
  );

  var sizesPtr = array<u32, limbs> (
    v_indices[28], v_indices[29], v_indices[30], v_indices[31], 
    v_indices[32], v_indices[33], v_indices[34], v_indices[35],
    v_indices[36], v_indices[37], v_indices[38], v_indices[39]
  );

  var wn1 = WideNumber(v_indices[40], v_indices[41]);
  var wn2 = WideNumber(v_indices[42], v_indices[43]);
  var wn3 = WideNumber(v_indices[44], v_indices[45]);
  var wn4 = WideNumber(v_indices[46], v_indices[47]);
  var wn5 = WideNumber(v_indices[48], v_indices[49]);
  var wn6 = WideNumber(v_indices[50], v_indices[51]);
  var wn7 = WideNumber(v_indices[52], v_indices[53]);
  var wn8 = WideNumber(v_indices[54], v_indices[55]);
  var wn9 = WideNumber(v_indices[56], v_indices[57]);
  var wn10 = WideNumber(v_indices[58], v_indices[59]);
  var wn11 = WideNumber(v_indices[60], v_indices[61]);
  var wn12 = WideNumber(v_indices[62], v_indices[63]);

  var countersPtr = array<WideNumber, limbs> (
    wn1, wn2, wn3, wn4, wn5, wn6, wn7, wn8, wn9, wn10, wn11, wn12
  );

  var atomicsPtr = array<u32, limbs> (
    v_indices[64], v_indices[65], v_indices[66], v_indices[67], 
    v_indices[68], v_indices[69], v_indices[70], v_indices[71],
    v_indices[72], v_indices[73], v_indices[74], v_indices[75]
  );

  var thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(v_indices[1], v_indices[1]),
    vec2<u32>(v_indices[0], v_indices[0]),
    v_indices[3],
  );

  sizesPrefixSumKernel(pagesPtr, &prefixSumSizesPtr, &sizesPtr, countersPtr, atomicsPtr, thread, global_id);

  for(var i = 0; i<12; i++) {
    v_indices[i] = prefixSumSizesPtr[i];
    v_indices[i+12] = sizesPtr[i];
  }
  for(var i = 0; i<300; i++) {
    v_indices[i+24] = memory.data[i];
  }
}

@compute
@workgroup_size(8)
fn partition4096Kernel_call(@builtin(global_invocation_id) global_id: vec3u) {
  var pointsPtr = array<u32, limbs> (
    v_indices[4], v_indices[5], v_indices[6], v_indices[7], 
    v_indices[8], v_indices[9], v_indices[10], v_indices[11],
    v_indices[12], v_indices[13], v_indices[14], v_indices[15]
  );

  var unsortedTriplePtr = array<vec4<u32>, limbs> (
    vec4<u32>(v_indices[16], v_indices[17], v_indices[18], v_indices[19]), 
    vec4<u32>(v_indices[20], v_indices[21], v_indices[22], v_indices[23]),
    vec4<u32>(v_indices[24], v_indices[25], v_indices[26], v_indices[27]),
    vec4<u32>(v_indices[28], v_indices[29], v_indices[30], v_indices[31]), 
    vec4<u32>(v_indices[32], v_indices[33], v_indices[34], v_indices[35]),
    vec4<u32>(v_indices[36], v_indices[37], v_indices[38], v_indices[39]),
    vec4<u32>(v_indices[40], v_indices[41], v_indices[42], v_indices[43]), 
    vec4<u32>(v_indices[44], v_indices[45], v_indices[46], v_indices[47]),
    vec4<u32>(v_indices[48], v_indices[49], v_indices[50], v_indices[51]),
    vec4<u32>(v_indices[52], v_indices[53], v_indices[54], v_indices[55]), 
    vec4<u32>(v_indices[56], v_indices[57], v_indices[58], v_indices[59]),
    vec4<u32>(v_indices[60], v_indices[61], v_indices[62], v_indices[63]),
  );

  var scratchPtr = array<u32, limbs> (
    v_indices[64], v_indices[65], v_indices[66], v_indices[67], 
    v_indices[68], v_indices[69], v_indices[70], v_indices[71],
    v_indices[72], v_indices[73], v_indices[74], v_indices[75]
  );

  var prefixSumSizesPtr = array<u32, limbs> (
    v_indices[76], v_indices[77], v_indices[78], v_indices[79], 
    v_indices[80], v_indices[81], v_indices[82], v_indices[83],
    v_indices[84], v_indices[85], v_indices[86], v_indices[87]
  );

  var sizesPtr = array<u32, limbs> (
    v_indices[88], v_indices[89], v_indices[90], v_indices[91], 
    v_indices[92], v_indices[93], v_indices[94], v_indices[95],
    v_indices[96], v_indices[97], v_indices[98], v_indices[99]
  );

  var pagesPtr = array<u32, limbs> (
    v_indices[100], v_indices[101], v_indices[102], v_indices[103], 
    v_indices[104], v_indices[105], v_indices[106], v_indices[107],
    v_indices[108], v_indices[109], v_indices[110], v_indices[111]
  );

  var atomicsPtr = array<u32, limbs> (
    v_indices[112], v_indices[113], v_indices[114], v_indices[115], 
    v_indices[116], v_indices[117], v_indices[118], v_indices[119],
    v_indices[120], v_indices[121], v_indices[122], v_indices[123]
  );

  var points = v_indices[124];

  var thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(v_indices[1], v_indices[1]),
    vec2<u32>(v_indices[0], v_indices[0]),
    v_indices[3],
  );

  partition4096Kernel(&pointsPtr, unsortedTriplePtr, &scratchPtr, prefixSumSizesPtr, sizesPtr, pagesPtr, atomicsPtr, points, thread, global_id);

  for(var i = 0; i<12; i++) {
    v_indices[i] = pointsPtr[i];
    v_indices[i + 12] = scratchPtr[i];
  }

  for(var i = 0; i<300; i++) {
    v_indices[i + 24] = memory.data[i];
  }
}

@compute
@workgroup_size(8)
fn histogramPrefixSumKernel_call(@builtin(global_invocation_id) global_id: vec3u) {
  var histogramPtr = array<u32, limbs> (
    v_indices[4], v_indices[5], v_indices[6], v_indices[7], 
    v_indices[8], v_indices[9], v_indices[10], v_indices[11],
    v_indices[12], v_indices[13], v_indices[14], v_indices[15]
  );

  var unsortedTriplePtr = array<u32, limbs> (
    v_indices[16], v_indices[17], v_indices[18], v_indices[19], 
    v_indices[20], v_indices[21], v_indices[22], v_indices[23],
    v_indices[24], v_indices[25], v_indices[26], v_indices[27]
  );

  var thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(v_indices[1], v_indices[1]),
    vec2<u32>(v_indices[0], v_indices[0]),
    v_indices[3],
  );

  histogramPrefixSumKernel(&histogramPtr, unsortedTriplePtr, thread, global_id);

  for(var i = 0; i<12; i++) {
    v_indices[i] = histogramPtr[i];
  }
}

@compute
@workgroup_size(8)
fn sortCountsKernel_call(@builtin(global_invocation_id) global_id: vec3u) {
  var sortedTriplePtr = array<u32, limbs> (
    v_indices[4], v_indices[5], v_indices[6], v_indices[7], 
    v_indices[8], v_indices[9], v_indices[10], v_indices[11],
    v_indices[12], v_indices[13], v_indices[14], v_indices[15]
  );

  var histogramPtr = array<u32, limbs> (
    v_indices[16], v_indices[17], v_indices[18], v_indices[19], 
    v_indices[20], v_indices[21], v_indices[22], v_indices[23],
    v_indices[24], v_indices[25], v_indices[26], v_indices[27]
  );

  var unsortedTriplePtr = array<u32, limbs> (
    v_indices[28], v_indices[29], v_indices[30], v_indices[31], 
    v_indices[32], v_indices[33], v_indices[34], v_indices[35],
    v_indices[36], v_indices[37], v_indices[38], v_indices[39]
  );

  var thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(v_indices[1], v_indices[1]),
    vec2<u32>(v_indices[0], v_indices[0]),
    v_indices[3],
  );

  sortCountsKernel(&sortedTriplePtr, histogramPtr, unsortedTriplePtr, thread, global_id);

  for(var i = 0; i<12; i++) {
    v_indices[i] = sortedTriplePtr[i];
  }

  for(var i = 0; i<300; i++) {
    v_indices[i+12] = memory.data[i];
  }
}

@compute
@workgroup_size(8)
fn computeBucketSums_call(@builtin(global_invocation_id) global_id: vec3u) {
  var bucketsPtr = array<vec4<u32>, 12> (
    vec4<u32>(v_indices[4], v_indices[5], v_indices[6], v_indices[7]),
    vec4<u32>(v_indices[8], v_indices[9], v_indices[10], v_indices[11]),
    vec4<u32>(v_indices[12], v_indices[13], v_indices[14], v_indices[15]),
    vec4<u32>(v_indices[16], v_indices[17], v_indices[18], v_indices[19]), 
    vec4<u32>(v_indices[20], v_indices[21], v_indices[22], v_indices[23]),
    vec4<u32>(v_indices[24], v_indices[25], v_indices[26], v_indices[27]),
    vec4<u32>(v_indices[28], v_indices[29], v_indices[30], v_indices[31]), 
    vec4<u32>(v_indices[32], v_indices[33], v_indices[34], v_indices[35]),
    vec4<u32>(v_indices[36], v_indices[37], v_indices[38], v_indices[39]),
    vec4<u32>(v_indices[40], v_indices[41], v_indices[42], v_indices[43]), 
    vec4<u32>(v_indices[44], v_indices[45], v_indices[46], v_indices[47]),
    vec4<u32>(v_indices[48], v_indices[49], v_indices[50], v_indices[51]),
  );

  var pointsPtr = array<u32, limbs> (
    v_indices[52], v_indices[53], v_indices[54], v_indices[55], 
    v_indices[56], v_indices[57], v_indices[58], v_indices[59],
    v_indices[60], v_indices[61], v_indices[62], v_indices[63],
  );

  var sortedTriplePtr = array<u32, limbs> (
    v_indices[64], v_indices[65], v_indices[66], v_indices[67], 
    v_indices[68], v_indices[69], v_indices[70], v_indices[71],
    v_indices[72], v_indices[73], v_indices[74], v_indices[75]
  );

  var pointIndexesPtr = array<u32, limbs> (
    v_indices[76], v_indices[77], v_indices[78], v_indices[79], 
    v_indices[80], v_indices[81], v_indices[82], v_indices[83],
    v_indices[84], v_indices[85], v_indices[86], v_indices[87]
  );

  var atomicsPtr = array<u32, limbs> (
    v_indices[88], v_indices[89], v_indices[90], v_indices[91], 
    v_indices[92], v_indices[93], v_indices[94], v_indices[95],
    v_indices[96], v_indices[97], v_indices[98], v_indices[99]
  );

  var thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(v_indices[1], v_indices[1]),
    vec2<u32>(v_indices[0], v_indices[0]),
    v_indices[3],
  );

  computeBucketSums(&bucketsPtr, pointsPtr, sortedTriplePtr, pointIndexesPtr, atomicsPtr, thread, global_id);

  var j = 0;
  for(var i = 0; i<12; i++) {
    v_indices[j] = bucketsPtr[i].x;
    v_indices[j+1] = bucketsPtr[i].y;
    v_indices[j+2] = bucketsPtr[i].z;
    v_indices[j+3] = bucketsPtr[i].w;
    j+=4;
  }

  for(var i = 0; i<300; i++) {
    v_indices[i+48] = memory.data[i];
  }
}

@compute
@workgroup_size(1)
fn reduceBuckets_call() {
  var reduced = array<vec4<u32>, 12> (
    vec4<u32>(v_indices[4], v_indices[5], v_indices[6], v_indices[7]),
    vec4<u32>(v_indices[8], v_indices[9], v_indices[10], v_indices[11]),
    vec4<u32>(v_indices[12], v_indices[13], v_indices[14], v_indices[15]),
    vec4<u32>(v_indices[16], v_indices[17], v_indices[18], v_indices[19]), 
    vec4<u32>(v_indices[20], v_indices[21], v_indices[22], v_indices[23]),
    vec4<u32>(v_indices[24], v_indices[25], v_indices[26], v_indices[27]),
    vec4<u32>(v_indices[28], v_indices[29], v_indices[30], v_indices[31]), 
    vec4<u32>(v_indices[32], v_indices[33], v_indices[34], v_indices[35]),
    vec4<u32>(v_indices[36], v_indices[37], v_indices[38], v_indices[39]),
    vec4<u32>(v_indices[40], v_indices[41], v_indices[42], v_indices[43]), 
    vec4<u32>(v_indices[44], v_indices[45], v_indices[46], v_indices[47]),
    vec4<u32>(v_indices[48], v_indices[49], v_indices[50], v_indices[51]),
  );

  var bucketsPtr = array<vec4<u32>, 12> (
    vec4<u32>(v_indices[52], v_indices[53], v_indices[54], v_indices[55]),
    vec4<u32>(v_indices[56], v_indices[57], v_indices[58], v_indices[59]),
    vec4<u32>(v_indices[60], v_indices[61], v_indices[62], v_indices[63]),
    vec4<u32>(v_indices[64], v_indices[65], v_indices[66], v_indices[67]), 
    vec4<u32>(v_indices[68], v_indices[69], v_indices[70], v_indices[71]),
    vec4<u32>(v_indices[72], v_indices[73], v_indices[74], v_indices[75]),
    vec4<u32>(v_indices[76], v_indices[77], v_indices[78], v_indices[79]), 
    vec4<u32>(v_indices[80], v_indices[81], v_indices[82], v_indices[83]),
    vec4<u32>(v_indices[84], v_indices[85], v_indices[86], v_indices[87]),
    vec4<u32>(v_indices[88], v_indices[89], v_indices[90], v_indices[91]), 
    vec4<u32>(v_indices[92], v_indices[93], v_indices[94], v_indices[95]),
    vec4<u32>(v_indices[96], v_indices[97], v_indices[98], v_indices[99]),
  );

  var thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(v_indices[1], v_indices[1]),
    vec2<u32>(v_indices[0], v_indices[0]),
    v_indices[3],
  );

  reduceBuckets(&reduced, bucketsPtr, thread);

  var j = 0;
  for(var i = 0; i<12; i++) {
    v_indices[j] = reduced[i].x;
    v_indices[j+1] = reduced[i].y;
    v_indices[j+2] = reduced[i].z;
    v_indices[j+3] = reduced[i].w;
    j+=4;
  }

  for(var i = 0; i<300; i++) {
    v_indices[i+48] = memory.data[i];
  }
}

@compute
@workgroup_size(8)
fn precomputePointsKernel_call() {
  var pointsPtr = array<vec4<u32>, 6> (
    vec4<u32>(v_indices[4], v_indices[5], v_indices[6], v_indices[7]),
    vec4<u32>(v_indices[8], v_indices[9], v_indices[10], v_indices[11]),
    vec4<u32>(v_indices[12], v_indices[13], v_indices[14], v_indices[15]),
    vec4<u32>(v_indices[16], v_indices[17], v_indices[18], v_indices[19]), 
    vec4<u32>(v_indices[20], v_indices[21], v_indices[22], v_indices[23]),
    vec4<u32>(v_indices[24], v_indices[25], v_indices[26], v_indices[27]),
  );

  var affinePointsPtr = array<u32, 24> (
    v_indices[28], v_indices[29], v_indices[30], v_indices[31], 
    v_indices[32], v_indices[33], v_indices[34], v_indices[35],
    v_indices[36], v_indices[37], v_indices[38], v_indices[39],
    v_indices[40], v_indices[41], v_indices[42], v_indices[43], 
    v_indices[44], v_indices[45], v_indices[46], v_indices[47],
    v_indices[48], v_indices[49], v_indices[50], v_indices[51],
  );

  var pointCount = v_indices[52];

  var thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(v_indices[1], v_indices[1]),
    vec2<u32>(v_indices[0], v_indices[0]),
    v_indices[3],
  );

  precomputePointsKernel(&pointsPtr, affinePointsPtr, pointCount, thread);

  var j = 0;
  for(var i = 0; i<24; i+=4) {
    v_indices[i] = pointsPtr[j].x;
    v_indices[i+1] = pointsPtr[j].y;
    v_indices[i+2] = pointsPtr[j].z;
    v_indices[i+3] = pointsPtr[j].w;
    j++;
  }

  // for(var i = 0; i<300; i++) {
  //   v_indices[i + 24] = memory.data[i];
  // }
}