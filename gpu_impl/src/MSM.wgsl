const NBUCKETS: u32 = 0x00400000u;
const PAGE_SIZE: u32 = 31744u;

const SCRATCH_MAX_COUNT: u32 = 9126u;
const SIZE_LIMIT: u32 = 283840u; // (SCRATCH_MAX_COUNT-256)*32
const SCRATCH_REQUIRED: u32 = 1460160u; // SCRATCH_MAX_COUNT*160

@compute
@workgroup_size(1)
fn MSM_run_call() {
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

  processSignedDigitsKernel(processedScalarData, &scalarData, v_indices[28], thread);

  var wn1 = WideNumber(v_indices[29], v_indices[30]);
  var wn2 = WideNumber(v_indices[31], v_indices[32]);
  var wn3 = WideNumber(v_indices[33], v_indices[34]);
  var wn4 = WideNumber(v_indices[35], v_indices[36]);
  var wn5 = WideNumber(v_indices[37], v_indices[38]);
  var wn6 = WideNumber(v_indices[39], v_indices[40]);
  var wn7 = WideNumber(v_indices[41], v_indices[42]);
  var wn8 = WideNumber(v_indices[43], v_indices[44]);
  var wn9 = WideNumber(v_indices[45], v_indices[46]);
  var wn10 = WideNumber(v_indices[47], v_indices[48]);
  var wn11 = WideNumber(v_indices[49], v_indices[50]);
  var wn12 = WideNumber(v_indices[51], v_indices[52]);

  var countersPtr = array<WideNumber, limbs> (
    wn1, wn2, wn3, wn4, wn5, wn6, wn7, wn8, wn9, wn10, wn11, wn12
  );
  var sizesPtr = array<u32, limbs> (
    v_indices[53], v_indices[54], v_indices[55], v_indices[56], 
    v_indices[57], v_indices[58], v_indices[59], v_indices[60],
    v_indices[61], v_indices[62], v_indices[63], v_indices[64]
  );
  var atomicsPtr = array<u32, limbs> (
    v_indices[65], v_indices[66], v_indices[67], v_indices[68],
    v_indices[69], v_indices[70], v_indices[71], v_indices[72], 
    v_indices[73], v_indices[74], v_indices[75], v_indices[76]
  );
  var histogramPtr = array<u32, limbs> (
    v_indices[77], v_indices[78], v_indices[79], v_indices[80],
    v_indices[81], v_indices[82], v_indices[83], v_indices[84],
    v_indices[85], v_indices[86], v_indices[87], v_indices[88], 
  );

  initializeCountersSizesAtomicsHistogramKernel(&countersPtr, &sizesPtr, &atomicsPtr, &histogramPtr, thread);

  var pagesPtr = array<u32, limbs> (
    v_indices[89], v_indices[90], v_indices[91], v_indices[92], 
    v_indices[93], v_indices[94], v_indices[95], v_indices[96],
    v_indices[97], v_indices[98], v_indices[99], v_indices[100]
  );

  var processedScalarsPtr = array<u32, limbs> (
    v_indices[101], v_indices[102], v_indices[103], v_indices[104], 
    v_indices[105], v_indices[106], v_indices[107], v_indices[108],
    v_indices[109], v_indices[110], v_indices[111], v_indices[112]
  );

  var points = v_indices[113];

  partition1024Kernel(pagesPtr, &sizesPtr, &countersPtr, processedScalarsPtr, points, thread);

  var prefixSumSizesPtr = array<u32, limbs> (
    v_indices[114], v_indices[115], v_indices[116], v_indices[117], 
    v_indices[118], v_indices[119], v_indices[120], v_indices[121],
    v_indices[122], v_indices[123], v_indices[124], v_indices[125]
  );

  sizesPrefixSumKernel(pagesPtr, &prefixSumSizesPtr, &sizesPtr, countersPtr, atomicsPtr, thread);

  var pointsPtr = array<u32, limbs> (
    v_indices[126], v_indices[127], v_indices[128], v_indices[129], 
    v_indices[130], v_indices[131], v_indices[132], v_indices[133],
    v_indices[134], v_indices[135], v_indices[136], v_indices[137]
  );

  var unsortedTriplePtr = array<vec4<u32>, limbs> (
    vec4<u32>(v_indices[138], v_indices[139], v_indices[140], v_indices[141]), 
    vec4<u32>(v_indices[142], v_indices[143], v_indices[144], v_indices[145]),
    vec4<u32>(v_indices[146], v_indices[147], v_indices[148], v_indices[149]),
    vec4<u32>(v_indices[150], v_indices[151], v_indices[152], v_indices[153]), 
    vec4<u32>(v_indices[154], v_indices[155], v_indices[156], v_indices[157]),
    vec4<u32>(v_indices[158], v_indices[159], v_indices[160], v_indices[161]),
    vec4<u32>(v_indices[162], v_indices[163], v_indices[164], v_indices[165]), 
    vec4<u32>(v_indices[166], v_indices[167], v_indices[168], v_indices[169]),
    vec4<u32>(v_indices[170], v_indices[171], v_indices[172], v_indices[173]),
    vec4<u32>(v_indices[174], v_indices[175], v_indices[176], v_indices[177]), 
    vec4<u32>(v_indices[178], v_indices[179], v_indices[180], v_indices[181]),
    vec4<u32>(v_indices[182], v_indices[183], v_indices[184], v_indices[185]),
  );

  var scratchPtr = array<u32, limbs> (
    v_indices[186], v_indices[187], v_indices[188], v_indices[189], 
    v_indices[190], v_indices[191], v_indices[192], v_indices[193],
    v_indices[194], v_indices[195], v_indices[196], v_indices[197]
  );

  partition4096Kernel(&pointsPtr, unsortedTriplePtr, &scratchPtr, prefixSumSizesPtr, sizesPtr, pagesPtr, atomicsPtr, points, thread);

  var unsortedTriplePtr1 = array<u32, limbs> (
    v_indices[198], v_indices[199], v_indices[200], v_indices[201], 
    v_indices[202], v_indices[203], v_indices[204], v_indices[205],
    v_indices[206], v_indices[207], v_indices[208], v_indices[209]
  );

  histogramPrefixSumKernel(&histogramPtr, unsortedTriplePtr1, thread);

  var sortedTriplePtr = array<u32, limbs> (
    v_indices[210], v_indices[211], v_indices[212], v_indices[213], 
    v_indices[214], v_indices[215], v_indices[216], v_indices[217],
    v_indices[218], v_indices[219], v_indices[220], v_indices[221]
  );

  sortCountsKernel(&sortedTriplePtr, histogramPtr, unsortedTriplePtr1, thread);

  var bucketsPtr = array<vec4<u32>, 12> (
    vec4<u32>(v_indices[222], v_indices[223], v_indices[224], v_indices[225]),
    vec4<u32>(v_indices[226], v_indices[227], v_indices[228], v_indices[229]),
    vec4<u32>(v_indices[230], v_indices[231], v_indices[232], v_indices[233]),
    vec4<u32>(v_indices[234], v_indices[235], v_indices[236], v_indices[237]), 
    vec4<u32>(v_indices[238], v_indices[239], v_indices[240], v_indices[241]),
    vec4<u32>(v_indices[242], v_indices[243], v_indices[244], v_indices[245]),
    vec4<u32>(v_indices[246], v_indices[247], v_indices[248], v_indices[249]), 
    vec4<u32>(v_indices[250], v_indices[251], v_indices[252], v_indices[253]),
    vec4<u32>(v_indices[254], v_indices[255], v_indices[256], v_indices[257]),
    vec4<u32>(v_indices[258], v_indices[259], v_indices[260], v_indices[261]), 
    vec4<u32>(v_indices[262], v_indices[263], v_indices[264], v_indices[265]),
    vec4<u32>(v_indices[266], v_indices[267], v_indices[268], v_indices[269]),
  );

  var pointIndexesPtr = array<u32, limbs> (
    v_indices[270], v_indices[271], v_indices[272], v_indices[273], 
    v_indices[274], v_indices[275], v_indices[276], v_indices[277],
    v_indices[278], v_indices[279], v_indices[280], v_indices[281]
  );

  computeBucketSums(&bucketsPtr, pointsPtr, sortedTriplePtr, pointIndexesPtr, atomicsPtr, thread);

  for(var i = 0; i < 48; i++) {
    v_indices[i] = memory.data[i];
  }
}
