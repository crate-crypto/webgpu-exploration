fn precomputePointsKernel(pointsPtr: ptr<function, array<vec4<u32>, 6>>, affinePointsPtr: array<u32, 24>, pointCount: u32, thread: Thread) {
  var acc: HighThroughput;
  var point: PointXY;
  var result: PointXY;

  var globalTID = thread.blockIdx.x*thread.blockDim.x+thread.threadIdx.x;
  var globalStride = thread.blockDim.x*thread.gridDim.x;

  var SHMData_vec4: array<vec4<u32>, 96>;
  var SHMData_copy = SHMData;
  var j = 0;
  for(var i = 0; i<2*i32(limbs); i+=4) {
    SHMData_vec4[j].x = SHMData_copy[i];
    SHMData_vec4[j].y = SHMData_copy[i+1];
    SHMData_vec4[j].z = SHMData_copy[i+2];
    SHMData_vec4[j].w = SHMData_copy[i+3];
    j++;
  }
  copyToShared(SHMData_vec4, thread);

  var affinePointsPtr_vec4: array<vec4<u32>, 6>;
  var affinePointsPtr_copy = affinePointsPtr;
  j = 0;
  for(var i = 0; i<24; i+=4) {
    affinePointsPtr_vec4[j].x = affinePointsPtr_copy[i];
    affinePointsPtr_vec4[j].y = affinePointsPtr_copy[i+1];
    affinePointsPtr_vec4[j].z = affinePointsPtr_copy[i+2];
    affinePointsPtr_vec4[j].w = affinePointsPtr_copy[i+3];
    j++;
  }


  for(var i = globalTID; i<pointCount; i+=globalStride) {
    loadUnaligned_PointXY(&point, &affinePointsPtr_vec4);

    store_PointXY(point, pointsPtr);
    setZero_HighThroughput(&acc);
    add1_HighThroughput(&acc, &point, true);
    workgroupBarrier();
    // for(var j = 1; j<6; j++) {
    //   for(var k = 0; k<46; k++) {
        dbl(&acc);
    //   }
    // }
    workgroupBarrier();
  }
  var p_xy = normalize(&acc);
  store_PointXY(p_xy, pointsPtr);
}