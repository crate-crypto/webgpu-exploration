@compute
@workgroup_size(16)
fn setZero_batch_call() {
  var field: array<u32, limbs>;

  for(var i = 0; i<60000; i+=12) { 
    workgroupBarrier();
    field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );
    workgroupBarrier();
    setZero(&field);
  }
}

@compute
@workgroup_size(1)
fn setOne_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    setOne(&field);
  }
}

@compute
@workgroup_size(1)
fn setR_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    setRSquared(&field);
  }
}

@compute
@workgroup_size(1)
fn set_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var r: array<u32, limbs>;

    set_(&r, field);
  }
}

@compute
@workgroup_size(1)
fn load_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<vec4<u32>, 3> (
      vec4<u32>(v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3]),
      vec4<u32>(v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7]),
      vec4<u32>(v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11])
    );

    var r: array<u32, limbs>;

    load(&r, field);
  }
}

@compute
@workgroup_size(1)
fn store_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var r: array<vec4<u32>, 3>;

    store(&r, field);
  }
}

@compute
@workgroup_size(1)
fn isZero_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    isZero(field);
  }
}

@compute
@workgroup_size(1)
fn addN_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var r: array<u32, limbs>;

    addN(&r, field);
  }
}

@compute
@workgroup_size(1)
fn add_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var a: array<u32, limbs>;
    var b: array<u32, limbs>;

    add(&a, field, b);
  }
}

@compute
@workgroup_size(1)
fn sub_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var a: array<u32, limbs>;
    var b: array<u32, limbs>;

    sub(&a, field, b);
  }
}

@compute
@workgroup_size(1)
fn mul_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var a: array<u32, limbs>;
    var b: array<u32, limbs>;

    mul1(&a, field, b, field);
  }
}

@compute
@workgroup_size(1)
fn swap_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var a: array<u32, limbs>;

    swap(&a, &field);
  }
}

@compute
@workgroup_size(1)
fn reduce_batch_call() {
  for(var i = 0; i<60000; i+=12) { 
    var field = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var a: array<u32, limbs>;

    reduce(&a, field);
  }
}

@compute
@workgroup_size(1)
fn reduce_PointXYZZ_batch_call() {
  for(var i = 0; i<60000; i+=48) { 
    var x = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var y = array<u32, limbs> (
      v_indices[i+12], v_indices[i+13], v_indices[i+13], v_indices[i+15],
      v_indices[i+16], v_indices[i+17], v_indices[i+18], v_indices[i+19],
      v_indices[i+20], v_indices[i+21], v_indices[i+22], v_indices[i+23]
    );

    var zz = array<u32, limbs> (
      v_indices[i+24], v_indices[i+25], v_indices[i+26], v_indices[i+27],
      v_indices[i+28], v_indices[i+29], v_indices[i+30], v_indices[i+31],
      v_indices[i+32], v_indices[i+33], v_indices[i+34], v_indices[i+35]
    );

    var zzz = array<u32, limbs> (
      v_indices[i+36], v_indices[i+37], v_indices[i+38], v_indices[i+39],
      v_indices[i+40], v_indices[i+41], v_indices[i+42], v_indices[i+43],
      v_indices[i+44], v_indices[i+45], v_indices[i+46], v_indices[i+47]
    );

    var a: PointXYZZ = PointXYZZ(x,y,zz,zzz);

    reduce_PointXYZZ(&a);
  }
}

@compute
@workgroup_size(1)
fn load_PointXYZZ_batch_call() {
  for(var i = 0; i<60000; i+=48) { 
    var x = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var y = array<u32, limbs> (
      v_indices[i+12], v_indices[i+13], v_indices[i+13], v_indices[i+15],
      v_indices[i+16], v_indices[i+17], v_indices[i+18], v_indices[i+19],
      v_indices[i+20], v_indices[i+21], v_indices[i+22], v_indices[i+23]
    );

    var zz = array<u32, limbs> (
      v_indices[i+24], v_indices[i+25], v_indices[i+26], v_indices[i+27],
      v_indices[i+28], v_indices[i+29], v_indices[i+30], v_indices[i+31],
      v_indices[i+32], v_indices[i+33], v_indices[i+34], v_indices[i+35]
    );

    var zzz = array<u32, limbs> (
      v_indices[i+36], v_indices[i+37], v_indices[i+38], v_indices[i+39],
      v_indices[i+40], v_indices[i+41], v_indices[i+42], v_indices[i+43],
      v_indices[i+44], v_indices[i+45], v_indices[i+46], v_indices[i+47]
    );

    var a: PointXYZZ = PointXYZZ(x,y,zz,zzz);

    var p = array<vec4<u32>, 12>(
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
      vec4<u32>(1u,2u,3u,4u)
    );

    load_PointXYZZ(&a, p);
  }
}

@compute
@workgroup_size(1)
fn store_PointXYZZ_batch_call() {
  for(var i = 0; i<60000; i+=48) { 
    var x = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var y = array<u32, limbs> (
      v_indices[i+12], v_indices[i+13], v_indices[i+13], v_indices[i+15],
      v_indices[i+16], v_indices[i+17], v_indices[i+18], v_indices[i+19],
      v_indices[i+20], v_indices[i+21], v_indices[i+22], v_indices[i+23]
    );

    var zz = array<u32, limbs> (
      v_indices[i+24], v_indices[i+25], v_indices[i+26], v_indices[i+27],
      v_indices[i+28], v_indices[i+29], v_indices[i+30], v_indices[i+31],
      v_indices[i+32], v_indices[i+33], v_indices[i+34], v_indices[i+35]
    );

    var zzz = array<u32, limbs> (
      v_indices[i+36], v_indices[i+37], v_indices[i+38], v_indices[i+39],
      v_indices[i+40], v_indices[i+41], v_indices[i+42], v_indices[i+43],
      v_indices[i+44], v_indices[i+45], v_indices[i+46], v_indices[i+47]
    );

    var a: PointXYZZ = PointXYZZ(x,y,zz,zzz);

    var p = array<vec4<u32>, 12>(
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
      vec4<u32>(1u,2u,3u,4u)
    );

    store_PointXYZZ(a, &p);
  }
}

@compute
@workgroup_size(1)
fn normalize_batch_call() {
  for(var i = 0; i<60000; i+=48) { 
    var x = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var y = array<u32, limbs> (
      v_indices[i+12], v_indices[i+13], v_indices[i+13], v_indices[i+15],
      v_indices[i+16], v_indices[i+17], v_indices[i+18], v_indices[i+19],
      v_indices[i+20], v_indices[i+21], v_indices[i+22], v_indices[i+23]
    );

    var zz = array<u32, limbs> (
      v_indices[i+24], v_indices[i+25], v_indices[i+26], v_indices[i+27],
      v_indices[i+28], v_indices[i+29], v_indices[i+30], v_indices[i+31],
      v_indices[i+32], v_indices[i+33], v_indices[i+34], v_indices[i+35]
    );

    var zzz = array<u32, limbs> (
      v_indices[i+36], v_indices[i+37], v_indices[i+38], v_indices[i+39],
      v_indices[i+40], v_indices[i+41], v_indices[i+42], v_indices[i+43],
      v_indices[i+44], v_indices[i+45], v_indices[i+46], v_indices[i+47]
    );

    var a: HighThroughput = HighThroughput(x,y,zz,zzz, true, true);

    normalize(&a);
  }
}

@compute
@workgroup_size(1)
fn setZero_AccumulatorXYZZ_batch_call() {
  for(var i = 0; i<60000; i+=48) { 
    var x = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var y = array<u32, limbs> (
      v_indices[i+12], v_indices[i+13], v_indices[i+13], v_indices[i+15],
      v_indices[i+16], v_indices[i+17], v_indices[i+18], v_indices[i+19],
      v_indices[i+20], v_indices[i+21], v_indices[i+22], v_indices[i+23]
    );

    var zz = array<u32, limbs> (
      v_indices[i+24], v_indices[i+25], v_indices[i+26], v_indices[i+27],
      v_indices[i+28], v_indices[i+29], v_indices[i+30], v_indices[i+31],
      v_indices[i+32], v_indices[i+33], v_indices[i+34], v_indices[i+35]
    );

    var zzz = array<u32, limbs> (
      v_indices[i+36], v_indices[i+37], v_indices[i+38], v_indices[i+39],
      v_indices[i+40], v_indices[i+41], v_indices[i+42], v_indices[i+43],
      v_indices[i+44], v_indices[i+45], v_indices[i+46], v_indices[i+47]
    );

    var a: HighThroughput = HighThroughput(x,y,zz,zzz, false, false);

    setZero_HighThroughput(&a);
  }
}

@compute
@workgroup_size(1)
fn dbl_AccumulatorXYZZ_batch_call() {
  for(var i = 0; i<60000; i+=48) { 
    var x = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var y = array<u32, limbs> (
      v_indices[i+12], v_indices[i+13], v_indices[i+13], v_indices[i+15],
      v_indices[i+16], v_indices[i+17], v_indices[i+18], v_indices[i+19],
      v_indices[i+20], v_indices[i+21], v_indices[i+22], v_indices[i+23]
    );

    var zz = array<u32, limbs> (
      v_indices[i+24], v_indices[i+25], v_indices[i+26], v_indices[i+27],
      v_indices[i+28], v_indices[i+29], v_indices[i+30], v_indices[i+31],
      v_indices[i+32], v_indices[i+33], v_indices[i+34], v_indices[i+35]
    );

    var zzz = array<u32, limbs> (
      v_indices[i+36], v_indices[i+37], v_indices[i+38], v_indices[i+39],
      v_indices[i+40], v_indices[i+41], v_indices[i+42], v_indices[i+43],
      v_indices[i+44], v_indices[i+45], v_indices[i+46], v_indices[i+47]
    );

    var a: HighThroughput = HighThroughput(x,y,zz,zzz, false, false);

    dbl(&a);
  }
}

@compute
@workgroup_size(1)
fn add_AccumulatorXYZZ_batch_call() {
  for(var i = 0; i<60000; i+=48) { 
    var x = array<u32, limbs> (
      v_indices[i], v_indices[i+1], v_indices[i+2], v_indices[i+3],
      v_indices[i+4], v_indices[i+5], v_indices[i+6], v_indices[i+7],
      v_indices[i+8], v_indices[i+9], v_indices[i+10], v_indices[i+11]
    );

    var y = array<u32, limbs> (
      v_indices[i+12], v_indices[i+13], v_indices[i+13], v_indices[i+15],
      v_indices[i+16], v_indices[i+17], v_indices[i+18], v_indices[i+19],
      v_indices[i+20], v_indices[i+21], v_indices[i+22], v_indices[i+23]
    );

    var zz = array<u32, limbs> (
      v_indices[i+24], v_indices[i+25], v_indices[i+26], v_indices[i+27],
      v_indices[i+28], v_indices[i+29], v_indices[i+30], v_indices[i+31],
      v_indices[i+32], v_indices[i+33], v_indices[i+34], v_indices[i+35]
    );

    var zzz = array<u32, limbs> (
      v_indices[i+36], v_indices[i+37], v_indices[i+38], v_indices[i+39],
      v_indices[i+40], v_indices[i+41], v_indices[i+42], v_indices[i+43],
      v_indices[i+44], v_indices[i+45], v_indices[i+46], v_indices[i+47]
    );

    var a: HighThroughput = HighThroughput(x,y,zz,zzz, true, true);
    var p: PointXYZZ;

    add2_HighThroughput(&a, &p, false);
  }
}

@compute
@workgroup_size(8)
fn precomputePointsKernel_batch_call() {

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
  for(var i = 0; i<60000; i+=1) { 
    precomputePointsKernel(&pointsPtr, affinePointsPtr, pointCount, thread);
  }
}