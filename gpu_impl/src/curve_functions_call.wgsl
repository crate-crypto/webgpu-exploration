@compute
@workgroup_size(1)
fn setZero_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  setZero(&field);

  for(var i = 0; i<12; i++) {
    v_indices[i] = field[i];
  }
}

@compute
@workgroup_size(1)
fn setOne_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  setOne(&field);

  for(var i = 0; i<12; i++) {
    v_indices[i] = field[i];
  }
}

@compute
@workgroup_size(1)
fn setRSquared_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  setRSquared(&field);

  for(var i = 0; i<12; i++) {
    v_indices[i] = field[i];
  }
}

@compute
@workgroup_size(1)
fn set_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var a = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  set_(&field, a);

  for(var i = 0; i<12; i++) {
    v_indices[i] = field[i];
  }
}

@compute
@workgroup_size(1)
fn load_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var pointer = array<vec4<u32>, 3> (
    vec4<u32>(v_indices[12], v_indices[13], v_indices[14], v_indices[15]),
    vec4<u32>(v_indices[16], v_indices[17], v_indices[18], v_indices[19]),
    vec4<u32>(v_indices[20], v_indices[21], v_indices[22], v_indices[23])
  );

  load(&field, pointer);

  for(var i = 0; i<12; i++) {
    v_indices[i] = field[i];
  }
}

@compute
@workgroup_size(1)
fn store_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var pointer = array<vec4<u32>, 3> (
    vec4<u32>(v_indices[12], v_indices[13], v_indices[14], v_indices[15]),
    vec4<u32>(v_indices[16], v_indices[17], v_indices[18], v_indices[19]),
    vec4<u32>(v_indices[20], v_indices[21], v_indices[22], v_indices[23])
  );

  store(&pointer, field);

  var j = 0;
  for(var i = 0; i<12; i+=4) {
    v_indices[i] = pointer[j].x;
    v_indices[i+1] = pointer[j].y;
    v_indices[i+2] = pointer[j].z;
    v_indices[i+3] = pointer[j].w;
    j++;
  }
}

@compute
@workgroup_size(1)
fn isZero_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  v_indices[0] = u32(isZero(field));
}

@compute
@workgroup_size(1)
fn addN_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var a = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  addN(&field, a);

  for(var i = 0; i<12; i++) {
    v_indices[i] = field[i];
  }
}

@compute
@workgroup_size(1)
fn add_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var a = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  var b = array<u32, limbs> (
    v_indices[24], v_indices[25], v_indices[26], v_indices[27],
    v_indices[28], v_indices[29], v_indices[30], v_indices[31],
    v_indices[32], v_indices[33], v_indices[34], v_indices[35]
  );

  add(&field, a, b);

  for(var i = 0; i<12; i++) {
    v_indices[i] = field[i];
  }
}

@compute
@workgroup_size(1)
fn sub_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var a = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  var b = array<u32, limbs> (
    v_indices[24], v_indices[25], v_indices[26], v_indices[27],
    v_indices[28], v_indices[29], v_indices[30], v_indices[31],
    v_indices[32], v_indices[33], v_indices[34], v_indices[35]
  );

  sub(&field, a, b);

  for(var i = 0; i<12; i++) {
    v_indices[i] = field[i];
  }
}

@compute
@workgroup_size(1)
fn mul_call() {
  var field = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var a = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  var b = array<u32, limbs> (
    v_indices[24], v_indices[25], v_indices[26], v_indices[27],
    v_indices[28], v_indices[29], v_indices[30], v_indices[31],
    v_indices[32], v_indices[33], v_indices[34], v_indices[35]
  );

  mul1(&field, a, b, a);

  for(var i = 0; i<12; i++) {
    v_indices[i] = field[i];
  }
}

@compute
@workgroup_size(1)
fn swap_call() {
  var a = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var b = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  swap(&a, &b);

  for(var i = 0; i<12; i++) {
    v_indices[i] = a[i];
  }
}

@compute
@workgroup_size(1)
fn reduce_call() {
  var r = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var field = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  reduce(&r, field);

  for(var i = 0; i<12; i++) {
    v_indices[i] = r[i];
  }
}

@compute
@workgroup_size(1)
fn reduce_PointXYZZ_call() {
  var x = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var y = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  var zz = array<u32, limbs> (
    v_indices[24], v_indices[25], v_indices[26], v_indices[27],
    v_indices[28], v_indices[29], v_indices[30], v_indices[31],
    v_indices[32], v_indices[33], v_indices[34], v_indices[35]
  );

  var zzz = array<u32, limbs> (
    v_indices[36], v_indices[37], v_indices[38], v_indices[39],
    v_indices[40], v_indices[41], v_indices[42], v_indices[43],
    v_indices[44], v_indices[45], v_indices[46], v_indices[47]
  );

  var p: PointXYZZ = PointXYZZ(x, y, zz, zzz);

  reduce_PointXYZZ(&p);

  var j = 0;
  for(var i = 0; i<48; i+=4) {
    v_indices[i] = p.x[j];
    v_indices[i+1] = p.y[j];
    v_indices[i+2] = p.zz[j];
    v_indices[i+3] = p.zzz[j];
    j++;
  }
}

@compute
@workgroup_size(1)
fn load_PointXYZZ_call() {
  var x = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var y = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  var zz = array<u32, limbs> (
    v_indices[24], v_indices[25], v_indices[26], v_indices[27],
    v_indices[28], v_indices[29], v_indices[30], v_indices[31],
    v_indices[32], v_indices[33], v_indices[34], v_indices[35]
  );

  var zzz = array<u32, limbs> (
    v_indices[36], v_indices[37], v_indices[38], v_indices[39],
    v_indices[40], v_indices[41], v_indices[42], v_indices[43],
    v_indices[44], v_indices[45], v_indices[46], v_indices[47]
  );

  var pt = array<vec4<u32>, 12>(
    vec4<u32>(v_indices[48], v_indices[49], v_indices[50], v_indices[51]),
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
  );

  var p: PointXYZZ = PointXYZZ(x, y, zz, zzz);

  load_PointXYZZ(&p, pt);

  var j = 0;
  for(var i = 0; i<48; i+=4) {
    v_indices[i] = p.x[j];
    v_indices[i+1] = p.y[j];
    v_indices[i+2] = p.zz[j];
    v_indices[i+3] = p.zzz[j];
    j++;
  }
}

@compute
@workgroup_size(1)
fn store_PointXYZZ_call() {
  var x = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var y = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  var zz = array<u32, limbs> (
    v_indices[24], v_indices[25], v_indices[26], v_indices[27],
    v_indices[28], v_indices[29], v_indices[30], v_indices[31],
    v_indices[32], v_indices[33], v_indices[34], v_indices[35]
  );

  var zzz = array<u32, limbs> (
    v_indices[36], v_indices[37], v_indices[38], v_indices[39],
    v_indices[40], v_indices[41], v_indices[42], v_indices[43],
    v_indices[44], v_indices[45], v_indices[46], v_indices[47]
  );

  var pt = array<vec4<u32>, 12>(
    vec4<u32>(v_indices[48], v_indices[49], v_indices[50], v_indices[51]),
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
  );

  var p: PointXYZZ = PointXYZZ(x, y, zz, zzz);

  store_PointXYZZ(p, &pt);

  var j = 0;
  for(var i = 0; i<48; i+=4) {
    v_indices[i] = p.x[j];
    v_indices[i+1] = p.y[j];
    v_indices[i+2] = p.zz[j];
    v_indices[i+3] = p.zzz[j];
    j++;
  }
}

@compute
@workgroup_size(1)
fn normalize_call() {
  var x = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var y = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  var zz = array<u32, limbs> (
    v_indices[24], v_indices[25], v_indices[26], v_indices[27],
    v_indices[28], v_indices[29], v_indices[30], v_indices[31],
    v_indices[32], v_indices[33], v_indices[34], v_indices[35]
  );

  var zzz = array<u32, limbs> (
    v_indices[36], v_indices[37], v_indices[38], v_indices[39],
    v_indices[40], v_indices[41], v_indices[42], v_indices[43],
    v_indices[44], v_indices[45], v_indices[46], v_indices[47]
  );

  var p: HighThroughput = HighThroughput(x, y, zz, zzz, true, false);

  normalize(&p);

  var j = 0;
  for(var i = 0; i<48; i+=4) {
    v_indices[i] = p.x[j];
    v_indices[i+1] = p.y[j];
    v_indices[i+2] = p.zz[j];
    v_indices[i+3] = p.zzz[j];
    j++;
  }
}

@compute
@workgroup_size(1)
fn setZero_HighThroughput_call() {
  var x = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var y = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  var zz = array<u32, limbs> (
    v_indices[24], v_indices[25], v_indices[26], v_indices[27],
    v_indices[28], v_indices[29], v_indices[30], v_indices[31],
    v_indices[32], v_indices[33], v_indices[34], v_indices[35]
  );

  var zzz = array<u32, limbs> (
    v_indices[36], v_indices[37], v_indices[38], v_indices[39],
    v_indices[40], v_indices[41], v_indices[42], v_indices[43],
    v_indices[44], v_indices[45], v_indices[46], v_indices[47]
  );

  var p: HighThroughput = HighThroughput(x, y, zz, zzz, false, false);

  setZero_HighThroughput(&p);

  var j = 0;
  for(var i = 0; i<48; i+=4) {
    v_indices[i] = p.x[j];
    v_indices[i+1] = p.y[j];
    v_indices[i+2] = p.zz[j];
    v_indices[i+3] = p.zzz[j];
    j++;
  }
}

@compute
@workgroup_size(1)
fn dbl_call() {
  var x = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var y = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  var zz = array<u32, limbs> (
    v_indices[24], v_indices[25], v_indices[26], v_indices[27],
    v_indices[28], v_indices[29], v_indices[30], v_indices[31],
    v_indices[32], v_indices[33], v_indices[34], v_indices[35]
  );

  var zzz = array<u32, limbs> (
    v_indices[36], v_indices[37], v_indices[38], v_indices[39],
    v_indices[40], v_indices[41], v_indices[42], v_indices[43],
    v_indices[44], v_indices[45], v_indices[46], v_indices[47]
  );

  var p: HighThroughput = HighThroughput(x, y, zz, zzz, false, false);

  dbl(&p);

  var j = 0;
  for(var i = 0; i<48; i+=4) {
    v_indices[i] = p.x[j];
    v_indices[i+1] = p.y[j];
    v_indices[i+2] = p.zz[j];
    v_indices[i+3] = p.zzz[j];
    j++;
  }
}

@compute
@workgroup_size(1)
fn add2_HighThroughput_call() {
  var x = array<u32, limbs> (
    v_indices[0], v_indices[1], v_indices[2], v_indices[3],
    v_indices[4], v_indices[5], v_indices[6], v_indices[7],
    v_indices[8], v_indices[9], v_indices[10], v_indices[11]
  );

  var y = array<u32, limbs> (
    v_indices[12], v_indices[13], v_indices[14], v_indices[15],
    v_indices[16], v_indices[17], v_indices[18], v_indices[19],
    v_indices[20], v_indices[21], v_indices[22], v_indices[23]
  );

  var zz = array<u32, limbs> (
    v_indices[24], v_indices[25], v_indices[26], v_indices[27],
    v_indices[28], v_indices[29], v_indices[30], v_indices[31],
    v_indices[32], v_indices[33], v_indices[34], v_indices[35]
  );

  var zzz = array<u32, limbs> (
    v_indices[36], v_indices[37], v_indices[38], v_indices[39],
    v_indices[40], v_indices[41], v_indices[42], v_indices[43],
    v_indices[44], v_indices[45], v_indices[46], v_indices[47]
  );

  var p: HighThroughput = HighThroughput(x, y, zz, zzz, true, true);
  var p2: PointXYZZ = PointXYZZ(x, y, zz, zzz);

  add2_HighThroughput(&p, &p2, false);

  var j = 0;
  for(var i = 0; i<48; i+=4) {
    v_indices[i] = p.x[j];
    v_indices[i+1] = p.y[j];
    v_indices[i+2] = p.zz[j];
    v_indices[i+3] = p.zzz[j];
    j++;
  }
}

fn add_two(a: u32, b: u32) -> u32 {
  return a + b;
}

@compute
@workgroup_size(1)
fn add_two_call() {
  var a = v_indices[0];
  var b = v_indices[1];

  var c = add_two(a, b);

  v_indices[0] = c;
}

fn add_two_vec(a: ptr<function, array<u32, 1000000>>, b: ptr<function,array<u32, 1000000>>) -> array<u32, 1000000> {
  var res: array<u32, 1000000>;
  var j = 0;
  for(var j = 0; j<100; j++) {
    for(var i = 0; i<10000; i++) {
      res[i + j * 10000] = (*a)[i + j * 10000] + (*b)[i + j * 10000];
    }
  }
  return res;
}

@compute
@workgroup_size(1)
fn add_two_vec_call() {
  var a: array<u32, 1000000>;
  var b: array<u32, 1000000>;
  var j = 0;
  for(var j = 0; j<100; j++) {
    for(var i = 0; i<10000; i++) {
      a[i + j * 10000] = v_indices[i + j * 10000];
      b[i + j * 10000] = v_indices[i + j * 10000];
    }  
  }

  var c = add_two_vec(&a, &b);
}