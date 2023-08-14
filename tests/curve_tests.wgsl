@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience

@compute
@workgroup_size(1)
fn qTerm_test() {
  v_indices[0] = qTerm(v_indices[0]);
}

@compute
@workgroup_size(1)
fn initialize_PointXY_test() {
  var arr = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  var arr0 = array<u32, limbs>(0u, 0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);

  var point: PointXY = PointXY(arr0, arr0);
  initialize_PointXY(&point, arr, arr);

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i] = point.x[i];
  }

  for (var i : i32 = 12; i < i32(2u*limbs); i = i + 1) {
    v_indices[i] = point.y[i - 12];
  }
}

@compute
@workgroup_size(1)
fn isEqual_test() {
  var arr = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  v_indices[0] = u32(isEqual(arr, arr));
}

@compute
@workgroup_size(1)
fn set_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr1 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  set_(&arr, arr1);

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn swap_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr1 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  swap(&arr, &arr1);

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i] = arr[i];
  }

  for (var i : i32 = 12; i < i32(2u*limbs); i = i + 1) {
    v_indices[i] = arr1[i];
  }
}

@compute
@workgroup_size(1)
fn initialize_PointXYZZ_test() {
  var arr = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  var arr0 = array<u32, limbs>(0u, 0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);

  var point: PointXYZZ = PointXYZZ(arr0, arr0, arr0, arr0);
  initialize_PointXYZZ(&point, arr, arr, arr, arr);

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i] = point.x[i];
  }

  for (var i : i32 = 12; i < i32(2u*limbs); i = i + 1) {
    v_indices[i] = point.y[i - 12];
  }
}

@compute
@workgroup_size(1)
fn load_test() {
  var arr: array<u32, limbs>;
  var v1 = vec4<u32>(0u,1u,2u,3u);
  var v2 = vec4<u32>(4u,5u,6u,7u);
  var v3 = vec4<u32>(8u,9u,10u,11u);
  var arr0 = array<vec4<u32>, 3>(v1,v2,v3);

  load(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn store_test() {
  var arr = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  var arr0: array<vec4<u32>, 3>;

  store(&arr0, arr);

  var j = 0;
  for (var i : i32 = 0; i < i32(limbs); i = i + 4) {
    v_indices[i] = arr0[j].x;
    v_indices[i+1] = arr0[j].y;
    v_indices[i+2] = arr0[j].z;
    v_indices[i+3] = arr0[j].w;
    j++;
  }
}

@compute
@workgroup_size(1)
fn load_PointXY_test() {
  var v1 = vec4<u32>(0u,1u,2u,3u);
  var v2 = vec4<u32>(4u,5u,6u,7u);
  var v3 = vec4<u32>(8u,9u,10u,11u);
  var v4 = vec4<u32>(0u,1u,2u,3u);
  var v5 = vec4<u32>(4u,5u,6u,7u);
  var v6 = vec4<u32>(8u,9u,10u,100u);
  var arr0 = array<vec4<u32>, 6>(v1,v2,v3,v4,v5,v6);

  var point: PointXY;

  load_PointXY(&point, arr0);

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i] = point.x[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+12] = point.y[i];
  }
}

@compute
@workgroup_size(1)
fn loadUnaligned_PointXY_test() {
  var v1 = vec4<u32>(0u,1u,2u,3u);
  var v2 = vec4<u32>(4u,5u,6u,7u);
  var v3 = vec4<u32>(8u,9u,10u,11u);
  var v4 = vec4<u32>(0u,1u,2u,3u);
  var v5 = vec4<u32>(4u,5u,6u,7u);
  var v6 = vec4<u32>(8u,9u,10u,101u);
  var arr0 = array<vec4<u32>, 6>(v1,v2,v3,v4,v5,v6);

  var point: PointXY;

  loadUnaligned_PointXY(&point, &arr0);

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i] = point.x[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+12] = point.y[i];
  }
}

@compute
@workgroup_size(1)
fn store_PointXY_test() {
  var arr = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);


  var point: PointXY = PointXY(arr, arr0);

  var arr_store: array<vec4<u32>, 6>;

  store_PointXY(point, &arr_store);

  var j = 0;
  for (var i : i32 = 0; i < 2 * i32(limbs); i = i + 4) {
    v_indices[i] = arr_store[j].x;
    v_indices[i+1] = arr_store[j].y;
    v_indices[i+2] = arr_store[j].z;
    v_indices[i+3] = arr_store[j].w;
    j++;
  }
}

@compute
@workgroup_size(1)
fn load_PointXYZZ_test() {
  var v1 = vec4<u32>(0u,1u,2u,3u);
  var v2 = vec4<u32>(4u,5u,6u,7u);
  var v3 = vec4<u32>(8u,9u,10u,11u);
  var v4 = vec4<u32>(0u,1u,2u,3u);
  var v5 = vec4<u32>(4u,5u,6u,7u);
  var v6 = vec4<u32>(8u,9u,10u,100u);
  var v7 = vec4<u32>(0u,1u,2u,3u);
  var v8 = vec4<u32>(4u,5u,6u,7u);
  var v9 = vec4<u32>(8u,9u,10u,11u);
  var v10 = vec4<u32>(0u,1u,2u,3u);
  var v11 = vec4<u32>(4u,5u,6u,7u);
  var v12 = vec4<u32>(8u,9u,10u,200u);
  var arr0 = array<vec4<u32>, 12>(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12);

  var point: PointXYZZ;

  load_PointXYZZ(&point, arr0);

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i] = point.x[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+12] = point.y[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+24] = point.zz[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+36] = point.zzz[i];
  }
}

@compute
@workgroup_size(1)
fn loadUnaligned_PointXYZZ_test() {
  var v1 = vec4<u32>(0u,1u,2u,3u);
  var v2 = vec4<u32>(4u,5u,6u,7u);
  var v3 = vec4<u32>(8u,9u,10u,11u);
  var v4 = vec4<u32>(0u,1u,2u,3u);
  var v5 = vec4<u32>(4u,5u,6u,7u);
  var v6 = vec4<u32>(8u,9u,10u,100u);
  var v7 = vec4<u32>(0u,1u,2u,3u);
  var v8 = vec4<u32>(4u,5u,6u,7u);
  var v9 = vec4<u32>(8u,9u,10u,11u);
  var v10 = vec4<u32>(0u,1u,2u,3u);
  var v11 = vec4<u32>(4u,5u,6u,7u);
  var v12 = vec4<u32>(8u,9u,10u,200u);
  var arr0 = array<vec4<u32>, 12>(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12);

  var point: PointXYZZ;

  loadUnaligned_PointXYZZ(&point, &arr0);

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i] = point.x[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+12] = point.y[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+24] = point.zz[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+36] = point.zzz[i];
  }
}

@compute
@workgroup_size(1)
fn store_PointXYZZ_test() {
  var arr = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);
  var arr1 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);
  var arr2 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,200u);

  var point: PointXYZZ = PointXYZZ(arr, arr0, arr1, arr2);

  var arr_store: array<vec4<u32>, 12>;

  store_PointXYZZ(point, &arr_store);

  var j = 0;
  for (var i : i32 = 0; i < 4 * i32(limbs); i = i + 4) {
    v_indices[i] = arr_store[j].x;
    v_indices[i+1] = arr_store[j].y;
    v_indices[i+2] = arr_store[j].z;
    v_indices[i+3] = arr_store[j].w;
    j++;
  }
}

@compute
@workgroup_size(1)
fn setZero_HighThroughput_test() {
  var acc: HighThroughput;

  setZero_HighThroughput(&acc);

  v_indices[0] = u32(acc.infinity);
  v_indices[1] = u32(acc.affine);
}

@compute
@workgroup_size(1)
fn load_HighThroughput_test() {
  var v1 = vec4<u32>(0u,1u,2u,3u);
  var v2 = vec4<u32>(4u,5u,6u,7u);
  var v3 = vec4<u32>(8u,9u,10u,11u);
  var v4 = vec4<u32>(0u,1u,2u,3u);
  var v5 = vec4<u32>(4u,5u,6u,7u);
  var v6 = vec4<u32>(8u,9u,10u,100u);
  var v7 = vec4<u32>(0u,1u,2u,3u);
  var v8 = vec4<u32>(4u,5u,6u,7u);
  var v9 = vec4<u32>(8u,9u,10u,11u);
  var v10 = vec4<u32>(0u,1u,2u,3u);
  var v11 = vec4<u32>(4u,5u,6u,7u);
  var v12 = vec4<u32>(8u,9u,10u,200u);
  var arr0 = array<vec4<u32>, 12>(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12);

  var acc: HighThroughput;

  load_HighThroughput(&acc, arr0);

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i] = acc.x[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+12] = acc.y[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+24] = acc.zz[i];
  }

  for (var i : i32 = 0; i < i32(limbs); i = i + 1) {
    v_indices[i+36] = acc.zzz[i];
  }

  v_indices[48] = u32(acc.infinity);
  v_indices[49] = u32(acc.affine);
}

@compute
@workgroup_size(1)
fn add_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);
  var arr1 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  add(&arr, arr0, arr1);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn sub_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 10u,20u,30u,40u,50u,60u,70u,80u,90u,100u,101u);
  var arr1 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  sub(&arr, arr0, arr1);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn addN_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);

  addN(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn add2N_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);

  add2N(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn add3N_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);

  add3N(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn add4N_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);

  add4N(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn add5N_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);

  add5N(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn add6N_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);

  add6N(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn negateAddN_test() {
  var arr = array<u32, limbs>(10u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);

  negateAddN(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn negateAdd4N_test() {
  var arr = array<u32, limbs>(10u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);

  negateAdd4N(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn isZero_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,101u);

  v_indices[0] = u32(isZero(arr));
  v_indices[1] = u32(isZero(arr0));
}

@compute
@workgroup_size(1)
fn loadShared_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);

  loadShared(&arr, 0u);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn setConstant_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);

  setConstant(&arr, 1u);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn setZero_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);

  setZero(&arr);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn setN_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);

  setN(&arr);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn setOne_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);

  setOne(&arr);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn setRSquared_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);

  setRSquared(&arr);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn setPermuteLow_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  setPermuteLow(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn setPermuteHigh_test() {
  var arr = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var arr0 = array<u32, limbs>(0u, 1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u);

  setPermuteHigh(&arr, arr0);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = arr[i];
  }
}

@compute
@workgroup_size(1)
fn sqr1_test() {
  var r: array<u32, limbs>;
  var temp = array<u32, limbs>(1u, 1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u);
  var a = array<u32, limbs>(100u, 100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var n = array<u32, limbs>(1u, 1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u);

  sqr1(&r, &temp, a, n);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = r[i];
  }
}

@compute
@workgroup_size(1)
fn mul1_test() {
  var r: array<u32, limbs>;
  var a = array<u32, limbs>(1u, 1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u);
  var b = array<u32, limbs>(100u, 100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var n = array<u32, limbs>(1u, 1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u);

  mul1(&r, a, b, n);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = r[i];
  }
}

@compute
@workgroup_size(1)
fn mul2_test() {
  var evenOdd1: WideNumber;
  var evenOdd2: WideNumber;
  var evenOdd3: WideNumber;
  var evenOdd4: WideNumber;
  var evenOdd5: WideNumber;
  var evenOdd6: WideNumber;
  var evenOdd7: WideNumber;
  var evenOdd8: WideNumber;
  var evenOdd9: WideNumber;
  var evenOdd10: WideNumber;
  var evenOdd11: WideNumber;
  var evenOdd12: WideNumber;

  var evenOdd = array<WideNumber, limbs>(evenOdd1, evenOdd2, evenOdd3,evenOdd4,evenOdd5,evenOdd6,evenOdd7,evenOdd8,evenOdd9,evenOdd10,evenOdd11,evenOdd12);
  var carry = true;
  var a = array<u32, limbs>(1u, 1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u);
  var b = array<u32, limbs>(100u, 100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var n = array<u32, limbs>(1u, 1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u);

  mul2(&evenOdd, &carry, a, b, n);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = evenOdd[i].first;
    v_indices[i+12] = evenOdd[i].second;
  }
}

@compute
@workgroup_size(1)
fn sqr2_test() { 
  var evenOdd1: WideNumber;
  var evenOdd2: WideNumber;
  var evenOdd3: WideNumber;
  var evenOdd4: WideNumber;
  var evenOdd5: WideNumber;
  var evenOdd6: WideNumber;
  var evenOdd7: WideNumber;
  var evenOdd8: WideNumber;
  var evenOdd9: WideNumber;
  var evenOdd10: WideNumber;
  var evenOdd11: WideNumber;
  var evenOdd12: WideNumber;

  var evenOdd = array<WideNumber, limbs>(evenOdd1, evenOdd2, evenOdd3,evenOdd4,evenOdd5,evenOdd6,evenOdd7,evenOdd8,evenOdd9,evenOdd10,evenOdd11,evenOdd12);
  var temp = array<u32, limbs>(100u, 100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var a = array<u32, limbs>(1u, 1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u);
  var n = array<u32, limbs>(1u, 1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u);
  var carry = true;

  sqr2(&evenOdd, &temp, &carry, a, n);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = evenOdd[i].first;
    v_indices[i+12] = evenOdd[i].second;
  }
}

@compute
@workgroup_size(1)
fn merge_test() {
  var evenOdd1 = WideNumber(100u, 99u);
  var evenOdd2: WideNumber = WideNumber(101u, 102u);
  var evenOdd3: WideNumber = WideNumber(101u, 102u);
  var evenOdd4: WideNumber = WideNumber(101u, 102u);
  var evenOdd5: WideNumber = WideNumber(101u, 102u);
  var evenOdd6: WideNumber = WideNumber(101u, 102u);
  var evenOdd7: WideNumber = WideNumber(101u, 102u);
  var evenOdd8: WideNumber;
  var evenOdd9: WideNumber;
  var evenOdd10: WideNumber;
  var evenOdd11: WideNumber;
  var evenOdd12: WideNumber;

  var evenOdd = array<WideNumber, limbs>(evenOdd1, evenOdd2, evenOdd3,evenOdd4,evenOdd5,evenOdd6,evenOdd7,evenOdd8,evenOdd9,evenOdd10,evenOdd11,evenOdd12);
  var r = array<u32, limbs>(0u, 0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var carry = true;

  merge(&r, evenOdd, carry);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = r[i];
  }
}

@compute
@workgroup_size(1)
fn add1_HighThroughput_test() {
  var x = array<u32, limbs>(100u, 100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var y = array<u32, limbs>(99u, 99u,99u,99u,99u,99u,99u,99u,99u,99u,99u,99u);
  var point = PointXY(x, y);

  var acc = HighThroughput(x, x, x, x, true, true);

  add1_HighThroughput(&acc, &point, false);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = acc.x[i];
    v_indices[i+12] = acc.y[i];
    v_indices[i+24] = acc.zz[i];
    v_indices[i+36] = acc.zzz[i];
  }
}

@compute
@workgroup_size(1)
fn add2_HighThroughput_test() {
  var x = array<u32, limbs>(100u, 100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var y = array<u32, limbs>(99u, 99u,99u,99u,99u,99u,99u,99u,99u,99u,99u,99u);
  var point = PointXYZZ(x, y, x, y);

  //var acc: HighThroughput;
  var acc = HighThroughput(x, x, x, x, true, true);

  add2_HighThroughput(&acc, &point, false);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = acc.x[i];
    v_indices[i+12] = acc.y[i];
    v_indices[i+24] = acc.zz[i];
    v_indices[i+36] = acc.zzz[i];
  }
}

@compute
@workgroup_size(1)
fn dbl_test() {
  var x = array<u32, limbs>(100u, 100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var y = array<u32, limbs>(99u, 99u,99u,99u,99u,99u,99u,99u,99u,99u,99u,99u);

  var acc = HighThroughput(x, x, x, x, true, true);

  dbl(&acc);

  for (var i : i32 = 0; i < i32(limbs); i++) {
    v_indices[i] = acc.x[i];
    v_indices[i+12] = acc.y[i];
    v_indices[i+24] = acc.zz[i];
    v_indices[i+36] = acc.zzz[i];
  }
}

@compute
@workgroup_size(1)
fn accumulator_test() {
  var x = array<u32, limbs>(100u, 100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var y = array<u32, limbs>(99u, 99u,99u,99u,99u,99u,99u,99u,99u,99u,99u,99u);

  var acc = HighThroughput(x, y, x, y, false, true);

  accumulator(&acc);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = acc.x[i];
    v_indices[i+12] = acc.y[i];
    v_indices[i+24] = acc.zz[i];
    v_indices[i+36] = acc.zzz[i];
  }
}

@compute
@workgroup_size(1)
fn normalize_test() {
  var x = array<u32, limbs>(100u, 100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var y = array<u32, limbs>(99u, 99u,99u,99u,99u,99u,99u,99u,99u,99u,99u,99u);

  var acc = HighThroughput(x, y, x, y, true, true);

  normalize(&acc);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = acc.x[i];
    v_indices[i+12] = acc.y[i];
    v_indices[i+24] = acc.zz[i];
    v_indices[i+36] = acc.zzz[i];
  }
}

@compute
@workgroup_size(1)
fn fromInternal_HighThroughput_test() {
  var x = array<u32, limbs>(100u, 100u,100u,100u,100u,100u,100u,100u,100u,100u,100u,100u);
  var y = array<u32, limbs>(99u, 99u,99u,99u,99u,99u,99u,99u,99u,99u,99u,99u);

  var acc = HighThroughput(x, y, x, y, true, true);

  fromInternal_HighThroughput(&acc);

  for (var i : i32 = 0; i < i32(limbs); i++ ) {
    v_indices[i] = acc.x[i];
    v_indices[i+12] = acc.y[i];
    v_indices[i+24] = acc.zz[i];
    v_indices[i+36] = acc.zzz[i];
  }
}