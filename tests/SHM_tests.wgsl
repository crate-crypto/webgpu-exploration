@compute
@workgroup_size(1)
fn copyToShared_test() {
  var vector = vec4<u32>(0u,1u,2u,3u);
  var arr = array<vec4<u32>, 96>(
    vector,vector,vector,vector,vector,vector,vector,vector,vector,vector,
    vector,vector,vector,vector,vector,vector,vector,vector,vector,vector,
    vector,vector,vector,vector,vector,vector,vector,vector,vector,vector,
    vector,vector,vector,vector,vector,vector,vector,vector,vector,vector,
    vector,vector,vector,vector,vector,vector,vector,vector,vector,vector,
    vector,vector,vector,vector,vector,vector,vector,vector,vector,vector,
    vector,vector,vector,vector,vector,vector,vector,vector,vector,vector,
    vector,vector,vector,vector,vector,vector,vector,vector,vector,vector,
    vector,vector,vector,vector,vector,vector,vector,vector,vector,vector,
    vector,vector,vector,vector,vector,vector);
  var vector2 = vec2<u32>(1u, 1u);

  var thread: Thread = Thread(vector2, vector2, vector2, vector2, 32u);

  copyToShared(arr, thread);

  for (var i = 1u; i < 300u; i = i + 1u) {
    v_indices[i] = memory.data[i];
  }
}
