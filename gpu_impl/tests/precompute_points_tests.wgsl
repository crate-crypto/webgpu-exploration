@compute
@workgroup_size(1)
fn precomputePointsKernel_test() {
  var pointsPtr: array<vec4<u32>, 6>;
  var affinePointsPtr = array<u32, 24>(0u, 1u,2u,3u,4u,5u,6u,70u,8u,9u,10u,11u,0u, 1u,2u,3u,4u,5u,6u,70u,8u,9u,10u,11u);
  var pointCount = 100u;

  var thread: Thread = Thread(
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    vec2<u32>(1u, 2u),
    1u,
  );
  
  precomputePointsKernel(&pointsPtr, affinePointsPtr, pointCount, thread);


  for(var i = 0; i < 6; i+=4) {
    v_indices[i] = pointsPtr[i].x;
    v_indices[i+1] = pointsPtr[i].y;
    v_indices[i+2] = pointsPtr[i].z;
    v_indices[i+3] = pointsPtr[i].w;
  }
}